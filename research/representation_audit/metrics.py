"""CPU-only descriptive diagnostics for frozen model outputs.

This sidecar does not change a loss, train a model, rank task semantics, or
establish a causal mechanism. Numeric host objectives use their frozen code.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from laplacian import losses
from laplacian.operators_torch import fractional_matrix, fractional_policy

SCHEMA = "representation_metrics_v1"


def _validate(outputs):
    keys = ("pooled", "stream", "hidden", "attn")
    count = len(outputs["pooled"])
    if not count or any(len(outputs[key]) != count for key in keys):
        raise ValueError("Nonempty equal layer counts required")
    tensors = [outputs["embedded"], *(item for key in keys for item in outputs[key])]
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("Diagnostics accept CPU tensors only")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Diagnostics require finite floating tensors")
    embedded = outputs["embedded"]
    if embedded.ndim != 3 or min(embedded.shape) < 1:
        raise ValueError("Embedded tensor must be nonempty B,T,D")
    b, t, d = embedded.shape
    for pooled, stream, hidden, attention in zip(*(outputs[k] for k in keys)):
        if pooled.shape != (b, d) or stream.shape != embedded.shape:
            raise ValueError("Pooled/stream dimensions differ from embeddings")
        if hidden.ndim != 3 or hidden.shape[:2] != (b, t) or hidden.shape[2] < 1:
            raise ValueError("Hidden tensor must be B,T,N")
        if attention.ndim != 4 or attention.shape[0] != b or attention.shape[2:] != (t, t) or attention.shape[1] < 1:
            raise ValueError("Attention tensor must be B,H,T,T")


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite diagnostic output")
    return number


def _stats(values):
    if values.numel() == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {"count": values.numel(), "min": _finite(values.min()),
            "max": _finite(values.max()), "mean": _finite(values.mean())}


def _pooled(states, exponent, tau):
    centered = states.double() - states.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    mass = singular.square()
    total = mass.sum()
    if total > 0:
        p = mass[mass > 0] / total
        rank = _finite(torch.exp(-(p * p.log()).sum()))
    else:
        rank = 0.0
    points = states.float()
    distances = torch.cdist(points, points, p=2)
    affinity = torch.exp(-distances.square() / (2 * tau * tau))
    off_diagonal = affinity[~torch.eye(len(points), dtype=torch.bool)]
    statistics = _stats(off_diagonal)
    statistics.update({"tau": tau, "self_loops_retained": True,
                       "near_zero_fraction": _finite((off_diagonal <= 1e-6).float().mean()) if off_diagonal.numel() else None,
                       "near_one_fraction": _finite((off_diagonal >= 1 - 1e-6).float().mean()) if off_diagonal.numel() else None,
                       "saturation_cutoff": 1e-6})
    lap = losses._affinity(points, tau)
    eigenvalues = torch.linalg.eigvalsh((lap.double() + lap.double().T) / 2)
    powered = fractional_matrix(lap.double(), exponent).float()
    return {"pooled_variance": _finite(centered.square().mean()),
            "effective_rank": rank, "effective_rank_definition": "exp(entropy(centered singular-value-squared mass)); zero mass -> 0",
            "affinity": statistics,
            "normalized_spectrum": {"eigenvalues": [_finite(x) for x in eigenvalues],
                                    "second_eigenvalue": max(0.0, _finite(eigenvalues[1])) if len(points) > 1 else None,
                                    "nullity_at_1e_6": int((eigenvalues.abs() <= 1e-6).sum()),
                                    "definition": "Sorted eigenvalue index 1, counting nulls; not smallest strictly positive eigenvalue"},
            "fractional_energy": _finite(.5 * losses._quadratic(powered, points)),
            "fractional_exponent": exponent,
            "fractional_numeric_scope": "Current C1 policy on an FP32-built affinity; null-mode roundoff can change energy under otherwise invariant transformations"}


def _band(embedded, states, attention, c1, c2):
    affinity = attention.float().mean(dim=1)
    energies, base = [], []
    for index in range(len(affinity)):
        energy, adjacency = losses._token_energy(states[index], affinity[index])
        base_energy, _ = losses._token_energy(embedded[index], affinity[index])
        energies.append(energy)
        base.append(base_energy)
        if index == 0:
            adjacency0 = adjacency
    mean_energy, mean_base = torch.stack(energies).mean(), torch.stack(base).mean()
    sigma2 = embedded.float().var(unbiased=False).clamp_min(1e-8)
    estimate = losses._lambda2(adjacency0)
    lower = c1 * sigma2 * estimate
    upper = torch.maximum(c2 * mean_base, lower + 1e-4)
    low_gap, high_gap = (lower - mean_energy).clamp_min(0), (mean_energy - upper).clamp_min(0)
    return {"energy": _finite(mean_energy), "base_energy": _finite(mean_base),
            "embedding_variance_clamped": _finite(sigma2), "lower": _finite(lower), "upper": _finite(upper),
            "lower_active": bool(low_gap > 0), "upper_active": bool(high_gap > 0),
            "penalty": _finite(low_gap.square() + high_gap.square()), "lambda2_estimate": _finite(estimate),
            "spectral_estimate_scope": "Frozen detached heuristic on first example only; no residual/eigengap acceptance gate",
            "energy_scope": "Batch-mean combinatorial energy divided by positive undirected edge count",
            "host_precision": "FP32 CPU, outside autocast; not GPU mixed-precision attestation",
            "c1": c1, "c2": c2}


@torch.no_grad()
def analyze(outputs, selected_layers=(1, 4), *, exponent=.5, tau=1.0, c1=1.0, c2=4.0):
    """Return JSON-compatible per-layer observations from one fixed microbatch.

    All model layers are described; selected_layers labels the existing manifold
    subset only. No predictions about downstream quality are inferred.
    """
    if not math.isfinite(tau) or tau <= 0 or not 0 < exponent <= 1:
        raise ValueError("Positive finite tau and exponent in (0,1] required")
    if not all(math.isfinite(x) and x >= 0 for x in (c1, c2)):
        raise ValueError("Finite nonnegative band coefficients required")
    _validate(outputs)
    rows = []
    for layer, states in enumerate(outputs["pooled"]):
        row = {"layer": layer, "selected": layer in selected_layers, **_pooled(states, exponent, tau)}
        row["incidence_density"] = _finite((outputs["hidden"][layer] > 0).float().mean())
        row["incidence_definition"] = "Positive activations in the declared neuron coordinate system; fixed support per region"
        row["token_band"] = _band(outputs["embedded"], outputs["stream"][layer], outputs["attn"][layer], c1, c2)
        rows.append(row)
    return {"schema": SCHEMA, "device": "cpu", "fractional_policy": fractional_policy(), "layers": rows,
            "scope": "Descriptive training-window diagnostics, not held-out quality, causal mechanism or spectral regularization benefit"}
