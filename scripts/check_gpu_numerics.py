"""Small hardware gate for the repaired fractional objective; no quality inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from laplacian.losses import spectral_penalties
from laplacian.operators_torch import fractional_matrix, fractional_policy
from laplacian.runtime import write_metrics


# Fixed engineering admission limits for these small fixtures, selected before
# new hardware execution. They are not a measured error envelope or a training
# equivalence claim. No automatic tolerance adjustment is permitted.
REFERENCE_TOLERANCES = {
    "fp32": {"loss_atol": 1e-5, "loss_rtol": 1e-4, "gradient_atol": 1e-5, "gradient_rtol": 1e-3},
    "bf16_autocast": {"loss_atol": 1e-3, "loss_rtol": 2e-2, "gradient_atol": 1e-3, "gradient_rtol": 5e-2},
}


def require_gpu():
    ledger = os.environ.get("STUDY_BUDGET_LEDGER")
    if (not os.environ.get("STUDY_BUDGET_RUN_ID", "").strip() or not ledger
            or Path(ledger).resolve() != (ROOT / "runs/publication_study/budget.jsonl").resolve()):
        raise RuntimeError("Gate requires the canonical study budget wrapper and run ID before CUDA access")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Gate requires the approved RTX 5070 as visible CUDA device zero")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The approved device must support BF16 autocast")
    return device


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "src/laplacian/operators_torch.py", ROOT / "src/laplacian/losses.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def cpu_reference(data, exponent):
    """All-FP64 CPU affinity/quadratic; shares only the versioned matrix-power operator.

    Calling spectral_penalties with double inputs would not be a double oracle:
    it explicitly converts pooled states, affinities and quadratics to FP32.
    The separate repeated-eigenvalue cases check the shared operator analytically.
    """
    states = data.detach().to(device="cpu", dtype=torch.float64).clone().requires_grad_()
    with torch.autocast("cpu", enabled=False):
        distance = torch.cdist(states, states)
        affinity = torch.exp(-distance.square() / 2)
        inv = affinity.sum(dim=1).rsqrt()
        lap = torch.eye(states.shape[0], dtype=torch.float64) - inv[:, None] * affinity * inv[None, :]
        powered = fractional_matrix(lap, exponent)
        loss = .5 * (states * (powered @ states)).sum()
    gradient = torch.autograd.grad(loss, states)[0]
    return loss.detach(), gradient.detach()


def compare_reference(loss, gradient, reference_loss, reference_gradient, autocast):
    tolerance = dict(REFERENCE_TOLERANCES["bf16_autocast" if autocast else "fp32"])
    loss, gradient, reference_loss, reference_gradient = [value.detach().to(device="cpu", dtype=torch.float64)
        for value in (loss, gradient, reference_loss, reference_gradient)]
    finite = all(bool(torch.isfinite(value).all()) for value in (loss, gradient, reference_loss, reference_gradient))
    shape_match = loss.shape == reference_loss.shape and gradient.shape == reference_gradient.shape
    result = {"tolerances": tolerance, "finite": finite, "shape_match": shape_match,
              "passed": False, "loss_abs_error": None, "gradient_max_abs_error": None,
              "gradient_relative_l2_error": None}
    if not finite or not shape_match:
        return result
    loss_error = (loss-reference_loss).abs()
    gradient_error = (gradient-reference_gradient).abs()
    result.update(loss_abs_error=float(loss_error), gradient_max_abs_error=float(gradient_error.max()),
                  reference_loss=float(reference_loss), reference_gradient_l2=float(reference_gradient.norm()))
    if result["reference_gradient_l2"] > 0:
        result["gradient_relative_l2_error"] = float(gradient_error.norm()) / result["reference_gradient_l2"]
    result["loss_passed"] = bool((loss_error <= tolerance["loss_atol"] + tolerance["loss_rtol"] * reference_loss.abs()).all())
    result["gradient_passed"] = bool((gradient_error <= tolerance["gradient_atol"] + tolerance["gradient_rtol"] * reference_gradient.abs()).all())
    result["passed"] = result["loss_passed"] and result["gradient_passed"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = {"complete": False, "status": "running", "run_id": args.output.name, "host_gpu": False,
               "jovin_receipt": False,
               "scope": "Small fractional numerical hardware checks; no training or quality conclusion",
               "torch": str(torch.__version__), "cuda": torch.version.cuda, "fractional_policy": fractional_policy(),
               "budget": {key: os.environ.get(key) for key in ("STUDY_BUDGET_RUN_ID", "STUDY_BUDGET_LEDGER")},
               "reference_policy": "CPU FP64 affinity/quadratic and shared versioned fractional operator, autocast disabled",
               "comparison_policy": "Every loss/gradient element: abs(candidate-reference) <= atol + rtol*abs(reference)",
               "reference_tolerances": REFERENCE_TOLERANCES, "cases": []}
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        hashes = source_hashes()
        payload["source_sha256"] = hashes
        device = require_gpu()
        payload.update(host_gpu=True, device=torch.cuda.get_device_name(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        for exponent in (0.25, 0.5, 0.75, 1.0):
            for name, data in (("separated", torch.eye(4) * 100), ("zero", torch.zeros(4, 4)),
                               ("constant", torch.ones(4, 4)), ("duplicate", torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]))):
                reference_loss, reference_gradient = cpu_reference(data, exponent)
                for autocast in (False, True):
                    states = data.to(device).requires_grad_()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        loss = spectral_penalties({"pooled": [states]}, [0], exponent,
                            weights={"fractional": 1., "manifold": 0., "hyper": 0., "dirichlet": 0.})["fractional"]
                    gradient = torch.autograd.grad(loss, states)[0]
                    comparison = compare_reference(loss, gradient, reference_loss, reference_gradient, autocast)
                    finite = comparison["finite"]
                    payload["cases"].append({"fixture": name, "exponent": exponent,
                        "bf16_autocast": autocast, "loss": float(loss.detach()) if finite else None,
                        "gradient_norm": float(gradient.norm()) if finite else None, "finite": finite,
                        "cpu_fp64_comparison": comparison})
                    if not finite:
                        raise RuntimeError(f"nonfinite {name}/{exponent}/{autocast}")
                    if not comparison["passed"]:
                        raise RuntimeError(f"CPU FP64 reference mismatch {name}/{exponent}/{autocast}")
            # Analytic repeated-eigenvalue derivative, including the zero matrix.
            for eigenvalue in (0., 0.3):
                matrix = (torch.eye(4, dtype=torch.float64, device=device) * eigenvalue).requires_grad_()
                weight = torch.arange(16, dtype=torch.float64, device=device).reshape(4, 4) / 16
                result = fractional_matrix(matrix, exponent)
                gradient = torch.autograd.grad((result * weight).sum(), matrix)[0]
                slope = (2-exponent) * 1e-4 ** (exponent-1) if eigenvalue == 0 else exponent * eigenvalue ** (exponent-1)
                expected = slope * (weight + weight.T) / 2
                torch.testing.assert_close(gradient, expected, atol=1e-9, rtol=1e-9)
                payload["cases"].append({"fixture": "analytic_repeated", "eigenvalue": eigenvalue,
                    "exponent": exponent, "max_abs_error": float((gradient-expected).abs().max()), "finite": True})
        if hashes != source_hashes():
            raise RuntimeError("source changed during hardware gate")
        payload["complete"] = True
        payload["status"] = "passed"
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_metrics(args.output / "metrics.json", payload)
    print(json.dumps({"complete": True, "cases": len(payload["cases"])}))


if __name__ == "__main__":
    main()
