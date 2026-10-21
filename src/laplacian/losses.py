"""Cross-entropy plus the four spectral penalties.

Skipped terms stay at zero when their schedule weight is zero.
"""

from __future__ import annotations

import torch

from laplacian.culling import fiedler_from_adjacency
from laplacian.operators_torch import fractional_matrix


def _affinity(points, tau):
    dist = torch.cdist(points.float(), points.float(), p=2)
    affinity = torch.exp(-(dist * dist) / (2.0 * tau * tau))
    inv = torch.zeros(points.shape[0], device=points.device, dtype=torch.float32)
    degree = affinity.sum(dim=1)
    positive = degree > 0
    inv[positive] = degree[positive].rsqrt()
    normal = inv[:, None] * affinity * inv[None, :]
    eye = torch.eye(points.shape[0], device=points.device, dtype=torch.float32)
    return (eye - normal) * positive.to(torch.float32)[:, None]


def _quadratic(lap, features):
    features = features.float()
    return (features * (lap @ features)).sum()


@torch.no_grad()
def _lambda2(adjacency, steps=30):
    """Smallest nonzero eigenvalue of the normalized Laplacian.

    Plain power iteration on L returns the spectral radius. The Dirichlet
    band needs λ2, so this uses the same ||L|| I - L shift as the cull.
    The existing scalar estimate is detached; avoid constructing a discarded
    autograd graph through its power iterations.
    """
    _vector, rayleigh, _degenerate = fiedler_from_adjacency(adjacency.float(), steps=steps)
    return torch.tensor(rayleigh, device=adjacency.device, dtype=torch.float32).clamp_min(0)


def _token_energy(states, affinity):
    """Dirichlet energy of token states on one symmetrized affinity. states: [T, D]."""
    adjacency = 0.5 * (affinity + affinity.transpose(0, 1))
    adjacency = adjacency - torch.diag(torch.diag(adjacency))
    adjacency = adjacency.clamp_min(0)
    degree = adjacency.sum(dim=1)
    features = states.float()
    quad = (features.pow(2) * degree[:, None]).sum() - (features * (adjacency @ features)).sum()
    edges = (torch.triu(adjacency, diagonal=1) > 0).sum().clamp_min(1)
    return quad / edges, adjacency


def hypergraph_energy(activations):
    """The existing degree-scaled co-activation energy, with bounded workspace.

    For scaled incidence S and scaled features X, the correction is
    Tr(X^T S D_e^-1 S^T X). Associate through the smaller of vertices
    and events so the square intermediate is only min(N, E) by min(N, E).
    This preserves the existing extra vertex scaling of the features.
    """
    flat = activations.float().reshape(-1, activations.shape[-1]).transpose(0, 1)
    incidence = (flat > 0).to(torch.float32)
    edge_degree = incidence.sum(dim=0).clamp_min(1.0)
    vertex_degree = incidence.sum(dim=1)
    inv = torch.zeros_like(vertex_degree)
    positive = vertex_degree > 0
    inv[positive] = vertex_degree[positive].rsqrt()
    scaled = incidence * inv[:, None]
    scaled_feat = flat * inv[:, None]
    if flat.shape[0] < flat.shape[1]:
        propagation = (scaled / edge_degree[None, :]) @ scaled.transpose(0, 1)
        correction = (scaled_feat * (propagation @ scaled_feat)).sum()
    else:
        middle = (scaled.transpose(0, 1) @ scaled_feat).float()
        correction = (middle.square() / edge_degree[:, None]).sum()
    energy = scaled_feat.square().sum() - correction
    return energy.clamp_min(0)


def spectral_penalties(output, layers, exponent, tau=1.0, c1=1.0, c2=4.0, weights=None):
    """Unweighted penalties, optionally skipping terms with zero schedule weight.

    Omit weights for diagnostics that need every penalty. Skipped penalties
    are scalar zeros; token_energies is empty when Dirichlet is skipped.
    """
    pooled = [tensor.float() for tensor in output["pooled"]]

    def enabled(name):
        return weights is None or weights[name] != 0.0

    manifold = pooled[0].new_zeros(())
    if enabled("manifold"):
        chosen = [pooled[index] for index in layers]
        pair_count = 0
        for source in chosen:
            lap = _affinity(source, tau)
            for target in chosen:
                count = target.shape[0]
                manifold = manifold + _quadratic(lap, target) / (count * count)
                pair_count += 1
        if pair_count:
            manifold = manifold / pair_count

    fractional = pooled[0].new_zeros(())
    if enabled("fractional"):
        for states in pooled:
            lap = _affinity(states, tau)
            powered = fractional_matrix(lap.double(), exponent).float()
            fractional = fractional + 0.5 * _quadratic(powered, states)

    hyper = pooled[0].new_zeros(())
    if enabled("hyper"):
        for hidden in output["hidden"]:
            hyper = hyper + hypergraph_energy(hidden)

    energies = []
    e0_values = []
    if enabled("dirichlet"):
        embedded = output["embedded"].float()
        sigma2 = embedded.var(unbiased=False).clamp_min(1e-8)
        for layer_index, attention in enumerate(output["attn"]):
            affinity = attention.float().mean(dim=1)
            states = output["stream"][layer_index].float()
            layer_energy = []
            base_energy = []
            for batch_index in range(affinity.shape[0]):
                energy, adjacency = _token_energy(states[batch_index], affinity[batch_index])
                base, _base_adjacency = _token_energy(embedded[batch_index], affinity[batch_index])
                layer_energy.append(energy)
                base_energy.append(base)
                if batch_index == 0:
                    adjacency0 = adjacency
                    energies.append(float(energy.detach().item()))
            mean_energy = torch.stack(layer_energy).mean()
            mean_base = torch.stack(base_energy).mean()
            lam = _lambda2(adjacency0)
            e_min = c1 * sigma2 * lam
            e_max = torch.maximum(c2 * mean_base, e_min + 1e-4)
            gap_low = (e_min - mean_energy).clamp_min(0)
            gap_high = (mean_energy - e_max).clamp_min(0)
            e0_values.append(gap_low.square() + gap_high.square())
    dirichlet = torch.stack(e0_values).sum() if e0_values else pooled[0].new_zeros(())
    return {
        "manifold": manifold,
        "fractional": fractional,
        "hyper": hyper,
        "dirichlet": dirichlet,
        "token_energies": energies,
    }


def total_loss(cross_entropy, penalties, weights):
    total = cross_entropy
    total = total + weights["manifold"] * penalties["manifold"]
    total = total + weights["fractional"] * penalties["fractional"]
    total = total + weights["hyper"] * penalties["hyper"]
    total = total + weights["dirichlet"] * penalties["dirichlet"]
    return total
