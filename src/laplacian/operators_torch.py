"""PyTorch counterparts of the receipted reference operators.

Toy fixtures are computed in float64. Integer nilpotency stays exact.
The fractional operator uses the versioned boundary policy documented in
paper/planning/FRACTIONAL_POLICY.md; its smoothing is not a reference receipt.
"""

from __future__ import annotations

import math

import torch


def _tensor(matrix, dtype=torch.float64):
    return torch.tensor(matrix, dtype=dtype)


def nilpotency_l1(b_k, b_next):
    """L1 of B_k @ B_{k+1}. An empty factor is vacuously zero."""
    if not b_k or not b_next or not b_next[0]:
        return 0
    left = torch.tensor(b_k, dtype=torch.int64)
    right = torch.tensor(b_next, dtype=torch.int64)
    if left.numel() == 0 or right.numel() == 0 or right.shape[-1] == 0:
        return 0
    return int(torch.matmul(left, right).abs().sum().item())


def combinatorial_laplacian(n, edges, dtype=torch.float64, device="cpu"):
    lap = torch.zeros((n, n), dtype=dtype, device=device)
    for i, j, weight in edges:
        weight = float(weight)
        lap[i, i] += weight
        lap[j, j] += weight
        lap[i, j] -= weight
        lap[j, i] -= weight
    return lap


def normalized_laplacian(adjacency):
    adjacency = _tensor(adjacency) if not torch.is_tensor(adjacency) else adjacency.to(torch.float64)
    degree = adjacency.sum(dim=1)
    inv = torch.zeros_like(degree)
    positive = degree > 0
    inv[positive] = degree[positive].rsqrt()
    normal = inv[:, None] * adjacency * inv[None, :]
    eye = torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
    present = positive.to(adjacency.dtype)
    return (eye - normal) * present[:, None]


def quadratic_form(lap, features):
    lap = lap if torch.is_tensor(lap) else _tensor(lap)
    feat = features if torch.is_tensor(features) else _tensor(features)
    lap = lap.to(dtype=feat.dtype)
    mixed = feat.transpose(0, 1) @ lap @ feat
    return float(torch.trace(mixed).item()) if mixed.ndim == 2 else float(mixed.item())


def _edge_count(lap):
    count = int(((torch.triu(lap, diagonal=1) < 0).sum()).item())
    return max(count, 1)


def dirichlet_energy(lap, features):
    lap = lap if torch.is_tensor(lap) else _tensor(lap)
    return quadratic_form(lap, features) / _edge_count(lap)


def kernel_residual(lap):
    lap = lap if torch.is_tensor(lap) else _tensor(lap)
    return float(lap.sum(dim=1).abs().max().item())


def band_penalty(energies, e_min, e_max):
    total = 0.0
    for energy in energies:
        low = e_min - float(energy)
        high = float(energy) - e_max
        if low > 0.0:
            total += low * low
        if high > 0.0:
            total += high * high
    return total


def affinity_laplacian(points, tau):
    pts = points if torch.is_tensor(points) else _tensor(points)
    dist = torch.cdist(pts, pts, p=2)
    affinity = torch.exp(-(dist * dist) / (2.0 * tau * tau))
    degree = affinity.sum(dim=1)
    scale = degree.sqrt().clamp_min(1e-15)
    normal = affinity / (scale[:, None] * scale[None, :])
    eye = torch.eye(pts.shape[0], dtype=pts.dtype, device=pts.device)
    return eye - normal


def manifold_penalty(features, lap_k):
    feat = features if torch.is_tensor(features) else _tensor(features)
    lap = lap_k if torch.is_tensor(lap_k) else _tensor(lap_k)
    count = feat.shape[0]
    form = quadratic_form(lap, feat)
    return form / (count * count)


FRACTIONAL_POLICY_VERSION = "c1-quadratic-cap-frechet-v1"
FRACTIONAL_CUTOFF = 1e-4
FRACTIONAL_NEGATIVE_TOLERANCE = 1e-6


def fractional_policy():
    """Serializable identity for future runs sharing the repaired operator."""
    return {
        "version": FRACTIONAL_POLICY_VERSION,
        "cutoff": FRACTIONAL_CUTOFF,
        "negative_spectrum_relative_tolerance": FRACTIONAL_NEGATIVE_TOLERANCE,
        "exponent_domain": "0 < s <= 1",
        "gradient_order": 1,
    }


def _fractional_values(values, exponent, cutoff):
    """C1 scalar power: exact above cutoff, quadratic below, tangent below zero."""
    slope = (2.0 - exponent) * cutoff ** (exponent - 1.0)
    curvature = (exponent - 1.0) * cutoff ** (exponent - 2.0)
    small = values.clamp(0, cutoff)
    polynomial = slope * small + curvature * small.square()
    powered = values.clamp_min(cutoff).pow(exponent)
    return torch.where(values < 0, slope * values, torch.where(values < cutoff, polynomial, powered))


def _fractional_divided_difference(values, exponent, cutoff):
    """Stable secants, including the analytic derivative on repeated eigenvalues.

    Integrate the derivative over each scalar branch instead of subtracting
    nearly equal function values. This also handles pairs straddling a join.
    """
    left = torch.minimum(values[:, None], values[None, :])
    right = torch.maximum(values[:, None], values[None, :])
    gap = right - left
    slope = (2.0 - exponent) * cutoff ** (exponent - 1.0)
    curvature = (exponent - 1.0) * cutoff ** (exponent - 2.0)

    negative = slope * (right.clamp_max(0) - left.clamp_max(0))
    small_left, small_right = left.clamp(0, cutoff), right.clamp(0, cutoff)
    polynomial = (small_right - small_left) * (slope + curvature * (small_left + small_right))
    large_left, large_right = left.clamp_min(cutoff), right.clamp_min(cutoff)
    relative_gap = (large_right - large_left) / large_left
    powered = large_left.pow(exponent) * torch.expm1(exponent * torch.log1p(relative_gap))

    secant = (negative + polynomial + powered) / torch.where(gap > 0, gap, torch.ones_like(gap))
    derivative = torch.where(
        left < 0,
        torch.full_like(left, slope),
        torch.where(left < cutoff, slope + 2.0 * curvature * left, exponent * left.clamp_min(cutoff).pow(exponent - 1.0)),
    )
    return torch.where(gap > 0, secant, derivative)


class _FractionalMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lap, exponent, cutoff):
        # Autocast must not turn the spectral reconstruction/backward into BF16.
        with torch.autocast(device_type=lap.device.type, enabled=False):
            if not bool(torch.isfinite(lap).all()):
                raise ValueError("fractional_matrix requires a finite matrix")
            values, vectors = torch.linalg.eigh(lap)
            scale = values.abs().amax().clamp_min(1.0)
            if bool((values < -FRACTIONAL_NEGATIVE_TOLERANCE * scale).any()):
                raise ValueError("fractional_matrix requires a positive semidefinite matrix within the documented roundoff tolerance")
            ctx.exponent, ctx.cutoff = exponent, cutoff
            ctx.save_for_backward(values, vectors)
            if exponent == 1.0:
                return lap.clone()
            powered = _fractional_values(values, exponent, cutoff)
            return (vectors * powered) @ vectors.transpose(0, 1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        values, vectors = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            gradient = 0.5 * (grad_output + grad_output.transpose(0, 1))
            if ctx.exponent == 1.0:
                return gradient, None, None
            divided = _fractional_divided_difference(values, ctx.exponent, ctx.cutoff)
            local = vectors.transpose(0, 1) @ gradient @ vectors
            result = vectors @ (divided * local) @ vectors.transpose(0, 1)
            return 0.5 * (result + result.transpose(0, 1)), None, None


def fractional_matrix(lap, exponent, *, cutoff=FRACTIONAL_CUTOFF):
    """C1-regularized symmetric PSD matrix power with a stable first derivative.

    This is a versioned objective change near zero, not a claim that the
    unregularized fractional power has a finite derivative there. Values at
    eigenvalues >= cutoff are unchanged. The matrix is explicitly symmetrized;
    materially negative spectra are rejected. Half/BF16 inputs promote to FP32.
    The exponent/cutoff are fixed hyperparameters; double backward is unsupported.
    """
    exponent, cutoff = float(exponent), float(cutoff)
    if not math.isfinite(exponent) or not 0.0 < exponent <= 1.0:
        raise ValueError("fractional_matrix requires 0 < exponent <= 1")
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("fractional_matrix cutoff must be finite and positive")
    lap = lap if torch.is_tensor(lap) else _tensor(lap)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1] or lap.shape[0] == 0:
        raise ValueError("fractional_matrix requires a nonempty square matrix")
    if not lap.is_floating_point():
        raise TypeError("fractional_matrix requires a real floating-point matrix")
    with torch.autocast(device_type=lap.device.type, enabled=False):
        if lap.dtype not in (torch.float32, torch.float64):
            lap = lap.float()
        symmetric = 0.5 * (lap + lap.transpose(0, 1))
        return _FractionalMatrix.apply(symmetric, exponent, cutoff)


def fractional_penalty(lap, features, exponent, lam=1.0):
    feat = features if torch.is_tensor(features) else _tensor(features)
    powered = fractional_matrix(lap, exponent)
    return (float(lam) / 2.0) * quadratic_form(powered, feat)


def hypergraph_laplacian(incidence, edge_weights):
    incidence = incidence if torch.is_tensor(incidence) else _tensor(incidence)
    weights = edge_weights if torch.is_tensor(edge_weights) else _tensor(edge_weights)
    edge_degree = incidence.sum(dim=0).clamp_min(1e-15)
    vertex_degree = (incidence * weights).sum(dim=1)
    inv_sqrt = torch.zeros_like(vertex_degree)
    positive = vertex_degree > 0
    inv_sqrt[positive] = vertex_degree[positive].rsqrt()
    factor = weights / edge_degree
    scaled = incidence * inv_sqrt[:, None]
    mixed = (scaled * factor) @ scaled.transpose(0, 1)
    eye = torch.eye(incidence.shape[0], dtype=incidence.dtype, device=incidence.device)
    return eye - mixed


def _reduced_solve(lap, rhs):
    """Solve L x = rhs on the orthogonal complement of the constants."""
    mean = rhs.mean()
    shifted = rhs - mean
    last = lap[:-1, -1]
    reduced = lap[:-1, :-1] - last[:, None]
    head = torch.linalg.solve(reduced, shifted[:-1])
    tail = -head.sum()
    return torch.cat([head, tail.view(1)])


def effective_resistance(lap, i, j):
    lap = lap if torch.is_tensor(lap) else _tensor(lap)
    if i == j:
        return 0.0
    rhs = torch.zeros(lap.shape[0], dtype=lap.dtype, device=lap.device)
    rhs[i] = 1.0
    rhs[j] = -1.0
    potential = _reduced_solve(lap, rhs)
    return float((potential[i] - potential[j]).item())


def boundary_schur(lap, boundary):
    lap = lap if torch.is_tensor(lap) else _tensor(lap)
    boundary = list(boundary)
    chosen = set(boundary)
    interior = [index for index in range(lap.shape[0]) if index not in chosen]
    block_bb = lap[boundary][:, boundary]
    if not interior:
        return block_bb
    block_ii = lap[interior][:, interior]
    block_ib = lap[interior][:, boundary]
    block_bi = lap[boundary][:, interior]
    solved = torch.linalg.solve(block_ii, block_ib)
    return block_bb - block_bi @ solved


def path_adjacency(n, dtype=torch.float64):
    adjacency = torch.zeros((n, n), dtype=dtype)
    for i in range(n - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    return adjacency


def fiedler_aei(adjacency, steps=80):
    """Projected power iteration. `adjacency` is a tensor or a nested list."""
    if not torch.is_tensor(adjacency):
        adjacency = _tensor(adjacency)
    adjacency = adjacency.to(torch.float64)
    n = adjacency.shape[0]
    degree = adjacency.sum(dim=1)
    lap = normalized_laplacian(adjacency)
    v1 = degree.clamp_min(0).sqrt()
    norm = torch.linalg.vector_norm(v1)
    if float(norm) < 1e-15:
        return {
            "lambda2": 0.0,
            "fiedler": [0.0] * n,
            "aei": [0.0] * n,
            "degenerate": True,
            "orthogonality": 0.0,
        }
    v1 = v1 / norm

    def orthogonalize(vector):
        return vector - torch.dot(vector, v1) * v1

    spectral = torch.linalg.matrix_norm(lap, ord=2)
    shift = float(spectral) + 1e-6
    vector = orthogonalize(torch.tensor([float((i * 17 % 10) - 4) for i in range(n)], dtype=torch.float64))
    norm = torch.linalg.vector_norm(vector)
    if float(norm) < 1e-15:
        vector = orthogonalize(torch.tensor([1.0 if i == 0 else -1.0 / (n - 1) for i in range(n)], dtype=torch.float64))
        norm = torch.linalg.vector_norm(vector)
    vector = vector / norm
    eye_shift = shift
    for _step in range(steps):
        image = lap @ vector
        shifted = orthogonalize(eye_shift * vector - image)
        norm = torch.linalg.vector_norm(shifted)
        if float(norm) < 1e-15:
            break
        vector = shifted / norm
    image = lap @ vector
    rayleigh = float(torch.dot(vector, image).item())
    diff = (vector[:, None] - vector[None, :]).abs()
    aei = (adjacency * diff).sum(dim=1)
    return {
        "lambda2": rayleigh,
        "fiedler": vector.tolist(),
        "aei": aei.tolist(),
        "degenerate": rayleigh < 1e-8,
        "orthogonality": abs(float(torch.dot(vector, v1).item())),
    }


def _reference_solve_laplacian(lap, rhs):
    """List-valued reduced solve used so leverage sampling follows the reference path."""
    n = len(lap)
    mean = sum(rhs) / float(n)
    shifted = [float(value) - mean for value in rhs]
    reduced = []
    for i in range(n - 1):
        last = float(lap[i][n - 1])
        reduced.append([float(lap[i][j]) - last for j in range(n - 1)])
    head = _ge_solve(reduced, shifted[:-1])
    return head + [-sum(head)]


def _ge_solve(matrix, rhs):
    n = len(matrix)
    aug = [list(map(float, matrix[i])) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = col
        best = abs(aug[col][col])
        for row in range(col + 1, n):
            magnitude = abs(aug[row][col])
            if magnitude > best:
                best = magnitude
                pivot = row
        if best < 1e-14:
            raise ValueError("singular matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def spectral_sparsify(edges, n, probes, rounds, seed):
    """Leverage-score sampling. Sum of w R on a connected graph is n - 1."""
    weighted = [[int(edge[0]), int(edge[1]), float(edge[2])] for edge in edges]
    lap = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, j, weight in weighted:
        lap[i][i] += weight
        lap[j][j] += weight
        lap[i][j] -= weight
        lap[j][i] -= weight
    leverage = []
    for u, v, weight in weighted:
        rhs = [0.0] * n
        rhs[u] = 1.0
        rhs[v] = -1.0
        potential = _reference_solve_laplacian(lap, rhs)
        leverage.append(weight * (potential[u] - potential[v]))
    total = sum(leverage)
    probabilities = [value / total for value in leverage]
    state = seed % 2147483647
    counts = [0 for _ in weighted]
    for _draw in range(rounds):
        state = (1103515245 * state + 12345) % 2147483647
        sample = state / 2147483647
        cursor = 0.0
        chosen = len(weighted) - 1
        for index, probability in enumerate(probabilities):
            cursor += probability
            if sample <= cursor:
                chosen = index
                break
        counts[chosen] += 1
    sparse_edges = []
    for index, (u, v, weight) in enumerate(weighted):
        if counts[index] == 0:
            continue
        sparse_edges.append([u, v, counts[index] * weight / (rounds * probabilities[index])])
    sparse = combinatorial_laplacian(n, sparse_edges)

    def energy(matrix, probe):
        vec = torch.tensor(probe, dtype=torch.float64)
        mixed = matrix @ vec
        return float(torch.dot(vec, mixed).item())

    dense = combinatorial_laplacian(n, weighted)
    return {
        "sum_leverage": total,
        "edges": sparse_edges,
        "original_energy": [energy(dense, probe) for probe in probes],
        "sparse_energy": [energy(sparse, probe) for probe in probes],
    }


def residual_covariance(sigma, keep, tau):
    sigma = sigma if torch.is_tensor(sigma) else _tensor(sigma)
    kept = list(keep)
    block = sigma[kept][:, kept].clone()
    block = block + float(tau) * torch.eye(len(kept), dtype=sigma.dtype)
    cross = sigma[kept, :]
    solved = torch.linalg.solve(block, cross)
    tall = sigma[:, kept]
    correction = tall @ solved
    return sigma - correction


def info_loss(residual, alpha, w_out):
    residual = residual if torch.is_tensor(residual) else _tensor(residual)
    weight = w_out if torch.is_tensor(w_out) else _tensor(w_out)
    spread = weight @ residual @ weight.transpose(0, 1)
    return float(((1.0 - alpha) * torch.trace(residual) + alpha * torch.trace(spread)).item())


def duplicate_gram():
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return rows.transpose(0, 1) @ rows


def del_residual(q_prev, q_curr, q_next, h):
    """Free-particle discrete Euler-Lagrange residual, L1."""
    prev = torch.tensor(q_prev, dtype=torch.float64)
    curr = torch.tensor(q_curr, dtype=torch.float64)
    nxt = torch.tensor(q_next, dtype=torch.float64)
    left = (curr - prev) / h
    right = (curr - nxt) / h
    return float((left + right).abs().sum().item())


def jl_dimension(n, eps):
    """Achlioptas sketch width k = ceil(24 ln n / eps^2)."""
    return max(1, math.ceil(24.0 * math.log(max(n, 2)) / (eps * eps)))
