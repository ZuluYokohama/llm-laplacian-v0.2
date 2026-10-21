"""Grid-graph Laplacian solves without a dense n-by-n matrix.

The fine matvec is the combinatorial grid Laplacian. A non-recursive geometric
V-cycle (Jacobi, bilinear transfer, rediscretized coarse grids) preconditions
PCG. The Schur matvec eliminates the interior with a Dirichlet solve, so the
boundary effective resistance is not the same code path as the full solve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from laplacian.operators_torch import _reduced_solve


def touch_heartbeat(line):
    print(line, flush=True)
    try:
        from laplacian.runtime import RUNS

        path = RUNS / "heartbeat.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(line + "\n", encoding="utf-8")
    except OSError:
        return


def level_sizes(side):
    sizes = [int(side)]
    while sizes[-1] > 4:
        nxt = (sizes[-1] + 1) // 2
        if nxt >= sizes[-1] or nxt < 2:
            break
        sizes.append(int(nxt))
    return sizes


def graph_apply(field):
    """Combinatorial Laplacian of the grid graph with unit edge weights."""
    up = torch.zeros_like(field)
    down = torch.zeros_like(field)
    left = torch.zeros_like(field)
    right = torch.zeros_like(field)
    up[1:] = field[:-1]
    down[:-1] = field[1:]
    left[:, 1:] = field[:, :-1]
    right[:, :-1] = field[:, 1:]
    degree = torch.zeros_like(field)
    degree[1:] += 1
    degree[:-1] += 1
    degree[:, 1:] += 1
    degree[:, :-1] += 1
    return degree * field - (up + down + left + right)


def dirichlet_apply(field):
    """Five-point Dirichlet Laplacian. Missing neighbors are the zero boundary."""
    neighbors = torch.zeros_like(field)
    neighbors[1:] += field[:-1]
    neighbors[:-1] += field[1:]
    neighbors[:, 1:] += field[:, :-1]
    neighbors[:, :-1] += field[:, 1:]
    return 4.0 * field - neighbors


def _resample(field, shape):
    image = field[None, None]
    resized = F.interpolate(image, size=shape, mode="bilinear", align_corners=True)
    return resized[0, 0]


_DENSE = {}


def _dense_matrix(kind, side, dtype, device):
    key = (kind, int(side), str(dtype), str(device))
    cached = _DENSE.get(key)
    if cached is not None:
        return cached
    matrix = torch.zeros((side * side, side * side), dtype=dtype, device=device)
    for row in range(side):
        for col in range(side):
            index = row * side + col
            if kind == "dirichlet":
                matrix[index, index] = 4.0
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row + d_row
                next_col = col + d_col
                inside = 0 <= next_row < side and 0 <= next_col < side
                if kind == "neumann" and inside:
                    matrix[index, index] += 1.0
                    matrix[index, next_row * side + next_col] -= 1.0
                if kind == "dirichlet" and inside:
                    matrix[index, next_row * side + next_col] = -1.0
    _DENSE[key] = matrix
    return matrix


def _direct(rhs, kind):
    side = rhs.shape[0]
    matrix = _dense_matrix(kind, side, rhs.dtype, rhs.device)
    flat = rhs.reshape(-1)
    if kind == "neumann":
        solved = _reduced_solve(matrix, flat - flat.mean())
    else:
        solved = torch.linalg.solve(matrix, flat)
    return solved.reshape(rhs.shape)


def _smooth(field, rhs, kind, sweeps):
    updated = field
    for _sweep in range(sweeps):
        if kind == "neumann":
            operator = graph_apply(updated)
            degree = torch.zeros_like(updated)
            degree[1:] += 1
            degree[:-1] += 1
            degree[:, 1:] += 1
            degree[:, :-1] += 1
            updated = updated - (2.0 / 3.0) * (operator - rhs) / degree.clamp_min(1.0)
            updated = updated - updated.mean()
        else:
            neighbors = 4.0 * updated - dirichlet_apply(updated)
            updated = (1.0 / 3.0) * updated + (2.0 / 3.0) * (neighbors + rhs) / 4.0
    return updated


def _apply_kind(field, kind):
    if kind == "neumann":
        return graph_apply(field)
    return dirichlet_apply(field)


def vcycle(rhs, kind, sweeps=2):
    """One V-cycle. The level walk is an explicit loop, not a recursive call."""
    sizes = level_sizes(rhs.shape[0])
    count = len(sizes)
    fields = [None] * count
    rights = [None] * count
    if kind == "neumann":
        rights[0] = rhs - rhs.mean()
    else:
        rights[0] = rhs
    fields[0] = torch.zeros_like(rights[0])
    for level in range(count - 1):
        fields[level] = _smooth(fields[level], rights[level], kind, sweeps)
        residual = rights[level] - _apply_kind(fields[level], kind)
        if kind == "neumann":
            residual = residual - residual.mean()
        coarse_side = sizes[level + 1]
        coarse = _resample(residual, (coarse_side, coarse_side))
        if kind == "neumann":
            coarse = coarse - coarse.mean()
        rights[level + 1] = coarse
        fields[level + 1] = torch.zeros_like(coarse)
    fields[-1] = _direct(rights[-1], kind)
    for level in range(count - 2, -1, -1):
        correction = _resample(fields[level + 1], fields[level].shape)
        fields[level] = fields[level] + correction
        if kind == "neumann":
            fields[level] = fields[level] - fields[level].mean()
        fields[level] = _smooth(fields[level], rights[level], kind, sweeps)
    return fields[0]


def pcg(apply, precondition, rhs, tol, maxiter, project, log_prefix):
    unknown = torch.zeros_like(rhs)
    target = project(rhs) if project is not None else rhs
    residual = target - apply(unknown)
    if project is not None:
        residual = project(residual)
    direction_z = precondition(residual)
    if project is not None:
        direction_z = project(direction_z)
    direction = direction_z
    rz = torch.sum(residual * direction_z)
    scale = torch.linalg.vector_norm(target).clamp_min(1e-30)
    last = float(torch.linalg.vector_norm(residual).item())
    for iteration in range(1, maxiter + 1):
        image = apply(direction)
        if project is not None:
            image = project(image)
        denom = torch.sum(direction * image)
        if float(torch.abs(denom)) < 1e-30:
            break
        alpha = rz / denom
        unknown = unknown + alpha * direction
        if project is not None:
            unknown = project(unknown)
        residual = residual - alpha * image
        if project is not None:
            residual = project(residual)
        last = float(torch.linalg.vector_norm(residual).item())
        if log_prefix and (iteration == 1 or iteration % 5 == 0):
            touch_heartbeat(f"PROGRESS {log_prefix} cg={iteration}/{maxiter} residual={last:.3e}")
        if last <= tol * float(scale):
            return unknown, iteration, last, True
        direction_z = precondition(residual)
        if project is not None:
            direction_z = project(direction_z)
        rz_next = torch.sum(residual * direction_z)
        beta = rz_next / rz.clamp_min(1e-30)
        direction = direction_z + beta * direction
        rz = rz_next
    return unknown, maxiter, last, False


def dirichlet_solve(rhs, tol, maxiter, log_prefix):
    def apply(field):
        return dirichlet_apply(field)

    def precondition(field):
        return vcycle(field, "dirichlet")

    return pcg(apply, precondition, rhs, tol, maxiter, None, log_prefix)


def boundary_mask(side, device, dtype):
    mask = torch.ones((side, side), dtype=torch.bool, device=device)
    if side >= 3:
        mask[1:-1, 1:-1] = False
    return mask


def boundary_nodes(side):
    nodes = []
    for row in range(side):
        for col in range(side):
            if row == 0 or col == 0 or row == side - 1 or col == side - 1 or side < 3:
                nodes.append(row * side + col)
    return nodes


def schur_matvec(values, mask, tol, maxiter, log_prefix):
    field = torch.zeros(mask.shape, dtype=values.dtype, device=values.device)
    field[mask] = values
    lifted = graph_apply(field)
    if mask.shape[0] < 3:
        return values * 0
    interior_rhs = lifted[1:-1, 1:-1].contiguous()
    solved, _iters, _residual, _ok = dirichlet_solve(interior_rhs, tol, maxiter, log_prefix)
    interior = torch.zeros_like(field)
    interior[1:-1, 1:-1] = solved
    return (lifted - graph_apply(interior))[mask]


def iterative_resistances(side, pair, dtype, device, tol, maxiter, log_prefix):
    """Return full-graph and Schur effective resistances for one boundary pair."""
    rhs = torch.zeros((side, side), dtype=dtype, device=device)
    start, end = pair
    rhs[start] = 1.0
    rhs[end] = -1.0

    def apply_full(field):
        return graph_apply(field)

    def precondition_full(field):
        return vcycle(field, "neumann")

    def project(field):
        return field - field.mean()

    potential, full_iters, full_residual, full_ok = pcg(
        apply_full, precondition_full, rhs, tol, maxiter, project, f"{log_prefix} full"
    )
    full = float((potential[start] - potential[end]).item())
    mask = boundary_mask(side, device, dtype)
    packed = rhs[mask]
    positions = torch.cumsum(mask.reshape(-1).to(torch.int64), dim=0) - 1
    local_start = int(positions[start[0] * side + start[1]].item())
    local_end = int(positions[end[0] * side + end[1]].item())

    def apply_schur(values):
        return schur_matvec(values, mask, tol, max(20, maxiter // 2), None)

    def project_boundary(values):
        return values - values.mean()

    reduced, schur_iters, schur_residual, schur_ok = pcg(
        apply_schur,
        lambda values: values,
        packed,
        tol,
        maxiter,
        project_boundary,
        f"{log_prefix} schur",
    )
    boundary = float((reduced[local_start] - reduced[local_end]).item())
    return {
        "full": full,
        "schur": boundary,
        "gap": abs(full - boundary),
        "full_iterations": full_iters,
        "schur_iterations": schur_iters,
        "full_residual": full_residual,
        "schur_residual": schur_residual,
        "full_converged": full_ok,
        "schur_converged": schur_ok,
    }
