"""Product cubical complex whose boundary obeys the graded Leibniz rule.

Nilpotency is a consequence of d^2 = 0 on elementary intervals, not of a fitted model.
"""

from __future__ import annotations

import torch


def _enumerate(cells):
    """Return cubes keyed by dimension. A cube is a tuple of (start, end) pairs."""
    groups = {0: [], 1: [], 2: [], 3: []}
    for x in range(cells + 1):
        for y in range(cells + 1):
            for z in range(cells + 1):
                x_choices = [(x, x)] if x == cells else [(x, x), (x, x + 1)]
                y_choices = [(y, y)] if y == cells else [(y, y), (y, y + 1)]
                z_choices = [(z, z)] if z == cells else [(z, z), (z, z + 1)]
                for xv in x_choices:
                    for yv in y_choices:
                        for zv in z_choices:
                            cube = (xv, yv, zv)
                            dim = (xv[0] != xv[1]) + (yv[0] != yv[1]) + (zv[0] != zv[1])
                            groups[dim].append(cube)
    return groups


def _boundary_terms(cube):
    """Graded Leibniz boundary. Returns (sign, face) pairs."""
    terms = []
    degree = 0
    for axis in range(3):
        start, end = cube[axis]
        if start == end:
            continue
        left = list(cube)
        right = list(cube)
        left[axis] = (start, start)
        right[axis] = (end, end)
        sign = 1 if degree % 2 == 0 else -1
        terms.append((-sign, tuple(left)))
        terms.append((sign, tuple(right)))
        degree += 1
    return terms


def boundary_matrices(cells, dtype=torch.int64, device="cpu"):
    """Dense integer boundaries B1, B2, B3 with shapes n_{k-1} by n_k."""
    groups = _enumerate(cells)
    index = [{cube: i for i, cube in enumerate(groups[dim])} for dim in range(4)]
    matrices = []
    for dim in range(1, 4):
        rows = len(groups[dim - 1])
        cols = len(groups[dim])
        matrix = torch.zeros((rows, cols), dtype=dtype, device=device)
        for column, cube in enumerate(groups[dim]):
            for sign, face in _boundary_terms(cube):
                matrix[index[dim - 1][face], column] = sign
        matrices.append(matrix)
    counts = {f"n{dim}": len(groups[dim]) for dim in range(4)}
    return matrices, counts


def nilpotency_residual(cells, device, dtype=torch.float64):
    """CUDA/CPU residual ||B_k B_{k+1}||_1 for k = 1, 2."""
    matrices, counts = boundary_matrices(cells, dtype=dtype, device=device)
    b1, b2, b3 = matrices
    first = torch.matmul(b1, b2).abs().sum()
    second = torch.matmul(b2, b3).abs().sum()
    return {
        "cells": cells,
        "counts": counts,
        "residual_b1_b2": float(first.item()),
        "residual_b2_b3": float(second.item()),
        "residual_l1": float((first + second).item()),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
    }
