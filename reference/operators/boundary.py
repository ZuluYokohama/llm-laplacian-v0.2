"""Signed boundary nilpotency.

Guidance axiom 1: the L1 norm of B_k @ B_{k+1} is 0.
B_k has shape n_{k-1} by n_k. An empty next matrix (no faces) is vacuously nilpotent.
"""


def _product(left, right):
    rows = len(left)
    inner = len(right)
    cols = len(right[0]) if inner and right[0] else 0
    if cols == 0 or rows == 0:
        return []
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for t in range(inner):
            factor = left[i][t]
            if factor == 0:
                continue
            for j in range(cols):
                out[i][j] += factor * right[t][j]
    return out


def nilpotency_l1(b_k, b_next):
    """L1 norm of B_k @ B_{k+1}. Exact integer arithmetic."""
    product = _product(b_k, b_next)
    return sum(abs(entry) for row in product for entry in row)


def path_boundaries():
    """Vertices 0-1-2 and edges (0,1), (1,2). No 2-simplices."""
    b1 = [
        [-1, 0],
        [1, -1],
        [0, 1],
    ]
    b2 = [[], []]
    return {"b1": b1, "b2": b2}


def triangle_boundaries():
    """One oriented triangle. Columns of B1 are edges (0,1), (1,2), (2,0)."""
    b1 = [
        [-1, 0, 1],
        [1, -1, 0],
        [0, 1, -1],
    ]
    b2 = [[1], [1], [1]]
    return {"b1": b1, "b2": b2}
