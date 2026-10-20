"""Effective resistance and the boundary Schur complement.

R_ij(L) = (e_i - e_j)^T L^+ (e_i - e_j).
Lambda = L_BB - L_BI L_II^{-1} L_IB on a chosen boundary index set.
"""


def combinatorial_laplacian(n, edges):
    lap = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, j, weight in edges:
        weight = float(weight)
        lap[i][i] += weight
        lap[j][j] += weight
        lap[i][j] -= weight
        lap[j][i] -= weight
    return lap


def _solve(matrix, rhs):
    n = len(matrix)
    vector = isinstance(rhs[0], (int, float))
    width = 1 if vector else len(rhs[0])
    aug = []
    for i in range(n):
        row = [float(value) for value in matrix[i]]
        if vector:
            row.append(float(rhs[i]))
        else:
            row.extend(float(value) for value in rhs[i])
        aug.append(row)
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
        for j in range(col, n + width):
            aug[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + width):
                aug[row][j] -= factor * aug[col][j]
    if vector:
        return [aug[i][n] for i in range(n)]
    return [[aug[i][n + j] for j in range(width)] for i in range(n)]


def solve_laplacian(lap, rhs):
    """Solve L x = rhs on the subspace orthogonal to the constants."""
    n = len(lap)
    mean = sum(rhs) / float(n)
    shifted = [float(value) - mean for value in rhs]
    reduced = []
    for i in range(n - 1):
        last = float(lap[i][n - 1])
        reduced.append([float(lap[i][j]) - last for j in range(n - 1)])
    head = _solve(reduced, shifted[:-1])
    return head + [-sum(head)]


def effective_resistance(lap, i, j):
    if i == j:
        return 0.0
    rhs = [0.0 for _ in range(len(lap))]
    rhs[i] = 1.0
    rhs[j] = -1.0
    potential = solve_laplacian(lap, rhs)
    return potential[i] - potential[j]


def _take(matrix, rows, cols):
    return [[matrix[i][j] for j in cols] for i in rows]


def boundary_schur(lap, boundary):
    """Schur complement of `lap` onto the boundary index list."""
    chosen = set(boundary)
    interior = [index for index in range(len(lap)) if index not in chosen]
    block_bb = _take(lap, boundary, boundary)
    if not interior:
        return block_bb
    block_ii = _take(lap, interior, interior)
    block_ib = _take(lap, interior, boundary)
    block_bi = _take(lap, boundary, interior)
    solved = _solve(block_ii, block_ib)
    correction = [[0.0 for _ in boundary] for _ in boundary]
    for i in range(len(boundary)):
        for j in range(len(boundary)):
            correction[i][j] = sum(block_bi[i][t] * solved[t][j] for t in range(len(interior)))
    return [[block_bb[i][j] - correction[i][j] for j in range(len(boundary))] for i in range(len(boundary))]
