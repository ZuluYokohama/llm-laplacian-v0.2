"""Suzuki ridge recalibration of a covariance matrix.

A_J = Sigma_{*, J} (Sigma_{J, J} + tau I)^{-1}.
Sigma_rem = Sigma - Sigma_{*, J} (Sigma_{J, J} + tau I)^{-1} Sigma_{J, *}.
Info loss is (1 - alpha) Tr(Sigma_rem) + alpha Tr(W Sigma_rem W^T).
"""


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


def _trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def _matmul(left, right):
    rows = len(left)
    inner = len(right)
    cols = len(right[0])
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for t in range(inner):
            factor = left[i][t]
            if factor == 0.0:
                continue
            for j in range(cols):
                out[i][j] += factor * right[t][j]
    return out


def _transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def duplicate_gram():
    """6 features. Columns 3, 4, 5 copy columns 0, 1, 2. Rank 3."""
    rows = [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    ]
    size = len(rows[0])
    gram = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            gram[i][j] = sum(rows[t][i] * rows[t][j] for t in range(len(rows)))
    return gram


def residual_covariance(sigma, keep, tau):
    kept = list(keep)
    block = [[sigma[kept[a]][kept[b]] for b in range(len(kept))] for a in range(len(kept))]
    for index in range(len(kept)):
        block[index][index] += tau
    cross = [[sigma[kept[a]][j] for j in range(len(sigma))] for a in range(len(kept))]
    solved = _solve(block, cross)
    tall = [[sigma[i][kept[b]] for b in range(len(kept))] for i in range(len(sigma))]
    correction = _matmul(tall, solved)
    return [[sigma[i][j] - correction[i][j] for j in range(len(sigma))] for i in range(len(sigma))]


def info_loss(residual, alpha, w_out):
    spread = _matmul(w_out, _matmul(residual, _transpose(w_out)))
    return (1.0 - alpha) * _trace(residual) + alpha * _trace(spread)
