"""Leverage-score sampling on one small connected graph.

p_e is proportional to w_e R_eff(e). Each draw adds w_e / (rounds * p_e).
Sum of w R_eff on a connected graph equals n - 1.
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


def _solve_laplacian(lap, rhs):
    n = len(lap)
    mean = sum(rhs) / float(n)
    shifted = [float(value) - mean for value in rhs]
    reduced = []
    for i in range(n - 1):
        last = float(lap[i][n - 1])
        reduced.append([float(lap[i][j]) - last for j in range(n - 1)])
    head = _solve(reduced, shifted[:-1])
    return head + [-sum(head)]


def _laplacian(n, edges):
    lap = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, j, weight in edges:
        lap[i][i] += weight
        lap[j][j] += weight
        lap[i][j] -= weight
        lap[j][i] -= weight
    return lap


def _resistance(lap, i, j):
    rhs = [0.0 for _ in range(len(lap))]
    rhs[i] = 1.0
    rhs[j] = -1.0
    potential = _solve_laplacian(lap, rhs)
    return potential[i] - potential[j]


def _energy(lap, probe):
    mixed = [sum(lap[i][j] * probe[j] for j in range(len(probe))) for i in range(len(probe))]
    return sum(probe[i] * mixed[i] for i in range(len(probe)))


def spectral_sparsify(edges, n, probes, rounds, seed):
    """Sample `rounds` edges with replacement. `edges` are [u, v, weight]."""
    weighted = [[int(edge[0]), int(edge[1]), float(edge[2])] for edge in edges]
    lap = _laplacian(n, weighted)
    leverage = []
    for u, v, weight in weighted:
        leverage.append(weight * _resistance(lap, u, v))
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
    sparse = _laplacian(n, sparse_edges)
    return {
        "sum_leverage": total,
        "edges": sparse_edges,
        "original_energy": [_energy(lap, probe) for probe in probes],
        "sparse_energy": [_energy(sparse, probe) for probe in probes],
    }
