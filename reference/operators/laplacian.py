"""Combinatorial and normalized Laplacians, Dirichlet energy, and the band-pass hinge.

Energy uses E = Tr(H^T L H) / m with m the number of undirected edges.
The hinge is max(0, E_min - E)^2 + max(0, E - E_max)^2, summed over layers.
"""


def combinatorial_laplacian(n, edges):
    """Undirected edges as [i, j, weight]. Returns the combinatorial Laplacian."""
    lap = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, j, weight in edges:
        weight = float(weight)
        lap[i][i] += weight
        lap[j][j] += weight
        lap[i][j] -= weight
        lap[j][i] -= weight
    return lap


def normalized_laplacian(adjacency):
    """Symmetric normalized Laplacian I - D^{-1/2} A D^{-1/2}."""
    n = len(adjacency)
    degree = [sum(row) for row in adjacency]
    lap = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        if degree[i] == 0.0:
            continue
        lap[i][i] = 1.0
        scale_i = degree[i] ** 0.5
        for j in range(n):
            if i == j or adjacency[i][j] == 0.0 or degree[j] == 0.0:
                continue
            lap[i][j] = -adjacency[i][j] / (scale_i * (degree[j] ** 0.5))
    return lap


def _edge_count(lap):
    count = 0
    for i in range(len(lap)):
        for j in range(i + 1, len(lap)):
            if lap[i][j] < 0.0:
                count += 1
    return max(count, 1)


def quadratic_form(lap, features):
    """Tr(H^T L H) for features stored as rows."""
    total = 0.0
    width = len(features[0])
    for col in range(width):
        signal = [row[col] for row in features]
        mixed = [sum(lap[i][j] * signal[j] for j in range(len(signal))) for i in range(len(signal))]
        total += sum(signal[i] * mixed[i] for i in range(len(signal)))
    return total


def dirichlet_energy(lap, features):
    return quadratic_form(lap, features) / _edge_count(lap)


def kernel_residual(lap):
    """Max absolute entry of L times the all-ones vector."""
    worst = 0.0
    for row in lap:
        total = abs(sum(row))
        if total > worst:
            worst = total
    return worst


def band_penalty(energies, e_min, e_max):
    total = 0.0
    for energy in energies:
        low = e_min - energy
        high = energy - e_max
        if low > 0.0:
            total += low * low
        if high > 0.0:
            total += high * high
    return total
