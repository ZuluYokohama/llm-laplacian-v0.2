"""Spectral objective with a supplied cross-entropy term.

L_total = L_CE + g_man L_man + g_frac R_frac + g_hyp R_hyp + g_dir L_dir.
Coefficients follow the same phase table as schedule.py. Dirichlet energy is
Tr(H^T L H) / m on a combinatorial Laplacian built from an adjacency matrix.
"""


def _gammas(step):
    step = int(step)
    if step < 30000:
        return {"manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 1e-4}
    if step < 240000:
        return {"manifold": 0.05, "fractional": 0.02, "hyper": 0.01, "dirichlet": 1e-3}
    return {"manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 1e-4}


def total_loss(step, cross_entropy, manifold, fractional, hyper, dirichlet):
    weights = _gammas(step)
    return (
        cross_entropy
        + weights["manifold"] * manifold
        + weights["fractional"] * fractional
        + weights["hyper"] * hyper
        + weights["dirichlet"] * dirichlet
    )


def dirichlet_term(adjacency, features):
    count = len(adjacency)
    lap = [[0.0 for _ in range(count)] for _ in range(count)]
    edges = 0
    for i in range(count):
        for j in range(i + 1, count):
            weight = adjacency[i][j]
            if weight == 0.0:
                continue
            edges += 1
            lap[i][i] += weight
            lap[j][j] += weight
            lap[i][j] -= weight
            lap[j][i] -= weight
    edges = max(edges, 1)
    total = 0.0
    for col in range(len(features[0])):
        signal = [row[col] for row in features]
        mixed = [sum(lap[i][j] * signal[j] for j in range(count)) for i in range(count)]
        total += sum(signal[i] * mixed[i] for i in range(count))
    return total / edges
