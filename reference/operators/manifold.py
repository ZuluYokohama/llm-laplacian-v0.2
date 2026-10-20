"""Cross-layer manifold penalty on a batch of points.

W_ij = exp(-||z_i - z_j||^2 / (2 tau^2)), then the symmetric normalized Laplacian.
The penalty is Tr(Z_l^T L_k Z_l) / B^2.
"""

import math


def affinity_laplacian(points, tau):
    count = len(points)
    width = len(points[0])
    affinity = [[0.0 for _ in range(count)] for _ in range(count)]
    scale = 2.0 * tau * tau
    for i in range(count):
        for j in range(count):
            dist = 0.0
            for k in range(width):
                delta = points[i][k] - points[j][k]
                dist += delta * delta
            affinity[i][j] = math.exp(-dist / scale)
    degree = [sum(row) for row in affinity]
    lap = [[0.0 for _ in range(count)] for _ in range(count)]
    for i in range(count):
        lap[i][i] = 1.0
        scale_i = degree[i] ** 0.5
        for j in range(count):
            if i == j:
                continue
            lap[i][j] = -affinity[i][j] / (scale_i * (degree[j] ** 0.5))
    return lap


def manifold_penalty(features, lap_k):
    count = len(features)
    width = len(features[0])
    total = 0.0
    for col in range(width):
        signal = [row[col] for row in features]
        mixed = [sum(lap_k[i][j] * signal[j] for j in range(count)) for i in range(count)]
        total += sum(signal[i] * mixed[i] for i in range(count))
    return total / (count * count)
