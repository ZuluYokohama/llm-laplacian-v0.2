"""Independent finite representation controls. Standard library only.

No model, file, network, Torch, process or GPU operations are used.
"""
import math


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in transpose(b)] for row in a]


def gram(a):
    return multiply(a, transpose(a))


def affinity(points, tau=1.):
    return [[math.exp(-sum((x - y) ** 2 for x, y in zip(a, b)) / (2 * tau * tau)) for b in points] for a in points]


def laplacian(adjacency):
    degree = [sum(row) for row in adjacency]
    return [[(float(i == j) - adjacency[i][j] / math.sqrt(degree[i] * degree[j])) if degree[i] * degree[j] > 0 else 0.
             for j in range(len(degree))] for i in range(len(degree))]


def quadratic(lap, features):
    product = multiply(lap, features)
    return sum(x * y for row, result in zip(features, product) for x, y in zip(row, result))


def coordinate_ablation(h, w):
    return [sum(row[i] ** 2 for row in h) * sum(x ** 2 for x in w[i]) for i in range(len(w))]


def rotation_case():
    h = [[1., 0.], [0., 1.], [1., 1.], [2., -1.]]
    w = [[1.], [2.]]
    q = [[.6, -.8], [.8, .6]]
    return h, w, multiply(h, q), multiply(transpose(q), w)


def positive_density(features):
    return sum(x > 0 for row in features for x in row) / sum(len(row) for row in features)


def centered_variance(points):
    means = [sum(col) / len(points) for col in transpose(points)]
    return sum((x - y) ** 2 for row in points for x, y in zip(row, means)) / sum(len(row) for row in points)
