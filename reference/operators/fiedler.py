"""Fiedler vector by projected power iteration, and the Adjacency Edge Index.

The Laplacian is the symmetric normalized one. v1 = D^{1/2} 1.
R_i = sum_j A_ij |v2_i - v2_j|. A Rayleigh quotient under 1e-8 is reported as degenerate.
"""


def path_adjacency(n):
    adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n - 1):
        adjacency[i][i + 1] = 1.0
        adjacency[i + 1][i] = 1.0
    return adjacency


def threshold_adjacency(adjacency, tau):
    out = []
    for i, row in enumerate(adjacency):
        kept = []
        for j, weight in enumerate(row):
            kept.append(weight if i != j and weight >= tau else 0.0)
        out.append(kept)
    return out


def _matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _normalized(adjacency):
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
    return lap, degree


def _spectral_norm(matrix):
    n = len(matrix)
    vector = [float((i * 3 + 1) % 5) - 2.0 for i in range(n)]
    norm = sum(value * value for value in vector) ** 0.5
    vector = [value / norm for value in vector]
    rayleigh = 0.0
    for _step in range(24):
        image = _matvec(matrix, vector)
        norm = sum(value * value for value in image) ** 0.5
        if norm < 1e-15:
            return 0.0
        vector = [value / norm for value in image]
        rayleigh = _dot(vector, _matvec(matrix, vector))
    return abs(rayleigh)


def fiedler_aei(adjacency, steps=80):
    n = len(adjacency)
    lap, degree = _normalized(adjacency)
    v1 = [value ** 0.5 for value in degree]
    norm = sum(value * value for value in v1) ** 0.5
    if norm < 1e-15:
        return {
            "lambda2": 0.0,
            "fiedler": [0.0 for _ in range(n)],
            "aei": [0.0 for _ in range(n)],
            "degenerate": True,
            "orthogonality": 0.0,
        }
    v1 = [value / norm for value in v1]

    def orthogonalize(vector):
        projection = _dot(vector, v1)
        return [vector[i] - projection * v1[i] for i in range(n)]

    shift = _spectral_norm(lap) + 1e-6
    vector = orthogonalize([float((i * 17 % 10) - 4) for i in range(n)])
    norm = sum(value * value for value in vector) ** 0.5
    if norm < 1e-15:
        vector = orthogonalize([1.0 if i == 0 else -1.0 / (n - 1) for i in range(n)])
        norm = sum(value * value for value in vector) ** 0.5
    vector = [value / norm for value in vector]
    for _step in range(steps):
        image = _matvec(lap, vector)
        shifted = orthogonalize([shift * vector[i] - image[i] for i in range(n)])
        norm = sum(value * value for value in shifted) ** 0.5
        if norm < 1e-15:
            break
        vector = [value / norm for value in shifted]
    image = _matvec(lap, vector)
    rayleigh = _dot(vector, image)
    aei = []
    for i in range(n):
        score = 0.0
        for j in range(n):
            weight = adjacency[i][j]
            if weight:
                score += weight * abs(vector[i] - vector[j])
        aei.append(score)
    return {
        "lambda2": rayleigh,
        "fiedler": vector,
        "aei": aei,
        "degenerate": rayleigh < 1e-8,
        "orthogonality": abs(_dot(vector, v1)),
    }
