"""Fractional Laplacian penalty via a Jacobi eigensolve.

R = (lambda_frac / 2) Tr(Z^T L^s Z), with negative roundoff eigenvalues clamped to 0.
"""

import math


def _jacobi(matrix):
    """Eigenvalues and a column-eigenvector matrix for a symmetric matrix."""
    n = len(matrix)
    work = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _sweep in range(64):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += work[p][q] * work[p][q]
        if off < 1e-28:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = work[p][q]
                if abs(apq) < 1e-18:
                    continue
                app = work[p][p]
                aqq = work[q][q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    tangent = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    tangent = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine
                for i in range(n):
                    if i == p or i == q:
                        continue
                    aip = work[i][p]
                    aiq = work[i][q]
                    work[i][p] = work[p][i] = cosine * aip - sine * aiq
                    work[i][q] = work[q][i] = sine * aip + cosine * aiq
                work[p][p] = app - tangent * apq
                work[q][q] = aqq + tangent * apq
                work[p][q] = work[q][p] = 0.0
                for i in range(n):
                    vip = vectors[i][p]
                    viq = vectors[i][q]
                    vectors[i][p] = cosine * vip - sine * viq
                    vectors[i][q] = sine * vip + cosine * viq
    values = [work[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda index: values[index])
    values = [values[index] for index in order]
    vectors = [[vectors[row][index] for index in order] for row in range(n)]
    return values, vectors


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


def fractional_matrix(lap, exponent):
    values, vectors = _jacobi(lap)
    powered = [0.0 if value <= 0.0 else value ** exponent for value in values]
    scaled = [[vectors[row][col] * powered[col] for col in range(len(powered))] for row in range(len(vectors))]
    return _matmul(scaled, _transpose(vectors))


def fractional_penalty(lap, features, exponent, lam=1.0):
    powered = fractional_matrix(lap, exponent)
    total = 0.0
    width = len(features[0])
    count = len(features)
    for col in range(width):
        signal = [row[col] for row in features]
        mixed = [sum(powered[i][j] * signal[j] for j in range(count)) for i in range(count)]
        total += sum(signal[i] * mixed[i] for i in range(count))
    return (lam / 2.0) * total
