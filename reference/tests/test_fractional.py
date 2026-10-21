def _path_laplacian():
    return [
        [1.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, -1.0, 1.0],
    ]


def _features():
    return [[1.0, 0.0], [0.5, -1.0], [0.0, 0.25], [-0.5, 0.5]]


def _quadratic(lap, features):
    total = 0.0
    count = len(features)
    for col in range(len(features[0])):
        signal = [row[col] for row in features]
        mixed = [sum(lap[i][j] * signal[j] for j in range(count)) for i in range(count)]
        total += sum(signal[i] * mixed[i] for i in range(count))
    return total


def test_exponent_one_matches_the_quadratic_form():
    lap = _path_laplacian()
    features = _features()
    # lam = 2 makes (lam / 2) * Tr(Z^T L Z) equal the quadratic form.
    got = fractional_penalty(lap, features, 1.0, lam=2.0)
    assert abs(got - _quadratic(lap, features)) < 1e-8


def test_half_exponent_is_nonnegative():
    penalty = fractional_penalty(_path_laplacian(), _features(), 0.5, lam=1.0)
    assert penalty >= -1e-10
