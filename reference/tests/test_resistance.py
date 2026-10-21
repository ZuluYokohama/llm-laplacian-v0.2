def _path(n):
    return combinatorial_laplacian(n, [[i, i + 1, 1.0] for i in range(n - 1)])


def test_endpoint_resistance_matches_length():
    resistance = effective_resistance(_path(8), 0, 7)
    assert abs(resistance - 7.0) < 1e-8


def test_schur_preserves_endpoint_resistance():
    lap = _path(8)
    full = effective_resistance(lap, 0, 7)
    reduced = effective_resistance(boundary_schur(lap, [0, 7]), 0, 1)
    assert abs(full - reduced) < 1e-8


def test_diagonal_resistance_is_zero():
    assert effective_resistance(_path(4), 2, 2) == 0.0


def test_off_diagonal_resistance_is_positive():
    assert effective_resistance(_path(4), 0, 3) > 0.0
