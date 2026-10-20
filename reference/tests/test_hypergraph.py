def _triple():
    return hypergraph_laplacian([[1.0], [1.0], [1.0]], [1.0])


def test_constant_activation_has_zero_form():
    form = quadratic_form(_triple(), [[1.0], [1.0], [1.0]])
    assert abs(form) < 1e-8


def test_contrast_activation_is_positive():
    form = quadratic_form(_triple(), [[1.0], [-1.0], [0.0]])
    assert form > 0.1


def test_quadratic_form_is_nonnegative():
    lap = _triple()
    samples = ([[1.0], [0.0], [0.0]], [[0.2], [-0.4], [0.7]], [[-1.0], [-1.0], [2.0]])
    for features in samples:
        assert quadratic_form(lap, features) >= -1e-8
