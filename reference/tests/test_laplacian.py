def _path_edges():
    return [[0, 1, 1.0], [1, 2, 1.0]]


def test_constant_feature_has_zero_energy():
    lap = combinatorial_laplacian(3, _path_edges())
    energy = dirichlet_energy(lap, [[1.0], [1.0], [1.0]])
    assert abs(energy) < 1e-12


def test_path_bump_has_energy_one():
    lap = combinatorial_laplacian(3, _path_edges())
    energy = dirichlet_energy(lap, [[0.0], [1.0], [0.0]])
    assert abs(energy - 1.0) < 1e-12


def test_combinatorial_kernel_is_exact():
    lap = combinatorial_laplacian(3, _path_edges())
    assert kernel_residual(lap) == 0.0


def test_normalized_kernel_is_degree_weighted():
    adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    lap = normalized_laplacian(adjacency)
    weighted = [[1.0], [2.0 ** 0.5], [1.0]]
    assert abs(quadratic_form(lap, weighted)) < 1e-8


def test_nonconstant_energy_is_positive():
    lap = combinatorial_laplacian(3, _path_edges())
    assert dirichlet_energy(lap, [[0.0], [1.0], [0.0]]) > 0.0


def test_band_penalty_is_zero_inside_and_positive_outside():
    assert band_penalty([0.5], 0.2, 0.8) == 0.0
    assert band_penalty([0.0], 0.2, 0.8) > 0.0
    assert band_penalty([1.0], 0.2, 0.8) > 0.0
