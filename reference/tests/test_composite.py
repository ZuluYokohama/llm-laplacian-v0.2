def test_dirichlet_term_matches_the_path_value():
    adjacency = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    energy = dirichlet_term(adjacency, [[0.0], [1.0], [0.0]])
    assert abs(energy - 1.0) < 1e-12


def test_step_zero_uses_only_dirichlet():
    total = total_loss(0, 1.0, 10.0, 10.0, 10.0, 2.0)
    assert abs(total - (1.0 + 1e-4 * 2.0)) < 1e-12


def test_phase_two_uses_every_spectral_term():
    total = total_loss(100000, 1.0, 2.0, 3.0, 4.0, 5.0)
    expected = 1.0 + 0.05 * 2.0 + 0.02 * 3.0 + 0.01 * 4.0 + 1e-3 * 5.0
    assert abs(total - expected) < 1e-12
