def test_uniform_step_is_symplectic():
    residual = del_residual([0.0, 0.0], [1.0, -2.0], [2.0, -4.0], 0.5)
    assert residual < 1e-12


def test_broken_step_is_not_symplectic():
    residual = del_residual([0.0], [1.0], [3.0], 1.0)
    assert residual > 0.5
