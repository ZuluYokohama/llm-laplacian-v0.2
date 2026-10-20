import torch

from laplacian.cubical import boundary_matrices, nilpotency_residual


def test_one_cube_is_nilpotent_in_integers():
    matrices, counts = boundary_matrices(1)
    b1, b2, b3 = matrices
    assert counts["n3"] == 1
    assert int(torch.matmul(b1, b2).abs().sum().item()) == 0
    assert int(torch.matmul(b2, b3).abs().sum().item()) == 0


def test_fp64_residual_on_a_two_cell_grid():
    report = nilpotency_residual(2, device="cpu", dtype=torch.float64)
    assert report["residual_l1"] == 0.0
