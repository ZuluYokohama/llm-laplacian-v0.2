"""Host check: grid matvec, Schur matvec, and a 5-by-5 resistance identity."""

import torch

from laplacian.grid import (
    _dense_matrix,
    boundary_nodes,
    graph_apply,
    iterative_resistances,
    schur_matvec,
    boundary_mask,
)
from laplacian.operators_torch import boundary_schur, effective_resistance


def test_graph_apply_matches_dense_and_kills_constants():
    field = torch.randn(6, 6, dtype=torch.float64)
    dense = _dense_matrix("neumann", 6, torch.float64, field.device)
    applied = graph_apply(field).reshape(-1)
    reference = dense @ field.reshape(-1)
    assert torch.max(torch.abs(applied - reference)).item() < 1e-12
    assert torch.max(torch.abs(graph_apply(torch.ones(6, 6, dtype=torch.float64)))).item() < 1e-12


def test_schur_matvec_matches_dense_on_side_5():
    side = 5
    lap = _dense_matrix("neumann", side, torch.float64, "cpu")
    nodes = boundary_nodes(side)
    schur = boundary_schur(lap, nodes)
    mask = boundary_mask(side, "cpu", torch.float64)
    values = torch.randn(len(nodes), dtype=torch.float64)
    produced = schur_matvec(values, mask, tol=1e-10, maxiter=40, log_prefix=None)
    reference = schur @ values
    assert torch.max(torch.abs(produced - reference)).item() < 1e-6


def test_side5_resistance_identity():
    side = 5
    pair = ((0, 0), (0, side - 1))
    measured = iterative_resistances(side, pair, torch.float64, "cpu", 1e-8, 40, "side5")
    lap = _dense_matrix("neumann", side, torch.float64, "cpu")
    dense = effective_resistance(lap, 0, side - 1)
    assert abs(measured["full"] - dense) < 1e-4
    assert abs(measured["schur"] - dense) < 1e-4
    assert measured["gap"] < 1e-4
