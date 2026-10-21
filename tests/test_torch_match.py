"""Host check: torch twins agree with the receipted reference operators."""

import importlib.util
from pathlib import Path

import torch

from laplacian import operators_torch as twin
from laplacian.schedule import gammas

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "operators"


def load(name):
    spec = importlib.util.spec_from_file_location(f"ref_{name}", REF / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nilpotency_is_exact():
    boundary = load("boundary")
    pair = boundary.triangle_boundaries()
    assert twin.nilpotency_l1(pair["b1"], pair["b2"]) == 0
    assert twin.nilpotency_l1([[1, 0], [0, 1]], [[1, 0], [0, 1]]) == 2


def test_path_energy_matches():
    lap = twin.combinatorial_laplacian(3, [[0, 1, 1.0], [1, 2, 1.0]])
    energy = twin.dirichlet_energy(lap, [[0.0], [1.0], [0.0]])
    assert abs(energy - 1.0) < 1e-8
    assert twin.kernel_residual(lap) == 0.0


def test_resistance_matches_length_and_schur():
    edges = [[i, i + 1, 1.0] for i in range(7)]
    lap = twin.combinatorial_laplacian(8, edges)
    full = twin.effective_resistance(lap, 0, 7)
    reduced = twin.effective_resistance(twin.boundary_schur(lap, [0, 7]), 0, 1)
    assert abs(full - 7.0) < 1e-8
    assert abs(full - reduced) < 1e-8


def test_fiedler_endpoints_and_sparsify_identity():
    report = twin.fiedler_aei(twin.path_adjacency(6), steps=80)
    assert report["degenerate"] is False
    order = sorted(range(6), key=lambda index: report["aei"][index])
    assert set(order[:2]) == {0, 5}
    edges = [[0, 2, 1.0], [0, 3, 1.0], [1, 2, 1.0], [1, 3, 1.0]]
    probes = [[1.0, -1.0, 0.5, -0.5], [1.0, 1.0, -1.0, -1.0]]
    sparse = twin.spectral_sparsify(edges, 4, probes, 400, 7)
    assert abs(sparse["sum_leverage"] - 3.0) < 1e-6


def test_suzuki_fractional_and_symplectic():
    sigma = twin.duplicate_gram()
    raw = twin.info_loss(sigma, 0.5, torch.eye(6, dtype=torch.float64))
    residual = twin.residual_covariance(sigma, [0, 1, 2], 1e-8)
    assert twin.info_loss(residual, 0.5, torch.eye(6, dtype=torch.float64)) < min(raw, 1e-4)
    lap = [
        [1.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, -1.0, 1.0],
    ]
    features = [[1.0, 0.0], [0.5, -1.0], [0.0, 0.25], [-0.5, 0.5]]
    got = twin.fractional_penalty(lap, features, 1.0, lam=2.0)
    assert abs(got - twin.quadratic_form(torch.tensor(lap, dtype=torch.float64), features)) < 1e-6
    assert twin.del_residual([0.0, 0.0], [1.0, -2.0], [2.0, -4.0], 0.5) < 1e-12


def test_ramp_disagrees_with_the_piecewise_table_on_purpose():
    assert gammas(0)["manifold"] == 0.0
    assert gammas(30000)["phase"] == 2
    assert gammas(30000)["manifold"] == 0.0
    mid = gammas(135000)
    assert abs(mid["manifold"] - 0.025) < 1e-12
    assert abs(mid["dirichlet"] - 5.5e-4) < 1e-12
    assert abs(gammas(240000)["manifold"] - 0.05) < 1e-12
    assert gammas(300000)["manifold"] == 0.0


def test_reference_hypergraph_sign():
    form = twin.quadratic_form(
        twin.hypergraph_laplacian([[1.0], [1.0], [1.0]], [1.0]),
        [[1.0], [-1.0], [0.0]],
    )
    assert form > 0.1
