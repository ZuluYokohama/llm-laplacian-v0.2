"""Analytic CPU controls for descriptive model-output diagnostics."""
import json
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics

torch.set_num_threads(1)
TOL = 2e-6


def outputs(points):
    pooled = torch.tensor(points, dtype=torch.float32)
    b, d = pooled.shape
    stream = pooled[:, None, :].repeat(1, 3, 1)
    stream[:, 1] += .25
    stream[:, 2] -= .5
    adjacency = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    return {"embedded": torch.zeros((b, 3, d)), "pooled": [pooled], "stream": [stream],
            "hidden": [stream.clone()], "attn": [adjacency[None, None].repeat(b, 1, 1, 1)]}


def test_two_point_gaussian_spectrum_variance_rank_and_fractional_energy():
    value = metrics.analyze(outputs([[-1., 0.], [1., 0.]]), selected_layers=(0,))["layers"][0]
    a = math.exp(-2)
    eigenvalue = 2 * a / (1 + a)
    assert value["selected"] is True
    assert abs(value["pooled_variance"] - .5) < 1e-14
    assert abs(value["effective_rank"] - 1) < 1e-12
    assert abs(value["affinity"]["mean"] - a) < TOL
    assert abs(value["normalized_spectrum"]["second_eigenvalue"] - eigenvalue) < TOL
    assert abs(value["fractional_energy"] - math.sqrt(eigenvalue)) < TOL


def test_identical_points_have_zero_centered_rank_and_near_one_affinity():
    value = metrics.analyze(outputs([[2., -1.]] * 3))["layers"][0]
    assert value["pooled_variance"] == value["effective_rank"] == 0
    assert value["affinity"]["near_one_fraction"] == 1
    assert value["affinity"]["near_zero_fraction"] == 0
    # FP32 normalization leaves a small positive constant-mode eigenvalue.
    # The C1 slope near zero amplifies it; compare an analytic scalar oracle
    # instead of falsely asserting an exact zero or relaxing invariance tolerance.
    eigenvalue = 2 ** -24
    scalar = 150 * eigenvalue - 500000 * eigenvalue ** 2
    assert value["fractional_energy"] == pytest.approx(.5 * 15 * scalar, abs=TOL)
    assert value["fractional_energy"] > TOL


def test_disconnected_underflowed_graph_counts_nulls_in_second_eigenvalue():
    value = metrics.analyze(outputs([[0., 0.], [100., 0.], [-100., 0.]]))["layers"][0]
    assert value["affinity"]["near_zero_fraction"] == 1
    assert value["normalized_spectrum"]["nullity_at_1e_6"] == 3
    assert value["normalized_spectrum"]["second_eigenvalue"] == 0
    assert value["fractional_energy"] == 0


def test_repeated_nonzero_eigenvalues_are_finite_and_not_a_failure():
    # Equilateral triangle: two equal positive normalized eigenvalues.
    value = metrics.analyze(outputs([[0., 0.], [1., 0.], [.5, math.sqrt(3) / 2]]))["layers"][0]
    eigenvalues = value["normalized_spectrum"]["eigenvalues"]
    assert abs(eigenvalues[1] - eigenvalues[2]) < TOL
    assert value["effective_rank"] == pytest.approx(2, abs=TOL)
    assert math.isfinite(value["fractional_energy"])


def test_example_permutation_preserves_pooled_diagnostics():
    one = outputs([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
    two = {"embedded": one["embedded"][[2, 0, 3, 1]], **{key: [one[key][0][[2, 0, 3, 1]]] for key in ["pooled", "stream", "hidden", "attn"]}}
    a, b = [metrics.analyze(item)["layers"][0] for item in [one, two]]
    for key in ["pooled_variance", "effective_rank", "incidence_density"]:
        assert abs(a[key] - b[key]) < TOL
    assert abs(a["token_band"]["energy"] - b["token_band"]["energy"]) < TOL


def test_orthogonal_geometry_fixed_graph_energy_and_coordinate_dependence():
    one = outputs([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
    q = torch.tensor([[.6, -.8], [.8, .6]])
    two = {"embedded": one["embedded"] @ q, "attn": one["attn"],
           **{key: [one[key][0] @ q] for key in ["pooled", "stream", "hidden"]}}
    a, b = [metrics.analyze(item)["layers"][0] for item in [one, two]]
    for key in ["pooled_variance", "effective_rank"]:
        assert abs(a[key] - b[key]) < TOL
    assert abs(a["token_band"]["energy"] - b["token_band"]["energy"]) < TOL
    assert a["incidence_density"] != b["incidence_density"]
    # Rotation is a control at a linear feature/readout boundary only.
    w = torch.tensor([[1.], [2.]])
    assert torch.allclose(one["pooled"][0] @ w, two["pooled"][0] @ (q.T @ w), atol=TOL, rtol=0)


def test_scaling_is_not_affinity_invariant_at_fixed_bandwidth():
    one = outputs([[-1., 0.], [1., 0.]])
    two = {"embedded": one["embedded"] * 2, "attn": one["attn"],
           **{key: [one[key][0] * 2] for key in ["pooled", "stream", "hidden"]}}
    a, b = [metrics.analyze(item)["layers"][0] for item in [one, two]]
    assert b["pooled_variance"] == 4 * a["pooled_variance"]
    assert b["affinity"]["mean"] < a["affinity"]["mean"]
    assert b["token_band"]["energy"] == pytest.approx(4 * a["token_band"]["energy"], abs=TOL)


def test_token_band_has_analytic_path_energy_and_matches_frozen_squared_hinges():
    case = outputs([[0.]])
    case["stream"][0] = torch.tensor([[[0.], [1.], [2.]]])
    result = metrics.analyze(case)["layers"][0]["token_band"]
    assert result["energy"] == 1  # Two path edges each have unit squared difference.
    assert result["base_energy"] == 0
    assert result["upper_active"] is True and result["lower_active"] is False
    assert result["penalty"] == pytest.approx((result["energy"] - result["upper"]) ** 2, abs=TOL)
    frozen = metrics.losses.spectral_penalties(case, [0], .5, weights={"manifold": 0, "fractional": 0, "hyper": 0, "dirichlet": 1})
    assert result["penalty"] == float(frozen["dirichlet"])


def test_singleton_empty_off_diagonal_and_read_only_finite_json():
    case = outputs([[3., 1.]])
    before = [t.clone() for t in [case["embedded"], *(case[key][0] for key in ["pooled", "stream", "hidden", "attn"])]]
    report = metrics.analyze(case)
    value = report["layers"][0]
    assert value["affinity"]["count"] == 0 and value["affinity"]["mean"] is None
    assert value["normalized_spectrum"]["second_eigenvalue"] is None
    assert value["fractional_energy"] == 0
    json.dumps(report, allow_nan=False)
    assert all(torch.equal(a, b) for a, b in zip(before, [case["embedded"], *(case[key][0] for key in ["pooled", "stream", "hidden", "attn"])]))


@pytest.mark.parametrize("kind", ["nonfinite", "device", "shape"])
def test_invalid_input_is_rejected_without_gpu_allocation(kind):
    case = outputs([[0., 1.], [1., 0.]])
    if kind == "nonfinite": case["pooled"][0][0, 0] = float("nan")
    if kind == "device": case["pooled"][0] = torch.empty((2, 2), device="meta")
    if kind == "shape": case["hidden"][0] = torch.ones((1, 3, 2))
    with pytest.raises(ValueError): metrics.analyze(case)
