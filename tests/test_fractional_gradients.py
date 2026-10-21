"""Analytic and independent numerical oracles for the versioned matrix power."""

import math

import numpy as np
import pytest
import torch

from laplacian.losses import _affinity, spectral_penalties
from laplacian.operators_torch import (
    FRACTIONAL_CUTOFF,
    FRACTIONAL_POLICY_VERSION,
    fractional_matrix,
    fractional_policy,
)


def _scalar(value, exponent, cutoff=FRACTIONAL_CUTOFF):
    if value >= cutoff:
        return value ** exponent
    slope = (2 - exponent) * cutoff ** (exponent - 1)
    if value < 0:
        return slope * value
    return cutoff ** exponent * ((2 - exponent) * (value / cutoff) + (exponent - 1) * (value / cutoff) ** 2)


def _derivative(value, exponent, cutoff=FRACTIONAL_CUTOFF):
    if value >= cutoff:
        return exponent * value ** (exponent - 1)
    slope = (2 - exponent) * cutoff ** (exponent - 1)
    return slope if value < 0 else slope + 2 * (exponent - 1) * cutoff ** (exponent - 2) * value


def _numpy_power(matrix, exponent):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.array([_scalar(float(value), exponent) for value in values])) @ vectors.T


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75, 1.0])
def test_known_eigenbasis_forward_and_frechet_gradient(exponent):
    values = np.array([0, 0, FRACTIONAL_CUTOFF / 2, FRACTIONAL_CUTOFF, 0.7, 0.7])
    rng = np.random.default_rng(15)
    basis, _ = np.linalg.qr(rng.normal(size=(len(values), len(values))))
    matrix = torch.tensor((basis * values) @ basis.T, requires_grad=True)
    weight = rng.normal(size=matrix.shape)
    weight = (weight + weight.T) / 2
    result = fractional_matrix(matrix, exponent)
    gradient = torch.autograd.grad((result * torch.tensor(weight)).sum(), matrix)[0]
    expected = (basis * np.array([_scalar(value, exponent) for value in values])) @ basis.T
    divided = np.array([
        [(_scalar(a, exponent) - _scalar(b, exponent)) / (a - b) if a != b else _derivative(a, exponent) for b in values]
        for a in values
    ])
    expected_gradient = basis @ (divided * (basis.T @ weight @ basis)) @ basis.T
    np.testing.assert_allclose(result.detach().numpy(), expected, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(gradient.numpy(), expected_gradient, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75])
def test_zero_matrix_has_analytic_nonzero_derivative(exponent):
    matrix = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 16
    result = fractional_matrix(matrix, exponent)
    gradient = torch.autograd.grad((result * weight).sum(), matrix)[0]
    assert torch.count_nonzero(result) == 0
    expected = _derivative(0, exponent) * (weight + weight.T) / 2
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("spectrum", ["zero", "repeated", "joins", "near_repeated"])
def test_first_derivative_passes_independent_finite_differences(exponent, spectrum):
    values = {
        "zero": [0, 0, 0],
        "repeated": [0.3, 0.3, 0.9],
        "joins": [0, FRACTIONAL_CUTOFF, 0.8],
        "near_repeated": [0.4, 0.4 + 1e-12, 0.4 + 2e-12],
    }[spectrum]
    matrix = torch.diag(torch.tensor(values, dtype=torch.float64)).requires_grad_()
    # At the C1 (not C2) joins, central differences have O(h) error.
    assert torch.autograd.gradcheck(
        lambda item: fractional_matrix(item, exponent), (matrix,),
        eps=1e-8, atol=1e-4, rtol=2e-4,
    )


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75])
def test_dense_numpy_oracle_and_directional_derivative(exponent):
    rng = np.random.default_rng(33)
    raw = rng.normal(size=(5, 5))
    lap = raw @ raw.T / 10 + np.eye(5) * 0.01
    direction = rng.normal(size=(5, 5))
    direction = (direction + direction.T) / 2
    weight = rng.normal(size=(5, 5))
    matrix = torch.tensor(lap, requires_grad=True)
    result = fractional_matrix(matrix, exponent)
    gradient = torch.autograd.grad((result * torch.tensor(weight)).sum(), matrix)[0]
    h = 1e-6
    expected_direction = ((_numpy_power(lap + h * direction, exponent) - _numpy_power(lap - h * direction, exponent)) * weight).sum() / (2 * h)
    np.testing.assert_allclose(result.detach().numpy(), _numpy_power(lap, exponent), rtol=2e-12, atol=2e-12)
    assert float((gradient * torch.tensor(direction)).sum()) == pytest.approx(expected_direction, rel=2e-7, abs=2e-8)


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75])
def test_disconnected_graph_and_permutation_equivariance(exponent):
    block = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    lap = torch.block_diag(block, block, torch.zeros(1, 1, dtype=torch.float64)).requires_grad_()
    weight = torch.arange(25, dtype=torch.float64).reshape(5, 5) / 25
    result = fractional_matrix(lap, exponent)
    gradient = torch.autograd.grad((result * weight).sum(), lap)[0]
    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted = lap.detach()[permutation][:, permutation].requires_grad_()
    permuted_result = fractional_matrix(permuted, exponent)
    permuted_gradient = torch.autograd.grad((permuted_result * weight[permutation][:, permutation]).sum(), permuted)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(result, torch.tensor(_numpy_power(lap.detach().numpy(), exponent)))
    torch.testing.assert_close(permuted_result, result[permutation][:, permutation], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(permuted_gradient, gradient[permutation][:, permutation], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("fixture", ["separated", "zero", "constant", "duplicates"])
@pytest.mark.parametrize("autocast", [False, True])
def test_actual_fractional_loss_degenerate_backward(exponent, fixture, autocast):
    inputs = {
        "separated": torch.eye(4) * 100,
        "zero": torch.zeros(4, 4),
        "constant": torch.ones(4, 4),
        "duplicates": torch.tensor([[0., 0.], [0., 0.], [1., 1.], [1., 1.]]),
    }[fixture].requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        result = spectral_penalties(
            {"pooled": [inputs]}, [0], exponent,
            weights={"fractional": 1., "manifold": 0., "hyper": 0., "dirichlet": 0.},
        )["fractional"]
    gradient = torch.autograd.grad(result, inputs)[0]
    assert torch.isfinite(result)
    assert torch.isfinite(gradient).all()


def test_graph_gradient_is_not_silently_detached():
    points = torch.tensor([[0.1, 0.2], [0.4, -0.2], [-0.3, 0.1], [0.3, 0.4]], requires_grad=True)
    lap = _affinity(points, 1.0).double()
    power = fractional_matrix(lap, 0.5).float()
    loss = 0.5 * (points * (power @ points)).sum()
    gradient = torch.autograd.grad(loss, points)[0]
    detached_gradient = power.detach() @ points.detach()
    assert (gradient - detached_gradient).norm() > 1e-3

    def independent_objective(raw):
        affinity = np.exp(-((raw[:, None] - raw[None, :]) ** 2).sum(axis=2) / 2)
        inv = 1 / np.sqrt(affinity.sum(axis=1))
        lap = np.eye(len(raw)) - affinity * inv[:, None] * inv[None, :]
        return 0.5 * np.trace(raw.T @ _numpy_power(lap, 0.5) @ raw)

    raw = points.detach().double().numpy()
    finite_difference = np.empty_like(raw)
    for index in np.ndindex(raw.shape):
        plus, minus = raw.copy(), raw.copy()
        plus[index] += 1e-4
        minus[index] -= 1e-4
        finite_difference[index] = (independent_objective(plus) - independent_objective(minus)) / 2e-4
    np.testing.assert_allclose(gradient.numpy(), finite_difference, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_autocast_does_not_reduce_spectral_operator_precision(dtype):
    matrix = torch.diag(torch.tensor([0, 0.25, 0.25, 1], dtype=dtype)).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fractional_matrix(matrix, 0.5)
        actual_gradient = torch.autograd.grad(actual.sum(), matrix)[0]
    reference_input = matrix.detach().float().requires_grad_()
    reference = fractional_matrix(reference_input, 0.5)
    reference_gradient = torch.autograd.grad(reference.sum(), reference_input)[0]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, reference_gradient.to(dtype), rtol=0, atol=0)


def test_forward_preserves_unsmoothed_power_above_cutoff_and_exponent_one():
    values = torch.tensor([FRACTIONAL_CUTOFF, 0.2, 0.2, 1.5], dtype=torch.float64)
    for exponent in (0.25, 0.5, 0.75):
        torch.testing.assert_close(fractional_matrix(torch.diag(values), exponent), torch.diag(values.pow(exponent)), rtol=0, atol=0)
    matrix = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64, requires_grad=True)
    result = fractional_matrix(matrix, 1)
    torch.testing.assert_close(result, matrix, rtol=0, atol=0)
    assert fractional_policy()["version"] == FRACTIONAL_POLICY_VERSION


def test_materially_indefinite_input_rejected_but_roundoff_extension_is_defined():
    with pytest.raises(ValueError, match="positive semidefinite"):
        fractional_matrix(torch.diag(torch.tensor([-1e-3, 1.], dtype=torch.float64)), 0.5)
    matrix = torch.diag(torch.tensor([-1e-8, 1.], dtype=torch.float64)).requires_grad_()
    result = fractional_matrix(matrix, 0.5)
    gradient = torch.autograd.grad(result.sum(), matrix)[0]
    assert result[0, 0].item() == pytest.approx(_scalar(-1e-8, 0.5))
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("exponent", [0, -0.1, 1.1, math.inf, math.nan])
def test_unsupported_exponents_rejected(exponent):
    with pytest.raises(ValueError, match="exponent"):
        fractional_matrix(torch.eye(2), exponent)


@pytest.mark.parametrize("cutoff", [0, -1, math.inf, math.nan])
def test_invalid_cutoffs_rejected(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        fractional_matrix(torch.eye(2), 0.5, cutoff=cutoff)


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_nonfinite_matrix_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        fractional_matrix(torch.tensor([[value, 0], [0, 1.]]), 0.5)


def test_double_backward_explicitly_unsupported():
    matrix = (torch.eye(2, dtype=torch.float64) * 0.5).requires_grad_()
    result = fractional_matrix(matrix, 0.5)
    first = torch.autograd.grad(result.square().sum(), matrix, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(first.sum(), matrix)
