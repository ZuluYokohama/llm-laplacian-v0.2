"""Loss reassociation must preserve the objective and its activation gradients."""

import pytest
import torch

from laplacian import losses
from laplacian.operators_torch import hypergraph_laplacian


def _legacy_hypergraph(activations):
    """Frozen pre-optimization expression, including its vertex feature scaling."""
    flat = activations.float().reshape(-1, activations.shape[-1]).T
    incidence = (flat > 0).float()
    edge_degree = incidence.sum(dim=0).clamp_min(1)
    degree = incidence.sum(dim=1)
    inv = torch.zeros_like(degree)
    positive = degree > 0
    inv[positive] = degree[positive].rsqrt()
    scaled = incidence * inv[:, None]
    features = flat * inv[:, None]
    middle = (scaled.T @ features) / edge_degree[:, None]
    back = scaled @ middle
    return (features.square().sum() - (features * back).sum()).clamp_min(0)


def _dense_hypergraph(activations):
    """Independent normalized-Laplacian oracle applied to degree-scaled features."""
    flat = activations.float().reshape(-1, activations.shape[-1]).T
    incidence = (flat > 0).float()
    degree = incidence.sum(dim=1)
    inv = torch.where(degree > 0, degree.clamp_min(1).rsqrt(), 0)
    features = flat * inv[:, None]
    lap = hypergraph_laplacian(incidence, torch.ones(flat.shape[1]))
    return torch.trace(features.T @ lap @ features).clamp_min(0)


@pytest.mark.parametrize("shape", [(24, 7), (7, 24), (12, 12), (2, 5, 8), (0, 5)])
def test_hypergraph_values_and_gradients_match_legacy_and_dense_laplacian(shape):
    generator = torch.Generator().manual_seed(17)
    activations = torch.randn(shape, generator=generator).requires_grad_()
    result = losses.hypergraph_energy(activations)
    gradient = torch.autograd.grad(result, activations)[0]
    for reference in (_legacy_hypergraph, _dense_hypergraph):
        expected = reference(activations)
        expected_gradient = torch.autograd.grad(expected, activations)[0]
        torch.testing.assert_close(result, expected, rtol=3e-6, atol=2e-6)
        torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(24, 7), (7, 24)])
@pytest.mark.parametrize("kind", ["zero", "negative", "constant", "isolated"])
def test_hypergraph_degenerate_inputs(shape, kind):
    activations = torch.zeros(shape)
    if kind == "negative":
        activations.fill_(-1)
    elif kind == "constant":
        activations.fill_(1)
    elif kind == "isolated":
        activations[0, 0] = 2
        activations[1, 0] = 3
        activations[2, 1] = 1
    activations.requires_grad_()
    result = losses.hypergraph_energy(activations)
    expected = _legacy_hypergraph(activations)
    gradient = torch.autograd.grad(result, activations)[0]
    expected_gradient = torch.autograd.grad(expected, activations)[0]
    assert torch.isfinite(result)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(24, 7), (7, 24)])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_hypergraph_cpu_bfloat16_autocast(shape, input_dtype):
    activations = torch.randn(shape, generator=torch.Generator().manual_seed(31)).to(input_dtype)
    activations.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = losses.hypergraph_energy(activations)
        expected = _legacy_hypergraph(activations)
    gradient = torch.autograd.grad(result, activations)[0]
    expected_gradient = torch.autograd.grad(expected, activations)[0]
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.005)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("autocast", [False, True])
def test_quadratic_trace_reassociation_values_and_both_gradients(autocast):
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(7, 19, generator=generator).requires_grad_()
    lap = torch.randn(7, 7, generator=generator).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        result = losses._quadratic(lap, features)
        expected = torch.trace(features.T @ lap @ features)
    gradients = torch.autograd.grad(result, (features, lap))
    expected_gradients = torch.autograd.grad(expected, (features, lap))
    rtol, atol = (0.02, 0.08) if autocast else (1e-5, 1e-5)
    torch.testing.assert_close(result, expected.float(), rtol=rtol, atol=atol)
    for gradient, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("autocast", [False, True])
def test_token_energy_values_and_both_gradients(autocast):
    generator = torch.Generator().manual_seed(22)
    states = torch.randn(8, 19, generator=generator).requires_grad_()
    affinity = torch.rand(8, 8, generator=generator).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        result, adjacency = losses._token_energy(states, affinity)
        degree = adjacency.sum(dim=1)
        edges = (torch.triu(adjacency, diagonal=1) > 0).sum().clamp_min(1)
        expected = ((states.square() * degree[:, None]).sum() - torch.trace(states.T @ adjacency @ states)) / edges
    gradients = torch.autograd.grad(result, (states, affinity), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (states, affinity))
    rtol, atol = (0.02, 0.01) if autocast else (1e-5, 1e-5)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)
    for gradient, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, reference, rtol=rtol, atol=atol)


def _output():
    generator = torch.Generator().manual_seed(4)

    def rand(*shape):
        return torch.randn(shape, generator=generator)

    return {
        "pooled": [rand(4, 6), rand(4, 6)],
        "hidden": [rand(2, 3, 7), rand(2, 3, 7)],
        "embedded": rand(2, 3, 6),
        "stream": [rand(2, 3, 6), rand(2, 3, 6)],
        "attn": [rand(2, 2, 3, 3).softmax(-1), rand(2, 2, 3, 3).softmax(-1)],
    }


@pytest.mark.parametrize("enabled", ["manifold", "fractional", "hyper", "dirichlet", None])
def test_disabled_penalties_do_not_access_their_inputs(enabled):
    output = _output()
    expected = losses.spectral_penalties(output, [0, 1], 1.0)
    weights = {name: float(name == enabled) for name in ("manifold", "fractional", "hyper", "dirichlet")}
    # Each subset lacks the input tensors used only by the other terms.
    keys = ["pooled"]
    if enabled == "hyper":
        keys += ["hidden"]
    if enabled == "dirichlet":
        keys += ["embedded", "stream", "attn"]
    partial = {key: output[key] for key in keys}
    result = losses.spectral_penalties(partial, [0, 1], 1.0, weights=weights)
    for name in weights:
        if name == enabled:
            torch.testing.assert_close(result[name], expected[name])
        else:
            assert result[name].shape == ()
            assert result[name].item() == 0
    assert result["token_energies"] == (expected["token_energies"] if enabled == "dirichlet" else [])


def test_manifold_only_does_not_run_fractional_eigendecomposition(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("An inactive penalty was evaluated")

    monkeypatch.setattr(losses, "fractional_matrix", forbidden)
    monkeypatch.setattr(losses, "hypergraph_energy", forbidden)
    monkeypatch.setattr(losses, "_lambda2", forbidden)
    result = losses.spectral_penalties(
        _output(), [0, 1], 0.5,
        weights={"manifold": 0.05, "fractional": 0, "hyper": 0, "dirichlet": 0},
    )
    assert result["manifold"] > 0


def test_lambda2_does_not_build_a_discarded_autograd_graph(monkeypatch):
    def estimate(adjacency, steps):
        assert not torch.is_grad_enabled()
        return torch.ones(3), 0.25, False

    monkeypatch.setattr(losses, "fiedler_from_adjacency", estimate)
    result = losses._lambda2(torch.ones(3, 3, requires_grad=True))
    assert result.item() == 0.25
    assert not result.requires_grad


def test_phase_one_forward_loss_skips_unscheduled_terms(monkeypatch):
    from laplacian.arms import get_config
    from laplacian.model import SpectralTransformer
    from laplacian.train import forward_loss

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Phase I evaluated an inactive penalty")

    monkeypatch.setattr(losses, "fractional_matrix", forbidden)
    monkeypatch.setattr(losses, "hypergraph_energy", forbidden)
    monkeypatch.setattr(losses, "_affinity", forbidden)
    config = get_config("arm2")
    config["layers_1index"] = [1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        model = SpectralTransformer(vocab=32, m=2, d_model=16, n_layers=1, context=4)
        tokens = torch.randint(0, 32, (2, 4))
        loss, ce, _output, penalties = forward_loss(model, tokens, tokens, config, step=0)
        loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(ce)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    for name in ("manifold", "fractional", "hyper"):
        assert penalties[name].item() == 0
