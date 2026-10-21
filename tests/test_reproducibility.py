"""Seed, state-domain, and artifact safety checks, with no CUDA execution."""

import random

import numpy as np
import pytest
import torch

from laplacian.model import SpectralTransformer
from laplacian.reproducibility import config_hash, restore_rng_state, rng_state, seed_everything
from laplacian.train import checked_optimizer_step, finite_loss_values, save_weights


def draws():
    return random.random(), np.random.random(4), torch.rand(4)


def assert_draws_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def test_seed_reproduces_initialization_and_global_streams():
    def initialize():
        model = SpectralTransformer(vocab=13, m=1, d_model=16, n_layers=1, context=4)
        return {name: tensor.clone() for name, tensor in model.state_dict().items()}, draws()

    seed_everything(1729)
    first, random_first = initialize()
    seed_everything(1729)
    second, random_second = initialize()
    for name in first:
        assert torch.equal(first[name], second[name]), name
    assert_draws_equal(random_first, random_second)
    seed_everything(1730)
    third, _ = initialize()
    assert not torch.equal(first["tok.weight"], third["tok.weight"])


def test_global_rng_roundtrip_restores_all_cpu_domains():
    seed_everything(13)
    snapshot = rng_state("cpu")
    expected = draws()
    seed_everything(999)
    assert restore_rng_state(snapshot, "cpu")
    assert_draws_equal(expected, draws())


def test_cuda_rng_domain_is_explicit_without_initializing_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda index: torch.tensor([index], dtype=torch.uint8))
    restored = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: restored.append((state, str(device))))
    saved = rng_state("cuda:2")
    assert saved["torch_cuda"]["device_index"] == 2
    assert restore_rng_state(saved, "cuda:2")
    assert restored[0][1] == "cuda:2"
    assert torch.equal(restored[0][0], torch.tensor([2], dtype=torch.uint8))
    assert not restore_rng_state(saved, "cpu")
    assert not restore_rng_state(rng_state("cpu"), "cuda:2")


def test_configuration_identity_allows_relocation_but_not_different_updates():
    assert config_hash({"run_id": "a", "seed": 1, "steps": 8}) == config_hash({"run_id": "b", "steps": 8, "seed": 1})
    assert config_hash({"seed": 1, "steps": 8}) != config_hash({"seed": 1, "steps": 9})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_rejected_before_backward(bad, monkeypatch):
    monkeypatch.setattr("laplacian.train.log_line", lambda line: None)
    loss = torch.tensor(bad, requires_grad=True)
    with pytest.raises(FloatingPointError, match="before optimizer update"):
        finite_loss_values(loss, torch.tensor(1.0), 8)
    assert loss.grad is None


def test_nonfinite_gradient_does_not_mutate_parameters_or_optimizer(monkeypatch):
    monkeypatch.setattr("laplacian.train.log_line", lambda line: None)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(FloatingPointError, match="before optimizer update"):
        checked_optimizer_step(model, optimizer, 1.0, 8)
    assert optimizer.state == {}
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_weight_export_rejects_nonfinite_state_and_preserves_last_valid_file(tmp_path):
    model = torch.nn.Linear(2, 1)
    target = tmp_path / "weights_final.pt"
    save_weights(target, model, 2)
    original = target.read_bytes()
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="checkpoint was not written"):
        save_weights(target, model, 3)
    assert target.read_bytes() == original
    assert not target.with_suffix(".pt.tmp").exists()


def test_pulse_only_does_not_publish_success_when_weight_validation_fails(tmp_path, monkeypatch):
    import laplacian.train as training

    monkeypatch.setattr(training, "RUNS", tmp_path)
    monkeypatch.setattr(training, "memory_mib", lambda: 0.0)
    monkeypatch.setattr(training, "run_pulse", lambda *args: ([], 0.0))
    messages = []
    monkeypatch.setattr(training, "log_line", messages.append)
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    config = {"run_id": "invalid-pulse", "m": 1, "k": 1, "calibration_sequences": 1}
    with pytest.raises(FloatingPointError, match="checkpoint was not written"):
        training._pulse_only(config, model, None, None, None, "cpu", "CPU test", {"vocab": 13})
    directory = tmp_path / config["run_id"]
    assert not (directory / "weights_final.pt").exists()
    assert not (directory / "metrics.json").exists()
    assert not (directory / "metrics.json.sha256").exists()
    assert not any(message.startswith("DONE") for message in messages)
