"""CPU-only checks of hardware-gate comparison and failure receipts; no CUDA calls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpu_resume_gate_test", ROOT / "scripts/check_gpu_resume.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_snapshot_is_owned_and_exact_comparison_catches_optimizer_buffer_rng_changes():
    state = {"model_buffer": torch.tensor([1., 2.]),
             "optimizer": {0: {"exp_avg": torch.tensor([.5])}},
             "numpy_rng": np.array([13, 14], dtype=np.uint32),
             "torch_rng": torch.tensor([1, 2], dtype=torch.uint8),
             "flags": (True, "training", 4)}
    saved = gate.cpu_snapshot(state)
    assert gate.compare_state(saved, state)["exact"]
    state["model_buffer"][0] += .25
    state["optimizer"][0]["exp_avg"][0] += .125
    state["numpy_rng"][0] += 1
    state["torch_rng"][0] += 1
    comparison = gate.compare_state(saved, state)
    assert not comparison["exact"]
    assert comparison["mismatch_count"] == 4
    assert comparison["max_abs_difference"] == 1.0
    assert saved["model_buffer"][0] == 1


@pytest.mark.parametrize("other", [torch.tensor([1.], dtype=torch.float64),
                                   torch.tensor([[1.]]), torch.tensor([float("nan")]),
                                   torch.tensor([float("inf")])])
def test_exact_comparison_rejects_dtype_shape_and_nonfinite(other):
    result = gate.compare_state(torch.tensor([1.]), other)
    assert result["exact"] is False
    json.dumps(result, allow_nan=False)


def test_exact_comparison_does_not_allow_small_floating_difference():
    before = torch.tensor([1.], dtype=torch.float64)
    result = gate.compare_state(before, before+1e-12)
    assert not result["exact"]
    assert result["max_abs_difference"] > 0
    assert not gate.compare_state({"a": 1}, {"b": 1})["exact"]
    assert not gate.compare_state((1,), [1])["exact"]


def test_cpu_synthetic_batch_and_fixed_configuration_are_reproducible():
    config = gate.gate_config(7)
    left = gate.synthetic_batch(config, torch.Generator().manual_seed(7), "cpu")
    right = gate.synthetic_batch(config, torch.Generator().manual_seed(7), "cpu")
    assert gate.compare_state(left, right)["exact"]
    assert left[0].shape == left[1].shape == (8, 16)
    assert torch.equal(left[0][:, 1:], left[1][:, :-1])
    assert config["gate_fixed_forward_step"] == 135000
    assert config["gate_forced_resample_after"] == 3
    assert all(value > 0 for value in gate.training.penalty_weights(config, 135000).values())


def test_hardware_rejection_persists_failure_without_overwriting_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "require_gpu", lambda: (_ for _ in ()).throw(RuntimeError("CPU test rejection")))
    output = tmp_path / "attempt"
    with pytest.raises(RuntimeError, match="CPU test rejection"):
        gate.run_gate(output)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert report["host_gpu"] is False and report["sources_unchanged"] is True
    assert "CPU test rejection" in report["error"]
    assert (output / "metrics.json.sha256").is_file()
    original = (output / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        gate.run_gate(output)
    assert (output / "metrics.json").read_bytes() == original


def test_source_mutation_is_recorded_even_on_hardware_failure(tmp_path, monkeypatch):
    counter = 0
    def hashes():
        nonlocal counter
        counter += 1
        return gate.IMPORTED_SOURCE_HASHES if counter == 1 else {"changed.py": "different"}
    monkeypatch.setattr(gate, "source_hashes", hashes)
    monkeypatch.setattr(gate, "require_gpu", lambda: (_ for _ in ()).throw(RuntimeError("CPU fixture")))
    with pytest.raises(RuntimeError, match="CPU fixture"):
        gate.run_gate(tmp_path / "mutation")
    report = json.loads((tmp_path / "mutation/metrics.json").read_text())
    assert report["complete"] is False
    assert report["sources_unchanged"] is False
    assert "Source changed" in report["error"]


@pytest.mark.parametrize("seconds", [0, -1, 61, float("nan"), float("inf")])
def test_invalid_duration_rejected_before_creating_attempt(tmp_path, seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        gate.run_gate(tmp_path / "invalid", max_seconds=seconds)
    assert not (tmp_path / "invalid").exists()
