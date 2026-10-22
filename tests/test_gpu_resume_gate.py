"""CPU-only checks of hardware-gate comparison and failure receipts; no CUDA calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    changed = False
    def hashes():
        return {"changed.py": "different"} if changed else gate.IMPORTED_SOURCE_HASHES
    def reject_hardware():
        nonlocal changed
        changed = True
        raise RuntimeError("CPU fixture")
    monkeypatch.setattr(gate, "source_hashes", hashes)
    monkeypatch.setattr(gate, "require_gpu", reject_hardware)
    with pytest.raises(RuntimeError, match="CPU fixture"):
        gate.run_gate(tmp_path / "mutation")
    report = json.loads((tmp_path / "mutation/metrics.json").read_text())
    assert report["complete"] is False
    assert report["sources_unchanged"] is False
    assert "Source changed" in report["error"]


def test_new_state_uses_production_model_builder_and_gate_vocabulary(monkeypatch):
    config = gate.gate_config(7)
    config["gate_vocab"] = 19
    seen = []
    monkeypatch.setattr(gate, "seed_everything", lambda seed: seen.append(("seed", seed)))
    model = torch.nn.Linear(2, 1)
    def build(actual_config, vocab, device):
        assert actual_config is config and vocab == 19 and device == "cpu"
        seen.append(("build", vocab))
        return model
    monkeypatch.setattr(gate.training, "build_model", build)
    actual, optimizer, generator = gate.new_training_state(config, "cpu")
    assert actual is model and list(optimizer.param_groups[0]["params"]) == list(model.parameters())
    assert generator.initial_seed() == 7
    assert seen == [("seed", 7), ("build", 19)]


@pytest.fixture
def cpu_replay(monkeypatch):
    """Drive orchestration with explicit CPU fakes; no model/CUDA replay claim."""
    controls = {"failure": None, "loaded": False, "clock": 0.0}
    monkeypatch.setattr(gate, "source_hashes", lambda: gate.IMPORTED_SOURCE_HASHES)
    monkeypatch.setattr(gate, "require_gpu", lambda: torch.device("cpu"))
    monkeypatch.setattr(gate, "environment", lambda device: {"fixture": "CPU fake"})
    monkeypatch.setattr(gate.time, "perf_counter", lambda: controls["clock"])
    monkeypatch.setattr(gate, "new_training_state", lambda config, device: (
        SimpleNamespace(blocks=[None, None], d_model=3, m=4, reconstructed=config["seed"] == 1730, step=0),
        object(), object()))
    def updates(model, optimizer, generator, config, device, start, end, checkpoint, traces, deadline, started, **kwargs):
        if controls["failure"] == "timeout":
            controls["clock"] = 61.0
        deadline()
        if start == 2 and controls["failure"] == "after_load":
            assert controls["loaded"]
            raise RuntimeError("CPU injected failure after restored RNG/state")
        model.step = end
        checkpoint.write_bytes(f"CPU fake checkpoint at {end}".encode())
        traces.extend({"update": step} for step in range(start + 1, end + 1))
        return None, 0
    monkeypatch.setattr(gate, "run_updates", updates)
    def snapshot(model, *args):
        mismatch = model.reconstructed and ((controls["failure"] == "roundtrip" and model.step == 2)
                                             or (controls["failure"] == "final" and model.step == 4))
        return {"weight": torch.tensor([1.25 if mismatch else 1.0])}
    monkeypatch.setattr(gate, "final_snapshot", snapshot)
    def load(path, model, optimizer, generator, device, *, config, resume_metadata):
        controls["loaded"] = True
        model.step = 2
        resume_metadata["resume_status"] = "state_complete"
        if controls["failure"] == "load":
            raise RuntimeError("CPU injected checkpoint load failure")
        return 2, [], [], None, 0
    monkeypatch.setattr(gate.training, "load_checkpoint", load)
    def probe(device):
        if controls["failure"] == "rng":
            raise RuntimeError("CPU injected RNG failure")
        return {}
    monkeypatch.setattr(gate, "random_probe", probe)
    return controls


@pytest.mark.parametrize("failure,message,difference", [
    ("roundtrip", "restoration differs", .25), ("final", "Exact CUDA replay failed", .25),
    ("timeout", "cooperative time limit", 0), ("after_load", "after restored RNG/state", 0),
    ("load", "checkpoint load failure", 0), ("rng", "RNG failure", 0),
])
def test_cpu_fake_replay_failure_never_promotes_success(tmp_path, cpu_replay, failure, message, difference):
    cpu_replay["failure"] = failure
    output = tmp_path / failure
    with pytest.raises((RuntimeError, TimeoutError), match=message):
        gate.run_gate(output)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert report["max_abs_difference"] == difference
    assert report["sources_unchanged"] is True
    assert report["model"] == {"layers": 2, "width": 3, "mlp_multiplier": 4}
    assert message in report["error"]


def test_source_snapshot_prerequisite_failure_retains_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "IMPORTED_SOURCE_ERROR", "OSError: CPU injected source read failure")
    monkeypatch.setattr(gate, "require_gpu", lambda: pytest.fail("must reject source error before CUDA"))
    with pytest.raises(RuntimeError, match="Source snapshot failed"):
        gate.run_gate(tmp_path / "source_failure")
    report = json.loads((tmp_path / "source_failure/metrics.json").read_text())
    assert report["status"] == "failed" and report["complete"] is False and report["host_gpu"] is False


@pytest.mark.parametrize("seconds", [0, -1, 61, float("nan"), float("inf")])
def test_invalid_duration_rejected_before_creating_attempt(tmp_path, seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        gate.run_gate(tmp_path / "invalid", max_seconds=seconds)
    assert not (tmp_path / "invalid").exists()
