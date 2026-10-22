"""CPU contracts for the budgeted GPU rate probe; no GPU workload is executed."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from laplacian.arms import get_config
from laplacian.model import SpectralTransformer
from laplacian.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study_rate_test", ROOT / "scripts/benchmark_study_rate.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
PRODUCTION_VALIDATOR = probe.validate_probe_config


def test_budget_rejection_precedes_any_cuda_query(monkeypatch):
    monkeypatch.delenv("STUDY_BUDGET_RUN_ID", raising=False)
    monkeypatch.delenv("STUDY_BUDGET_LEDGER", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before budget rejection"))
    with pytest.raises(RuntimeError, match="canonical study-budget"):
        probe.require_gpu()
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "fixture")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(ROOT / "wrong-ledger.jsonl"))
    with pytest.raises(RuntimeError, match="canonical study-budget"):
        probe.require_gpu()


def test_ce_and_branch_configuration_and_intervention_boundary():
    config = get_config("arm0")
    config["study"] = {"stage": "ce_pretrain", "vocab_size": 50257, "penalty_step": None}
    assert not any(probe.validate_probe_config(config).values())
    config["spectral"] = True
    with pytest.raises(ValueError, match="CE rate probe"):
        probe.validate_probe_config(config)
    config["study"].update(stage="spectral_continuation", penalty_step=135000)
    assert probe.validate_probe_config(config) == {"manifold": .025, "fractional": .01, "hyper": .005, "dirichlet": .00055}
    config["context"] = 16
    with pytest.raises(ValueError, match="production"):
        probe.validate_probe_config(config)
    probe.reject_intervention_boundary(config, 50, 25)
    with pytest.raises(ValueError, match="resampling boundary"):
        probe.reject_intervention_boundary(config, 990, 25)
    probe.reject_intervention_boundary(config, 29990, 25)  # Production intervention ends before 30k.


@pytest.mark.parametrize("mutation,message", [
    ({"stage": "unknown"}, "CE or fixed-weight study stage"),
    ({"vocab_size": 128}, "pinned production GPT-2 vocabulary"),
    ({"penalty_step": 0}, "fixed all-active step135000"),
])
def test_production_admission_rejects_wrong_study_identity(mutation, message):
    config = get_config("arm2")
    config["study"] = {"stage": "spectral_continuation", "vocab_size": 50257, "penalty_step": 135000}
    config["study"].update(mutation)
    with pytest.raises(ValueError, match=message):
        probe.validate_probe_config(config)


def test_continuation_requires_every_active_weight(monkeypatch):
    config = get_config("arm2")
    config["study"] = {"stage": "spectral_continuation", "vocab_size": 50257, "penalty_step": 135000}
    monkeypatch.setattr(probe.training, "penalty_weights", lambda *args:
                        {"manifold": .025, "fractional": 0., "hyper": .005, "dirichlet": .00055})
    with pytest.raises(ValueError, match="fixed all-active step135000"):
        probe.validate_probe_config(config)


@pytest.fixture
def cpu_checkpoint(tmp_path, monkeypatch):
    study, training = probe.study, probe.training
    previous_threads, tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(study, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "log_line", lambda line: None)
    monkeypatch.setattr(training, "build_model", lambda config, vocab, device:
                        SpectralTransformer(vocab=vocab, m=1, d_model=16, n_layers=1, context=config["context"]).to(device))
    monkeypatch.setattr(training, "read_meta", lambda: pytest.fail("historical data accessed"))
    monkeypatch.setattr(training, "TokenStore", lambda *args, **kwargs: pytest.fail("historical data accessed"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.tile(np.arange(13, dtype="<u2"), 40).tofile(bundle / "train.u16")
    (bundle / "train.stories.jsonl").write_text('{"story_id":"fixture","offset":0,"length":520}\n')
    manifest = {"tokenizer": {"vocab_size": 13}, "files": {
        name: {key: value for key, value in study.file_record(bundle / name).items() if key != "path"}
        for name in ("train.u16", "train.stories.jsonl")}}
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fixture_validator(path):
        assert Path(path).resolve() == manifest_path.resolve()
        for name, expected in manifest["files"].items():
            actual = study.file_record(bundle / name)
            assert all(actual[key] == value for key, value in expected.items())
        return copy.deepcopy(manifest)

    monkeypatch.setattr(study, "validate_bundle", fixture_validator)
    config = study.make_pretrain_config(manifest_path, "pretrain", 7, steps=50)
    config.update(microbatch=2, accum=2, context=4, layers_1index=[1], dead_units=False)
    config_path = study.prepare_pretrain(config)
    seed_everything(7)
    model = training.build_model(config, 13, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(83)
    inputs = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    batch = (inputs, inputs+1)
    probe.ordinary_update(model, optimizer, [batch, batch], config, 0, torch.device("cpu"))
    checkpoint = config_path.parent / "last.pt"
    training.save_checkpoint(checkpoint, model, optimizer, generator, 1, [], [], config=config)
    # The run path is exercised on CPU with explicit fixture-only substitutions.
    monkeypatch.setattr(probe, "require_gpu", lambda: torch.device("cpu"))
    monkeypatch.setattr(probe, "validate_probe_config", lambda cfg: training.penalty_weights(cfg, 0))
    monkeypatch.setattr(probe, "memory_guard", lambda device: 0.)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0.)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0.)
    yield checkpoint, config, generator.get_state().clone(), bundle
    torch.set_num_threads(previous_threads)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def test_cpu_probe_restores_sampler_and_optimizer_leaves_source_unchanged(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, config, sampler_state, bundle = cpu_checkpoint
    original = checkpoint.read_bytes()
    observed_optimizer_steps = []
    ordinary = probe.ordinary_update

    def observe(model, optimizer, *args, **kwargs):
        observed_optimizer_steps.append(float(next(iter(optimizer.state.values()))["step"]))
        return ordinary(model, optimizer, *args, **kwargs)

    monkeypatch.setattr(probe, "ordinary_update", observe)
    output = tmp_path / "probe"
    report = probe.run_probe(checkpoint, output, warmup=1, steps=2)
    assert report["complete"] and report["freshness_checks_passed"]
    assert report["source_step"] == 1 and report["probe_virtual_step_end"] == 4
    assert len(report["updates"]) == 3 and sum(not row["warmup"] for row in report["updates"]) == 2
    assert len(report["data"]["batches"]) == 6
    assert observed_optimizer_steps == [1., 2., 3.]
    assert report["data"]["sampler_state_sha256_before"] == probe.tensor_hash(sampler_state)
    restored = torch.Generator().set_state(sampler_state)
    starts = torch.randint(0, 520-config["context"]-1, (config["microbatch"],), generator=restored).tolist()
    draws = [json.loads(line) for line in Path(report["data"]["sample_log"]["path"]).read_text().splitlines()]
    assert draws[1]["starts"] == starts
    tokens = np.memmap(bundle / "train.u16", dtype="<u2", mode="r")
    packed = np.stack([np.array(tokens[start:start+5], dtype=np.int64) for start in starts])
    assert report["data"]["batches"][0]["inputs_sha256"] == probe.tensor_hash(torch.from_numpy(packed[:, :-1]))
    assert checkpoint.read_bytes() == original
    assert not list(checkpoint.parent.glob("samples-*.jsonl"))
    metrics = output / "metrics.json"
    before = metrics.read_bytes()
    assert hashlib.sha256(metrics.read_text().encode()).hexdigest() == (output / "metrics.json.sha256").read_text().strip()
    with pytest.raises(FileExistsError):
        probe.run_probe(checkpoint, output)
    assert metrics.read_bytes() == before


def test_input_drift_revokes_completed_probe(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, _, _, bundle = cpu_checkpoint
    ordinary = probe.ordinary_update

    def mutate(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        with (bundle / "train.u16").open("ab") as handle:
            handle.write(b"\x00\x00")
        return result

    monkeypatch.setattr(probe, "ordinary_update", mutate)
    output = tmp_path / "drift"
    with pytest.raises(RuntimeError, match="Bound input changed"):
        probe.run_probe(checkpoint, output, warmup=1, steps=1)
    receipt = json.loads((output / "metrics.json").read_text())
    assert receipt["complete"] is False and receipt["status"] == "failed"
    assert "Bound input changed" in receipt["postcheck_error"]


def test_configuration_edit_during_load_cannot_bind_new_bytes_to_old_memory(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, _, _, _ = cpu_checkpoint
    config_path = checkpoint.parent / "study_config.json"
    original_record = probe.study.file_record(config_path)
    read_config = probe.study.read_config

    def edit_after_read(path):
        config = read_config(path)
        # Valid JSON with the same semantic config still has a different raw identity.
        with Path(path).open("ab") as handle:
            handle.write(b"\n")
        return config

    monkeypatch.setattr(probe.study, "read_config", edit_after_read)
    output = tmp_path / "configuration_race"
    with pytest.raises(ValueError, match="Study bound file changed"):
        probe.run_probe(checkpoint, output, warmup=1, steps=1)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert report["updates"] == []
    assert report["inputs_before"] == [original_record]
    assert report["inputs_after"][0] != original_record
    assert "Bound input changed" in report["postcheck_error"]


def test_nonfinite_loss_prevents_optimizer_mutation(cpu_checkpoint, monkeypatch):
    _, config, _, _ = cpu_checkpoint
    model = probe.training.build_model(config, 13, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    monkeypatch.setattr(probe.training, "forward_loss", lambda *args:
                        (torch.tensor(float("nan"), requires_grad=True), torch.tensor(1.), {}, None))
    with pytest.raises(FloatingPointError, match="before optimizer update"):
        probe.ordinary_update(model, optimizer, [(torch.zeros((2, 4), dtype=torch.long),)*2], config, 1, torch.device("cpu"))
    assert not optimizer.state
    assert all(torch.equal(left, right) for left, right in zip(before, model.parameters()))


def test_spectral_update_uses_fixed_weights_on_cpu(cpu_checkpoint, monkeypatch):
    _, config, _, _ = cpu_checkpoint
    config = copy.deepcopy(config)
    config["spectral"] = True
    config["study"].update(stage="spectral_continuation", penalty_step=135000)
    config["lr"], config["min_lr"], config["warmup"] = 3e-5, 3e-5, 0
    model = probe.training.build_model(config, 13, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters())
    called = []
    actual = probe.training.spectral_penalties

    def observe(*args, **kwargs):
        called.append(kwargs["weights"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(probe.training, "spectral_penalties", observe)
    inputs = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    values, _, _ = probe.ordinary_update(model, optimizer, [(inputs, inputs+1)]*2, config, 1, torch.device("cpu"))
    assert len(called) == 2 and all(weight > 0 for weights in called for weight in weights.values())
    assert values["loss"] >= values["ce"] and values["lr"] == 3e-5


@pytest.mark.parametrize("kwargs,message", [
    ({"warmup": 0}, "positive integers"), ({"steps": 0}, "positive integers"),
    ({"warmup": 1.5}, "positive integers"), ({"warmup": True}, "positive integers"),
    ({"steps": 2.5}, "positive integers"), ({"max_seconds": 121}, "Cooperative limit"),
    ({"max_seconds": float("nan")}, "Cooperative limit"),
    ({"max_seconds": float("inf")}, "Cooperative limit"),
    ({"max_seconds": 0}, "Cooperative limit"), ({"max_seconds": -1}, "Cooperative limit"),
])
def test_invalid_bounds_never_create_output(tmp_path, kwargs, message):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=message):
        probe.run_probe(tmp_path / "missing.pt", output, **kwargs)
    assert not output.exists()


def test_run_probe_real_validator_rejects_nonproduction_config(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, _, _, _ = cpu_checkpoint
    monkeypatch.setattr(probe, "validate_probe_config", PRODUCTION_VALIDATOR)
    output = tmp_path / "invalid_production_config"
    with pytest.raises(ValueError, match="production M4/k0/microbatch8/accum8/context256"):
        probe.run_probe(checkpoint, output, warmup=1, steps=1)
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and report["status"] == "failed" and report["updates"] == []


@pytest.mark.parametrize("failure", ["incomplete_state", "environment_drift", "pulses", "stripped"])
def test_incomplete_or_intervened_resume_is_rejected(cpu_checkpoint, tmp_path, monkeypatch, failure):
    checkpoint, _, _, _ = cpu_checkpoint
    original = probe.training.load_checkpoint

    def invalid_resume(*args, **kwargs):
        step, pulses, stripped, fired, tokens = original(*args, **kwargs)
        metadata = kwargs["resume_metadata"]
        if failure == "incomplete_state":
            metadata["resume_status"] = "legacy_partial"
        elif failure == "environment_drift":
            metadata["resume_limitations"] = ["changed environment"]
        elif failure == "pulses":
            pulses = [1]
        else:
            stripped = [1]
        return step, pulses, stripped, fired, tokens

    monkeypatch.setattr(probe.training, "load_checkpoint", invalid_resume)
    output = tmp_path / failure
    with pytest.raises(ValueError, match="complete checkpoint state in the same environment"):
        probe.run_probe(checkpoint, output, warmup=1, steps=1)
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and report["status"] == "failed" and report["updates"] == []


def test_copied_checkpoint_and_config_outside_bound_run_are_rejected(cpu_checkpoint, tmp_path):
    checkpoint, _, _, _ = cpu_checkpoint
    copied_folder = tmp_path / "wrong_run"
    copied_folder.mkdir()
    copied = copied_folder / "last.pt"
    copied.write_bytes(checkpoint.read_bytes())
    (copied_folder / "study_config.json").write_bytes((checkpoint.parent / "study_config.json").read_bytes())
    output = tmp_path / "wrong_checkpoint"
    # read_config rejects the moved run before checkpoint deserialization.
    with pytest.raises(ValueError, match="Study configuration must remain in its bound run directory"):
        probe.run_probe(copied, output, warmup=1, steps=1)
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and report["updates"] == []


def test_hardware_rejection_persists_incomplete_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "require_gpu", lambda: (_ for _ in ()).throw(RuntimeError("CPU fixture rejection")))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="CPU fixture rejection"):
        probe.run_probe(tmp_path / "missing.pt", output)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed" and not report["host_gpu"]


def test_source_drift_is_retained_after_an_independent_failure(tmp_path, monkeypatch):
    calls = []

    def hashes():
        calls.append(None)
        return probe.IMPORTED_SOURCE_HASHES if len(calls) == 1 else {"changed": "source"}

    monkeypatch.setattr(probe, "source_hashes", hashes)
    monkeypatch.setattr(probe, "require_gpu", lambda: (_ for _ in ()).throw(RuntimeError("CPU fixture rejection")))
    output = tmp_path / "source_drift"
    with pytest.raises(RuntimeError, match="CPU fixture rejection"):
        probe.run_probe(tmp_path / "missing.pt", output)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False
    assert "Source changed" in report["postcheck_error"]


def test_cooperative_deadline_never_promotes_success(cpu_checkpoint, tmp_path):
    checkpoint, _, _, _ = cpu_checkpoint
    output = tmp_path / "deadline"
    with pytest.raises(TimeoutError, match="cooperative limit"):
        probe.run_probe(checkpoint, output, warmup=1, steps=1, max_seconds=1e-12)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert report["updates"] == []
