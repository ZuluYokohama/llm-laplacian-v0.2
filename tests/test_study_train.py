"""CPU integration fixtures: token validation itself is covered in study_tokens tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from laplacian import study_train as study
from laplacian import train as training
from laplacian.model import SpectralTransformer
from laplacian.reproducibility import config_hash, seed_everything
from laplacian.schedule import gammas


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            nested_equal(a, b)
    else:
        assert left == right


@pytest.fixture
def cpu_study(tmp_path, monkeypatch):
    threads, tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(study, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "apply_env", lambda: None)
    monkeypatch.setattr(training, "pick_5070", lambda: (torch.device("cpu"), "CPU fixture"))
    monkeypatch.setattr(training, "memory_mib", lambda: 0.0)
    monkeypatch.setattr(training, "enforce_guard", lambda: None)
    monkeypatch.setattr(training, "log_line", lambda line: None)
    monkeypatch.setattr(training, "build_model", lambda config, vocab, device:
                        SpectralTransformer(vocab=vocab, m=1, d_model=16, n_layers=1, context=config["context"]))
    # Any accidental fallback to historical data must fail, even if it exists locally.
    monkeypatch.setattr(training, "read_meta", lambda: pytest.fail("historical metadata accessed"))
    monkeypatch.setattr(training, "TokenStore", lambda *a, **k: pytest.fail("historical tokens accessed"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.tile(np.arange(13, dtype="<u2"), 16).tofile(bundle / "train.u16")
    (bundle / "train.stories.jsonl").write_text('{"story_id":"fixture","offset":0,"length":208}\n')
    manifest = {"tokenizer": {"vocab_size": 13}, "files": {
        name: {key: value for key, value in study.file_record(bundle / name).items() if key != "path"}
        for name in ("train.u16", "train.stories.jsonl")}}
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def fixture_validator(path):
        assert Path(path).resolve() == manifest_path.resolve()
        for name, expected in manifest["files"].items():
            actual = study.file_record(bundle / name)
            if any(actual[key] != value for key, value in expected.items()):
                raise ValueError("fixture token data changed")
        return copy.deepcopy(manifest)

    monkeypatch.setattr(study, "validate_bundle", fixture_validator)
    config = study.make_pretrain_config(manifest_path, "pretrain", 7, steps=3)
    config.update(microbatch=2, accum=2, context=4, layers_1index=[1], dead_units=False)
    config["study"]["checkpoint_every"] = 1
    yield config
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def run_pretrain(config):
    path = study.prepare_pretrain(config)
    study.execute(path)
    return path.parent / "last.pt"


def test_train_uses_manifest_only_and_records_reconstructable_draws(cpu_study):
    checkpoint = run_pretrain(cpu_study)
    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    assert metrics["complete"] and not metrics["host_gpu"]
    assert metrics["study"]["freshness_checked_at_completion"]
    assert [Path(row["path"]).name for row in metrics["study"]["timing_states"]] == ["state_step0000000.pt", "state_step0000001.pt", "state_step0000003.pt"]
    records = [json.loads(line) for line in Path(metrics["study"]["sample_logs"][0]["path"]).read_text().splitlines()]
    assert records[0]["window_tokens"] == 5
    draws = records[1:]
    assert [(row["step"], row["micro"]) for row in draws] == [(s, m) for s in range(3) for m in range(2)]
    generator = torch.Generator().manual_seed(7)
    for row in draws:
        assert row["starts"] == torch.randint(0, 208 - 4 - 1, (2,), generator=generator).tolist()
        assert all(0 <= start and start + 5 <= 208 for start in row["starts"])
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        study.execute(checkpoint.parent / "study_config.json")
    assert checkpoint.read_bytes() == before
    with pytest.raises(FileExistsError):
        study.prepare_pretrain(cpu_study)


def test_forks_preserve_all_parent_states_and_share_draws(cpu_study):
    parent_path = run_pretrain(cpu_study)
    parent = torch.load(parent_path, weights_only=False)
    original = study.file_record(parent_path)
    optimized_path = study.fork_checkpoint(parent_path, "optimized", loss_implementation="optimized", steps=2)
    baseline_path = study.fork_checkpoint(parent_path, "baseline", loss_implementation="baseline", steps=2)
    optimized = torch.load(optimized_path.parent / "last.pt", weights_only=False)
    baseline = torch.load(baseline_path.parent / "last.pt", weights_only=False)
    for key in ("model", "optimizer", "generator", "rng_state", "requires_grad", "module_training", "pulses_done", "stripped", "fired_tokens", "environment"):
        nested_equal(parent[key], optimized[key])
        nested_equal(parent[key], baseline[key])
    for fork in (optimized, baseline):
        assert fork["step"] == 0
        cfg = fork["training_config"]
        assert cfg["steps"] == 2 and cfg["warmup"] == 0 and cfg["lr"] == cfg["min_lr"] == 3e-5
        assert not cfg["dead_units"]
        assert cfg["study"]["lineage"]["checkpoint"] == original
        assert cfg["study"]["lineage"]["parent_elapsed_seconds_total"] > 0
        assert fork["reproducibility"]["elapsed_seconds_total"] == 0
        assert training.penalty_weights(cfg, 0) == training.penalty_weights(cfg, 9999)
        assert training.penalty_weights(cfg, 0) == {key: gammas(135000)[key] for key in ("manifold", "fractional", "hyper", "dirichlet")}
    study.execute(optimized_path)
    study.execute(baseline_path)
    draws = []
    for path in (optimized_path, baseline_path):
        sample_file = next(path.parent.glob("samples-*.jsonl"))
        draws.append([json.loads(line) for line in sample_file.read_text().splitlines()[1:]])
    assert draws[0] == draws[1]
    assert study.file_record(parent_path) == original


def test_interrupted_study_resume_matches_uninterrupted(cpu_study, monkeypatch):
    whole_config = copy.deepcopy(cpu_study)
    whole_config["run_id"] = "whole"
    whole_path = run_pretrain(whole_config)
    whole = torch.load(whole_path, weights_only=False)
    path = study.prepare_pretrain(cpu_study)
    original = training.checked_optimizer_step
    original_save = training.save_checkpoint

    class Interrupt(Exception):
        pass

    # Interrupt after the first committed update, then execute the remaining
    # updates with restored full model/optimizer/global and local RNG state.
    def interrupt_after_save(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[4] == 1:
            raise Interrupt()

    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_save)
    with pytest.raises(Interrupt):
        study.execute(path)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    seed_everything(999)
    study.execute(path)
    resumed = torch.load(path.parent / "last.pt", weights_only=False)
    for key in ("model", "optimizer", "rng_state", "generator", "step"):
        nested_equal(whole[key], resumed[key])
    assert training.checked_optimizer_step is original
    assert len(list(path.parent.glob("samples-*.jsonl"))) == 2
    assert json.loads((path.parent / "metrics.json").read_text())["resumed_from_step"] == 1


def test_source_data_and_policy_changes_fail_before_training(cpu_study, monkeypatch):
    original = study.source_records()
    with monkeypatch.context() as patch:
        patch.setattr(study, "source_records", lambda: original[:-1])
        with pytest.raises(ValueError, match="source files changed"):
            study.validate_config(cpu_study)
    changed = copy.deepcopy(cpu_study)
    changed["study"]["fractional_policy"]["cutoff"] *= 2
    with pytest.raises(ValueError, match="fractional policy changed"):
        study.validate_config(changed)


def test_data_or_manifest_change_and_missing_manifest_are_rejected(cpu_study):
    manifest = Path(cpu_study["study"]["token_manifest"]["path"])
    tokens = manifest.parent / "train.u16"
    data = tokens.read_bytes()
    tokens.write_bytes(data + b"\0\0")
    with pytest.raises(ValueError, match="token data changed"):
        study.validate_config(cpu_study)
    tokens.write_bytes(data)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="bound file changed"):
        study.validate_config(cpu_study)


def test_changed_config_legacy_parent_and_unapproved_baseline_fail(cpu_study, tmp_path):
    parent_path = run_pretrain(cpu_study)
    parent = torch.load(parent_path, weights_only=False)
    changed = copy.deepcopy(cpu_study)
    changed["steps"] += 1
    with pytest.raises(ValueError, match="configuration differs"):
        study.validate_checkpoint(parent, changed)
    legacy = copy.deepcopy(parent)
    del legacy["rng_state"]
    with pytest.raises(ValueError, match="full current trusted"):
        study.validate_checkpoint(legacy, cpu_study)
    legacy = copy.deepcopy(parent)
    legacy["reproducibility"]["resume_limitations"] = ["non-exact"]
    with pytest.raises(ValueError, match="complete RNG"):
        study.validate_checkpoint(legacy, cpu_study)
    evil = tmp_path / "other_losses.py"
    evil.write_text("raise RuntimeError('must never import')")
    with pytest.raises(ValueError, match="immutable prior"):
        study.fork_checkpoint(parent_path, "bad_baseline", loss_implementation="baseline", baseline_loss_source=evil)
    assert not (study.RUNS / "bad_baseline").exists()
    with pytest.raises(ValueError, match="escapes RUNS"):
        study.fork_checkpoint(parent_path, "../../escape", loss_implementation="optimized")


def test_fork_rejects_mutated_completion_receipt_and_checkpoint(cpu_study):
    parent_path = run_pretrain(cpu_study)
    metrics = parent_path.parent / "metrics.json"
    original_metrics = metrics.read_bytes()
    metrics.write_text(metrics.read_text() + " ")
    with pytest.raises(ValueError, match="sidecar digest"):
        study.fork_checkpoint(parent_path, "bad_receipt", loss_implementation="optimized")
    metrics.write_bytes(original_metrics)
    parent = torch.load(parent_path, weights_only=False)
    parent["model"]["tok.weight"][0, 0] += 0.01
    torch.save(parent, parent_path)
    with pytest.raises(ValueError, match="checkpoint bytes"):
        study.fork_checkpoint(parent_path, "bad_checkpoint", loss_implementation="optimized")


@pytest.mark.parametrize("mutation,message", [
    ("name", "final last.pt checkpoint"), ("sidecar", "sidecar digest"),
    ("incomplete", "Parent completion metrics"), ("step", "Parent completion metrics"),
    ("bytes", "checkpoint bytes"), ("directory", "final last.pt checkpoint"),
    ("config", "Study fractional policy changed"),
])
def test_fork_checks_name_receipt_and_bytes_before_unpickling(cpu_study, monkeypatch, mutation, message):
    parent = run_pretrain(cpu_study)
    metrics_path = parent.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if mutation == "name":
        parent = parent.with_name("weights_final.pt")
    elif mutation == "sidecar":
        metrics_path.write_text(metrics_path.read_text() + " ")
    elif mutation in ("incomplete", "step"):
        metrics["complete" if mutation == "incomplete" else "steps"] = False if mutation == "incomplete" else 999
        study.write_metrics(metrics_path, metrics)
    elif mutation == "config":
        metrics["config"]["study"]["fractional_policy"] = {"invalid": "changed policy"}
        study.write_metrics(metrics_path, metrics)  # Valid sidecar; config itself must fail before unpickling.
    elif mutation == "bytes":
        with parent.open("ab") as handle:
            handle.write(b"changed checkpoint bytes")
    else:
        moved = parent.parent.with_name("moved_parent")
        moved.mkdir()
        copied = moved / "last.pt"
        copied.write_bytes(parent.read_bytes())
        parent = copied
        metrics["study"]["final_checkpoint"] = study.file_record(parent)
        study.write_metrics(moved / "metrics.json", metrics)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Rejected parent was unpickled"))
    with pytest.raises(ValueError, match=message):
        study.fork_checkpoint(parent, "rejected_before_load", loss_implementation="optimized")
    assert not (study.RUNS / "rejected_before_load").exists()


def test_fork_rechecks_completion_identity_after_checkpoint_load(cpu_study, monkeypatch):
    parent = run_pretrain(cpu_study)
    metrics = parent.parent / "metrics.json"
    original = torch.load

    def changed_during_load(*args, **kwargs):
        payload = original(*args, **kwargs)
        metrics.write_text(metrics.read_text() + " ")
        return payload

    monkeypatch.setattr(torch, "load", changed_during_load)
    with pytest.raises(ValueError, match="Study bound file changed"):
        study.fork_checkpoint(parent, "changed_during_load", loss_implementation="optimized")
    assert not (study.RUNS / "changed_during_load").exists()


def test_resume_rejects_incomplete_flags_and_firing_window(cpu_study):
    parent_path = run_pretrain(cpu_study)
    parent = torch.load(parent_path, weights_only=False)
    model = training.build_model(cpu_study, 13, torch.device("cpu"))
    del parent["requires_grad"]["tok.weight"]
    with pytest.raises(ValueError, match="flag keysets"):
        study.validate_checkpoint(parent, cpu_study, model=model)
    parent = torch.load(parent_path, weights_only=False)
    parent["fired_tokens"] = 5
    parent.pop("fired_sums", None)
    with pytest.raises(ValueError, match="firing-window sums"):
        study.validate_checkpoint(parent, cpu_study)


def test_completion_freshness_failure_does_not_claim_complete(cpu_study, monkeypatch):
    path = study.prepare_pretrain(cpu_study)
    original = study.validate_config
    checks = []

    def changed_at_completion(config):
        checks.append(True)
        # read_config + train startup are valid; completion detects drift.
        if len(checks) > 2:
            raise ValueError("source drift at completion")
        return original(config)

    monkeypatch.setattr(study, "validate_config", changed_at_completion)
    with pytest.raises(ValueError, match="source drift"):
        study.execute(path)
    assert not (path.parent / "metrics.json").exists()


def test_run_lock_rejects_concurrency(cpu_study):
    path = study.prepare_pretrain(cpu_study)
    with study.run_lock(path.parent):
        with pytest.raises(OSError):
            with study.run_lock(path.parent):
                pytest.fail("concurrent lock admitted")


def test_cli_resume_rejects_overrides_and_baseline_oneoff_flag():
    with pytest.raises(SystemExit):
        study.main(["train", "--config", "x", "--steps", "3"])
    with pytest.raises(SystemExit):
        study.main(["fork", "--parent-checkpoint", "x", "--run-id", "y", "--loss-implementation", "optimized", "--baseline-loss-source", "x"])
