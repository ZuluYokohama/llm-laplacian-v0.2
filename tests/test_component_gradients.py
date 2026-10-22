"""CPU-only contracts for the exploratory individual-component GPU diagnostic."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from laplacian.model import SpectralTransformer
from laplacian.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("component_gradient_test", ROOT / "scripts/check_component_gradients.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("bad", ["missing", "blank", "missing_ledger", "wrong_ledger", "device1"])
def test_budget_admission_precedes_every_cuda_query(monkeypatch, bad):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "fixture")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(ROOT / "runs/publication_study/budget.jsonl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if bad == "missing":
        monkeypatch.delenv("STUDY_BUDGET_RUN_ID")
    elif bad == "blank":
        monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "  ")
    elif bad == "missing_ledger":
        monkeypatch.delenv("STUDY_BUDGET_LEDGER")
    elif bad == "wrong_ledger":
        monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(ROOT / "wrong.jsonl"))
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before budget admission"))
    with pytest.raises(RuntimeError, match="canonical study-budget"):
        gate.require_gpu()


@pytest.mark.parametrize("name,bf16,message", [("CPU fake GTX", True, "5070"), ("RTX 5070", False, "BF16")])
def test_wrong_device_or_precision_is_rejected_with_cpu_fakes(monkeypatch, name, bf16, message):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "fixture")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(ROOT / "runs/publication_study/budget.jsonl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gate.budget, "read_ledger", lambda path: [])
    monkeypatch.setattr(gate.budget, "accounting", lambda rows: {
        "fixture": {"settled": False, "launch_authorized": True}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    with pytest.raises(RuntimeError, match=message):
        gate.require_gpu()


@pytest.mark.parametrize("state", [None, {"settled": True, "launch_authorized": True},
                                  {"settled": False, "launch_authorized": False}])
def test_env_strings_cannot_replace_active_reservation(monkeypatch, state):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "fixture")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(ROOT / "runs/publication_study/budget.jsonl"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gate.budget, "read_ledger", lambda path: [])
    monkeypatch.setattr(gate.budget, "accounting", lambda rows: {} if state is None else {"fixture": state})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA before reservation admission"))
    with pytest.raises(RuntimeError, match="actually launched open canonical reservation"):
        gate.require_gpu()


@pytest.mark.parametrize("component", gate.COMPONENTS)
@pytest.mark.parametrize("mutation", ["sign", "reweight", "erase"])
def test_one_component_gradient_error_cannot_hide_in_other_penalties(component, mutation):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.]))

        def forward(self, inputs):
            return {"value": self.weight * inputs.mean()}

    def penalties(output, *args, **kwargs):
        # Other components are enormous; only the selected unweighted term matters.
        return {name: output["value"].sum() * (1e-30 if name == component else 1e20)
                for name in gate.COMPONENTS}

    def corrupted(output, *args, **kwargs):
        result = penalties(output)
        original = result[component]
        factor = {"sign": -1., "reweight": .5, "erase": 0.}[mutation]
        result[component] = original * factor + original.detach() * (1-factor)
        return result

    model, config = Model(), {"layers_1index": [1], "exponent": .5}
    baseline, candidate = SimpleNamespace(spectral_penalties=penalties), SimpleNamespace(spectral_penalties=corrupted)
    for selected in gate.COMPONENTS:
        left = gate.component_gradient(model, [torch.ones(2, 4)], baseline, config, selected,
                                       optimized=False, device=torch.device("cpu"))
        right = gate.component_gradient(model, [torch.ones(2, 4)], candidate, config, selected,
                                        optimized=True, device=torch.device("cpu"))
        result = gate.compare_component(left[0], right[0], left[1], right[1])
        assert result["passed"] is (selected != component)
        assert result["value_passed"]  # Error injection preserves forward values.
        if selected == component:
            assert result["tiny_reference_norm"]
            assert not result["gradient_norm_passed"]
            assert result["gradient_relative_l2_error"] == pytest.approx({"sign": 2., "reweight": .5, "erase": 1.}[mutation])


def test_arbitrarily_small_nonzero_gradient_erasure_fails_despite_absolute_allowance():
    result = gate.compare_component(1., 1., {"p": torch.tensor([1e-30])}, {"p": torch.zeros(1)})
    assert result["tensor_guards_passed"]
    assert result["reference_gradient_l2"] > 0
    assert result["gradient_relative_l2_error"] == 1.
    assert not result["passed"]


def test_zero_reference_requires_exact_zero_and_matching_connectivity():
    same = {"connected": torch.zeros(2), "unused": None}
    result = gate.compare_component(0., 0., same, same)
    assert result["passed"] and result["gradient_relative_l2_error"] is None
    assert not result["nonzero_gradient_observed"]
    assert not gate.compare_component(0., 0., same, {"connected": torch.tensor([1e-40, 0.]), "unused": None})["passed"]
    assert not gate.compare_component(0., 0., same, {"connected": torch.zeros(2), "unused": torch.zeros(1)})["passed"]


def test_tensor_guard_catches_localized_error_hidden_by_other_parameter_norm():
    left = {"large": torch.tensor([1e6]), "small": torch.tensor([1.])}
    right = {"large": torch.tensor([1e6]), "small": torch.tensor([2.])}
    result = gate.compare_component(1., 1., left, right)
    assert result["gradient_norm_passed"] and not result["tensor_guards_passed"]
    assert not result["passed"]


@pytest.mark.parametrize("kind", ["value", "gradient", "shape", "missing"])
def test_nonfinite_or_mismatched_component_is_rejected(kind):
    reference, actual, value = {"p": torch.ones(2)}, {"p": torch.ones(2)}, 1.
    if kind == "value":
        value = float("nan")
    elif kind == "gradient":
        actual["p"][0] = float("inf")
    elif kind == "shape":
        actual["p"] = torch.ones(1)
    else:
        actual = {}
    result = gate.compare_component(1., value, reference, actual)
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_single_component_forward_value_error_is_rejected():
    assert not gate.compare_component(1., 1.1, {"p": torch.ones(1)}, {"p": torch.ones(1)})["passed"]


@pytest.fixture
def cpu_checkpoint(tmp_path, monkeypatch):
    previous_threads, tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(gate, "PRELOADED_LAPLACIAN", [])
    monkeypatch.setattr(gate, "require_gpu", lambda: torch.device("cpu"))
    monkeypatch.setattr(gate, "validate_production", lambda config: None)  # Explicit nonproduction fixture.
    monkeypatch.setattr(gate.study, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(gate.training, "build_model", lambda config, vocab, device:
                        SpectralTransformer(vocab=vocab, m=1, d_model=16, n_layers=1, context=config["context"]).to(device))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.tile(np.arange(13, dtype="<u2"), 40).tofile(bundle / "train.u16")
    (bundle / "train.stories.jsonl").write_text('{"story_id":"fixture","offset":0,"length":520}\n', encoding="utf-8")
    manifest = {"tokenizer": {"vocab_size": 13}, "files": {
        name: {key: value for key, value in gate.study.file_record(bundle / name).items() if key != "path"}
        for name in ("train.u16", "train.stories.jsonl")}}
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fixture_validator(path):
        assert Path(path).resolve() == manifest_path.resolve()
        for name, expected in manifest["files"].items():
            actual = gate.study.file_record(bundle / name)
            assert all(actual[key] == value for key, value in expected.items())
        return copy.deepcopy(manifest)

    monkeypatch.setattr(gate.study, "validate_bundle", fixture_validator)
    config = gate.study.make_pretrain_config(manifest_path, "synthetic_parent", 71, steps=50)
    config.update(microbatch=2, accum=2, context=4, layers_1index=[1], dead_units=False)
    config_path = gate.study.prepare_pretrain(config)
    seed_everything(71)
    model = gate.training.build_model(config, 13, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(83)
    checkpoint = config_path.parent / "last.pt"
    gate.training.save_checkpoint(checkpoint, model, optimizer, generator, 50, [], [], config=config)
    snapshot = checkpoint.parent / "state_step0000000.pt"
    gate.training.save_checkpoint(snapshot, model, optimizer, generator, 0, [], [], config=config)
    # A synthetic completed receipt tests admission; no actual 50-update claim.
    gate.write_metrics(checkpoint.parent / "metrics.json", {
        "complete": True, "steps": 50, "run_id": config["run_id"], "config": config,
        "study": {"config_sha256": gate.config_hash(config), "fractional_policy": gate.fractional_policy(),
                  "final_checkpoint": gate.study.file_record(checkpoint),
                  "timing_states": [gate.study.file_record(snapshot)]}})
    yield checkpoint, config, bundle, generator.get_state().clone()
    torch.set_num_threads(previous_threads)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def test_full_cpu_flow_restores_every_pass_and_preserves_source_run(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, config, bundle, sampler_state = cpu_checkpoint
    original = checkpoint.read_bytes()
    observed, capture = [], gate.component_gradient

    def inspect(model, *args, **kwargs):
        observed.append((gate.state_hash(model.state_dict()), float(torch.rand(()))))
        return capture(model, *args, **kwargs)

    monkeypatch.setattr(gate, "component_gradient", inspect)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: pytest.fail("Optimizer update in diagnostic"))
    output = tmp_path / "component_gate"
    report = gate.run_gate(checkpoint, output)
    assert report["complete"] and report["freshness_checks_passed"] and not report["host_gpu"]
    assert len(report["components"]) == 4 and all(row["passed"] for row in report["components"])
    assert len(observed) == 8 and len(set(observed)) == 1
    assert report["data"]["sampler_before"] == gate.tensor_hash(sampler_state)
    assert len(report["data"]["batches"]) == config["accum"]
    assert checkpoint.read_bytes() == original
    assert not list(checkpoint.parent.glob("samples-*.jsonl"))
    for record in report["source_snapshot"]:
        gate.study.check_record(record)
    metrics = output / "metrics.json"
    assert hashlib.sha256(metrics.read_text(encoding="utf-8").encode()).hexdigest() == (output / "metrics.json.sha256").read_text().strip()
    raw = metrics.read_bytes()
    with pytest.raises(FileExistsError):
        gate.run_gate(checkpoint, output)
    assert metrics.read_bytes() == raw


def test_declared_initial_snapshot_from_completed_run_is_admitted(cpu_checkpoint, tmp_path):
    checkpoint, _, _, _ = cpu_checkpoint
    report = gate.run_gate(checkpoint.parent / "state_step0000000.pt", tmp_path / "initial", phase=0)
    assert report["source_step"] == 0 and report["complete"]
    assert [row["component"] for row in report["components"]] == ["dirichlet"]


@pytest.mark.parametrize("mutation", ["incomplete", "checkpoint_bytes", "config_bytes", "undeclared_name"])
def test_json_admission_rejects_before_unpickle(cpu_checkpoint, tmp_path, monkeypatch, mutation):
    checkpoint, _, _, _ = cpu_checkpoint
    if mutation == "incomplete":
        path = checkpoint.parent / "metrics.json"
        metrics = gate.study.read_verified_metrics(path)
        metrics["complete"] = False
        gate.write_metrics(path, metrics)
    elif mutation == "checkpoint_bytes":
        with checkpoint.open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "config_bytes":
        original_read = gate.study.read_config

        def edit(path):
            config = original_read(path)
            with Path(path).open("ab") as handle:
                handle.write(b"\n")
            return config

        monkeypatch.setattr(gate.study, "read_config", edit)
    else:
        renamed = checkpoint.parent / "unlisted.pt"
        renamed.write_bytes(checkpoint.read_bytes())
        checkpoint = renamed
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Unpickled before admission"))
    with pytest.raises(ValueError):
        gate.run_gate(checkpoint, tmp_path / mutation)
    report = json.loads((tmp_path / mutation / "metrics.json").read_text())
    assert not report["complete"] and not report["components"]


@pytest.mark.parametrize("mutation", ["append", "delete"])
def test_changed_input_revokes_success(cpu_checkpoint, tmp_path, monkeypatch, mutation):
    checkpoint, _, bundle, _ = cpu_checkpoint
    capture, changed = gate.component_gradient, []

    def mutate(*args, **kwargs):
        result = capture(*args, **kwargs)
        if not changed:
            if mutation == "append":
                with (bundle / "train.u16").open("ab") as handle:
                    handle.write(b"\0\0")
            else:
                (bundle / "train.u16").unlink()
            changed.append(True)
        return result

    monkeypatch.setattr(gate, "component_gradient", mutate)
    output = tmp_path / "changed_input"
    with pytest.raises((RuntimeError, FileNotFoundError)):
        gate.run_gate(checkpoint, output)
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and report["status"] == "failed"
    assert all(row["passed"] for row in report["components"])


def test_completed_zero_gradient_component_is_explicitly_uninformative(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, _, _, _ = cpu_checkpoint
    capture = gate.component_gradient

    def zero_component(model, batches, module, config, component, **kwargs):
        if component == "dirichlet":
            model.zero_grad(set_to_none=True)
            return 0., {name: None for name, _ in model.named_parameters()}
        return capture(model, batches, module, config, component, **kwargs)

    monkeypatch.setattr(gate, "component_gradient", zero_component)
    report = gate.run_gate(checkpoint, tmp_path / "zero_component")
    assert report["complete"] and not report["all_active_components_nonzero"]
    row = next(row for row in report["components"] if row["component"] == "dirichlet")
    assert row["passed"] and row["matching_connectivity_and_shapes"]
    assert not row["nonzero_gradient_observed"]
    assert row["gradient_relative_l2_error"] is None


def test_one_bad_component_retains_failed_receipt(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, _, _, _ = cpu_checkpoint
    capture = gate.component_gradient

    def erase(*args, **kwargs):
        value, gradients = capture(*args, **kwargs)
        if args[4] == "hyper" and kwargs["optimized"]:
            gradients = {name: None if value is None else torch.zeros_like(value) for name, value in gradients.items()}
        return value, gradients

    monkeypatch.setattr(gate, "component_gradient", erase)
    output = tmp_path / "failed_component"
    with pytest.raises(ValueError, match="individual component"):
        gate.run_gate(checkpoint, output)
    report = json.loads((output / "metrics.json").read_text())
    assert report["status"] == "failed" and not report["complete"]
    assert [row["component"] for row in report["components"] if not row["passed"]] == ["hyper"]
    assert report["freshness_checks_passed"]


def test_source_drift_and_deadline_cannot_promote_success(cpu_checkpoint, tmp_path, monkeypatch):
    checkpoint, _, _, _ = cpu_checkpoint
    with pytest.raises(TimeoutError, match="cooperative"):
        gate.run_gate(checkpoint, tmp_path / "deadline", max_seconds=1e-12)
    report = json.loads((tmp_path / "deadline/metrics.json").read_text())
    assert not report["complete"]
    monkeypatch.setattr(gate, "source_hashes", lambda: {"changed": "bytes"})
    with pytest.raises(RuntimeError, match="unchanged package"):
        gate.run_gate(checkpoint, tmp_path / "changed_source")
    report = json.loads((tmp_path / "changed_source/metrics.json").read_text())
    assert not report["complete"] and "Source changed" in report["postcheck_error"]


def test_real_production_contract_accepts_shape_and_rejects_fixture():
    from laplacian.arms import get_config
    config = get_config("arm0")
    config["study"] = {"vocab_size": 50257, "stage": "ce_pretrain"}
    gate.validate_production(config)
    config["context"] = 4
    with pytest.raises(ValueError, match="production"):
        gate.validate_production(config)
