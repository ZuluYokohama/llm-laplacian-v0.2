"""A benchmark must reject numerical regressions, even when timing improves."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_optimization.py"
SPEC = importlib.util.spec_from_file_location("benchmark_optimization", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def compare(left, right, old_loss=2.0, new_loss=2.0):
    return benchmark.compare_gradients(
        left, right, {"loss": old_loss, "ce": 1.0}, {"loss": new_loss, "ce": 1.0},
        loss_rtol=1e-4, grad_rtol=1e-3, atol=1e-7,
    )


def test_identical_gradients_and_unused_parameter_pass():
    gradients = {"weight": torch.tensor([1.0, 0.0, -2.0]), "unused": None}
    result = compare(gradients, gradients)
    assert result["passed"]
    assert result["gradient_relative_l2"] == 0


@pytest.mark.parametrize("candidate", [torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("inf")])])
def test_nonfinite_gradients_fail(candidate):
    result = compare({"weight": torch.ones(2)}, {"weight": candidate})
    assert not result["passed"]
    assert not result["finite"]


def test_missing_parameter_gradient_fails():
    result = compare({"weight": torch.ones(2)}, {"weight": None})
    assert not result["passed"]
    assert result["missing_or_shape_mismatch"] == ["weight"]


@pytest.mark.parametrize("reference,actual,missing", [
    ({"weight": torch.ones(2)}, {"weight": torch.ones(3)}, ["weight"]),
    ({"weight": None}, {"weight": torch.ones(2)}, ["weight"]),
    ({"weight": torch.ones(2)}, {"weight": torch.ones(2), "extra": None}, ["extra"]),
    ({"weight": torch.ones(2), "unused": None}, {"weight": torch.ones(2)}, ["unused"]),
])
def test_shape_presence_and_key_set_mismatches_fail(reference, actual, missing):
    result = compare(reference, actual)
    assert not result["passed"] and not result["matching_gradients"]
    assert result["missing_or_shape_mismatch"] == missing


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_fails_with_finite_gradients(value):
    result = compare({"weight": torch.ones(2)}, {"weight": torch.ones(2)}, new_loss=value)
    assert not result["passed"] and not result["finite"]


def test_component_regression_fails_even_when_total_loss_matches():
    gradients = {"weight": torch.ones(2)}
    result = benchmark.compare_gradients(
        gradients, gradients, {"loss": 10., "ce": 10., "penalty_hyper": 1.},
        {"loss": 10., "ce": 10., "penalty_hyper": 2.},
        loss_rtol=1e-4, grad_rtol=1e-3, atol=1e-7,
    )
    assert not result["passed"]
    assert result["value_absolute_errors"]["penalty_hyper"] == 1.


def test_loss_regression_fails_with_identical_gradients():
    gradients = {"weight": torch.ones(2)}
    assert not compare(gradients, gradients, new_loss=3.0)["passed"]


def test_gradient_regression_fails_with_identical_loss():
    assert not compare({"weight": torch.ones(2)}, {"weight": torch.zeros(2)})["passed"]


def test_localized_gradient_regression_fails_global_max_gate():
    reference = torch.ones(100000)
    actual = reference.clone()
    actual[0] += 0.1
    result = compare({"weight": reference}, {"weight": actual})
    assert result["gradient_relative_l2"] < 1e-3
    assert not result["passed"]


def test_existing_metrics_are_never_overwritten(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text("previous evidence", encoding="utf-8")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        benchmark.main(["--baseline-source", str(tmp_path / "unused.py"), "--output-dir", str(tmp_path)])
    assert metrics.read_text(encoding="utf-8") == "previous evidence"


@pytest.mark.parametrize("case", ["missing_run", "blank_run", "missing_ledger", "wrong_ledger", "nonzero_device"])
def test_cuda_admission_precedes_every_cuda_query(monkeypatch, tmp_path, case):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "cpu-admission-fixture")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(benchmark.ROOT / "runs/publication_study/budget.jsonl"))
    device = "cuda:0"
    if case == "missing_run":
        monkeypatch.delenv("STUDY_BUDGET_RUN_ID")
    elif case == "blank_run":
        monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "  \t")
    elif case == "missing_ledger":
        monkeypatch.delenv("STUDY_BUDGET_LEDGER")
    elif case == "wrong_ledger":
        monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(tmp_path / "other.jsonl"))
    else:
        device = "cuda:1"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before admission"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: pytest.fail("CUDA queried before admission"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: pytest.fail("CUDA initialized before admission"))
    with pytest.raises(RuntimeError, match="canonical study-budget wrapper and visible device0"):
        benchmark.admit_device(device)


@pytest.mark.parametrize("available,name,bf16,message", [
    (False, "NVIDIA GeForce RTX 5070", True, "approved RTX 5070"),
    (True, "NVIDIA GeForce RTX 4090", True, "approved RTX 5070"),
    (True, "NVIDIA GeForce RTX 5070", False, "BF16 support"),
])
def test_cuda_hardware_rejection_uses_cpu_fakes(monkeypatch, available, name, bf16, message):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "cpu-admission-fixture")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(benchmark.ROOT / "runs/publication_study/budget.jsonl"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: name)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: pytest.fail("Admission must not initialize a device"))
    with pytest.raises(RuntimeError, match=message):
        benchmark.admit_device("cuda:0")


def test_main_cuda_admission_failure_preserves_receipt_before_baseline_exec(tmp_path, monkeypatch):
    fixture_sources(monkeypatch)
    monkeypatch.delenv("STUDY_BUDGET_RUN_ID", raising=False)
    monkeypatch.delenv("STUDY_BUDGET_LEDGER", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before budget admission"))
    monkeypatch.setattr(benchmark, "load_baseline", lambda *args: pytest.fail("Baseline executed before budget admission"))
    output = tmp_path / "admission"
    with pytest.raises(RuntimeError, match="canonical study-budget"):
        benchmark.main(["--baseline-source", str(tmp_path / "unused.py"), "--output-dir", str(output), "--device", "cuda:0"])
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and not report["host_gpu"] and report["cases"] == []


BASELINE_SOURCE = '''import torch
def spectral_penalties(output, layers, exponent):
    value = output["signal"].square().sum()
    return {name: value * (index + 1) for index, name in enumerate(("manifold", "fractional", "hyper", "dirichlet"))}
def total_loss(ce, penalties, weights):
    return ce + sum(weights[name] * value for name, value in penalties.items())
'''


def fixture_sources(monkeypatch):
    identity = {"fixture_source.py": "stable"}
    monkeypatch.setattr(benchmark, "sources", lambda: identity.copy())
    monkeypatch.setattr(benchmark, "IMPORTED_SOURCE_HASHES", identity.copy())
    monkeypatch.setattr(benchmark, "POST_IMPORT_SOURCE_HASHES", identity.copy())
    monkeypatch.setattr(benchmark, "PREEXISTING_LAPLACIAN_MODULES", [])


@pytest.fixture
def tiny_main(tmp_path, monkeypatch):
    """Exercise the real main/capture/optimizer path on an explicit CPU fixture."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    fixture_sources(monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.json").write_text('{"vocab": 2}', encoding="utf-8")
    (data / "tinystories_val.u16").write_bytes(b"\x00\x00\x01\x00")
    monkeypatch.setattr(benchmark, "DATA", data)
    monkeypatch.setattr(benchmark, "git_info", lambda: {"fixture": True})
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda name: "cpu-fixture")
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: pytest.fail("CPU test queried CUDA"))
    config = {"m": 1, "k": 0, "exponent": .5, "lr": .001, "betas": (.9, .95), "weight_decay": 0., "clip": 1.}
    monkeypatch.setattr(benchmark, "get_config", lambda name: config.copy())
    monkeypatch.setattr(benchmark, "layers_0index", lambda cfg: [0])
    weights = {name: .01 for name in ("manifold", "fractional", "hyper", "dirichlet")}
    monkeypatch.setattr(benchmark, "penalty_weights", lambda cfg, phase: weights.copy())

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.3, -.2]))

        def forward(self, inputs):
            return {"logits": self.weight.expand(*inputs.shape, 2), "signal": self.weight}

    class TinyStore:
        def __init__(self, *args, **kwargs):
            pass

        def batch(self, batch_size, generator):
            values = torch.randint(0, 2, (batch_size, 2), generator=generator)
            return values, 1 - values

    monkeypatch.setattr(benchmark, "SpectralTransformer", TinyModel)
    monkeypatch.setattr(benchmark, "TokenStore", TinyStore)
    baseline = tmp_path / "baseline.py"
    baseline.write_text(BASELINE_SOURCE, encoding="utf-8")
    original = benchmark.load_baseline(baseline, benchmark.sha256(baseline))

    def candidate_penalties(output, layers, exponent, *, weights):
        return {name: value if weights[name] else value.new_zeros(())
                for name, value in original.spectral_penalties(output, layers, exponent).items()}

    module = SimpleNamespace(spectral_penalties=candidate_penalties, total_loss=original.total_loss)
    monkeypatch.setattr(benchmark, "candidate", module)
    args = ["--baseline-source", str(baseline), "--device", "cpu", "--steps", "1", "--warmup", "1",
            "--accum", "1", "--microbatch", "2", "--context", "2", "--seeds", "0", "--phases", "0"]
    yield SimpleNamespace(args=args, candidate=module, weights=weights, baseline=baseline, model=TinyModel)
    torch.set_num_threads(threads)


def test_ce_masked_penalty_gradient_regression_writes_failure_metrics(tiny_main, tmp_path, monkeypatch):
    fixture = tiny_main
    fixture.weights.update({name: 1e-8 for name in fixture.weights})
    original = fixture.candidate.spectral_penalties

    def wrong_gradients(*args, **kwargs):
        return {name: .5 * value + .5 * value.detach() for name, value in original(*args, **kwargs).items()}

    fixture.candidate.spectral_penalties = wrong_gradients
    monkeypatch.setattr(benchmark, "measure", lambda *args: pytest.fail("Failed numerical gate reached timing"))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Penalty-only equivalence failed"):
        benchmark.main(fixture.args + ["--output-dir", str(output)])
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and "Penalty-only equivalence failed" in report["error"]
    assert report["cases"][0]["equivalence"]["passed"]  # CE alone hid the injected error.
    assert not report["cases"][0]["penalty_equivalence"]["passed"]
    assert report["cases"][0]["penalty_equivalence"]["gradient_relative_l2"] == pytest.approx(.5)
    assert benchmark.hashlib.sha256((output / "metrics.json").read_text().encode()).hexdigest() == (output / "metrics.json.sha256").read_text().strip()


def test_failed_total_equivalence_writes_incomplete_metrics(tiny_main, tmp_path, monkeypatch):
    original = tiny_main.candidate.total_loss
    tiny_main.candidate.total_loss = lambda *args: original(*args) + 10.
    monkeypatch.setattr(benchmark, "measure", lambda *args: pytest.fail("Failed total gate reached timing"))
    output = tmp_path / "total_failure"
    with pytest.raises(RuntimeError, match="Loss/gradient equivalence failed"):
        benchmark.main(tiny_main.args + ["--output-dir", str(output)])
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and not report["cases"][0]["equivalence"]["passed"]
    assert "Loss/gradient equivalence failed" in report["error"]


def test_inactive_terms_may_be_skipped_and_order_uses_seed_and_phase(tiny_main, tmp_path):
    fixture = tiny_main
    fixture.weights.update(manifold=0., fractional=0., hyper=0.)
    output = tmp_path / "orders"
    assert benchmark.main(fixture.args + ["--seeds", "0", "1", "--phases", "0", "135000", "--output-dir", str(output)]) == 0
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] and report["schema_version"] == 2
    assert [case["timing_order"] for case in report["cases"]] == [
        ["baseline", "candidate"], ["candidate", "baseline"],
        ["candidate", "baseline"], ["baseline", "candidate"],
    ]
    for case in report["cases"]:
        assert case["penalty_equivalence"]["active_terms"] == ["dirichlet"]
        assert case["equivalence"]["initial_state_scope"] == "random_initialization_only"
        assert not case["equivalence"]["checkpoint_supplied"]
        assert "penalty_manifold" not in case["equivalence"]["baseline"]
    for summary in report["summary"].values():
        assert summary["case_count"] == 2
        assert {row["case_count"] for row in summary["by_timing_order"].values()} == {1}
        assert len(summary["by_timing_order"]) == 2


def test_checkpoint_scope_is_explicit(tiny_main, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": tiny_main.model().state_dict(), "step": 123}, checkpoint)
    output = tmp_path / "checkpoint_run"
    assert benchmark.main(tiny_main.args + ["--checkpoint", str(checkpoint), "--output-dir", str(output)]) == 0
    report = json.loads((output / "metrics.json").read_text())
    assert report["cases"][0]["equivalence"]["checkpoint_supplied"]
    assert report["cases"][0]["equivalence"]["initial_state_scope"] == "supplied_checkpoint"
    assert report["checkpoint"]["step"] == 123


@pytest.mark.parametrize("failure", ["source_during_import", "source_since_import", "cached_modules"])
def test_candidate_import_identity_failure_is_retained(tmp_path, monkeypatch, failure):
    fixture_sources(monkeypatch)
    if failure == "source_during_import":
        monkeypatch.setattr(benchmark, "POST_IMPORT_SOURCE_HASHES", {"changed": "during import"})
    elif failure == "source_since_import":
        monkeypatch.setattr(benchmark, "sources", lambda: {"changed": "since import"})
    else:
        monkeypatch.setattr(benchmark, "PREEXISTING_LAPLACIAN_MODULES", ["laplacian.losses"])
    output = tmp_path / "source_failure"
    monkeypatch.setattr(benchmark, "load_baseline", lambda *args: pytest.fail("Import gate did not stop execution"))
    with pytest.raises(RuntimeError, match="fresh"):
        benchmark.main(["--baseline-source", str(tmp_path / "unused.py"), "--output-dir", str(output), "--device", "cpu"])
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False and report["cases"] == [] and "fresh" in report["error"]


def test_baseline_change_during_import_is_rejected_before_data_access(tmp_path, monkeypatch):
    fixture_sources(monkeypatch)
    baseline = tmp_path / "mutating.py"
    baseline.write_text('from pathlib import Path\nPath(__file__).write_text("mutated")\n', encoding="utf-8")
    digest = benchmark.sha256(baseline)
    output = tmp_path / "baseline_failure"
    with pytest.raises(RuntimeError, match="Baseline source changed during import"):
        benchmark.main(["--baseline-source", str(baseline), "--output-dir", str(output), "--device", "cpu"])
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and report["baseline_sha256"] == digest
    assert "Baseline source changed" in report["error"]


def test_baseline_changed_before_import_is_never_executed(tmp_path):
    baseline = tmp_path / "baseline.py"
    baseline.write_text("raise AssertionError('must not execute')", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed before execution"):
        benchmark.load_baseline(baseline, "incorrect-digest")


def test_final_source_drift_revokes_success(tiny_main, tmp_path, monkeypatch):
    original = benchmark.measure

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with tiny_main.baseline.open("a", encoding="utf-8") as handle:
            handle.write("\n# source changed\n")
        return result

    monkeypatch.setattr(benchmark, "measure", mutate)
    output = tmp_path / "final_drift"
    with pytest.raises(RuntimeError, match="Source files changed during benchmark"):
        benchmark.main(tiny_main.args + ["--output-dir", str(output)])
    report = json.loads((output / "metrics.json").read_text())
    assert not report["complete"] and "Source files changed" in report["error"]


def test_nonfinite_timing_loss_rejected_before_optimizer_step(tiny_main, monkeypatch):
    model = tiny_main.model()
    before = model.weight.detach().clone()
    monkeypatch.setattr(benchmark, "forward_loss", lambda *args: (model.weight.sum() * float("inf"), torch.tensor(1.), {}))
    args = SimpleNamespace(accum=1, warmup=1, steps=1, microbatch=2, context=2, memory_limit_mib=1000)
    config = benchmark.get_config("fixture")
    with pytest.raises(RuntimeError, match="Non-finite loss before benchmark optimizer update"):
        benchmark.measure(model, [(None, None)], tiny_main.candidate, config, 0, True, torch.device("cpu"), args)
    assert torch.equal(model.weight, before)
