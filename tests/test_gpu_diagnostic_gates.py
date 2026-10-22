"""CPU-only admission, comparison and failed-receipt tests; no CUDA execution."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_cpu_test", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


numerics = load("check_gpu_numerics")
sensitivity = load("check_fractional_sensitivity")
resume = load("check_gpu_resume")


@pytest.mark.parametrize("module", [numerics, sensitivity, resume])
@pytest.mark.parametrize("invalid", ["missing_id", "blank_id", "missing_ledger", "wrong_ledger"])
def test_budget_guard_precedes_every_cuda_query(module, monkeypatch, invalid):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "cpu-test-not-a-launch")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(module.ROOT / "runs/publication_study/budget.jsonl"))
    if invalid == "missing_id":
        monkeypatch.delenv("STUDY_BUDGET_RUN_ID")
    elif invalid == "blank_id":
        monkeypatch.setenv("STUDY_BUDGET_RUN_ID", " ")
    elif invalid == "missing_ledger":
        monkeypatch.delenv("STUDY_BUDGET_LEDGER")
    else:
        monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(module.ROOT / "wrong-ledger.jsonl"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried before budget rejection"))
    with pytest.raises(RuntimeError, match="canonical study budget wrapper"):
        module.require_gpu()


@pytest.mark.parametrize("module", [numerics, sensitivity, resume])
def test_approved_context_still_requires_hardware_and_bf16(module, monkeypatch):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "cpu-test-not-a-launch")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(module.ROOT / "runs/publication_study/budget.jsonl"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="approved RTX 5070"):
        module.require_gpu()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fake RTX 5070")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="BF16 autocast"):
        module.require_gpu()


@pytest.mark.parametrize("module", [numerics, sensitivity])
def test_prerequisite_rejection_persists_immutable_failure(module, tmp_path, monkeypatch):
    monkeypatch.delenv("STUDY_BUDGET_RUN_ID", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA must not be touched"))
    output = tmp_path / module.__name__
    args = ["--output", str(output)]
    if module is sensitivity:
        args += ["--checkpoint", str(tmp_path / "missing-checkpoint.pt")]
    with pytest.raises(RuntimeError, match="canonical study budget wrapper"):
        module.main(args)
    path = output / "metrics.json"
    report = json.loads(path.read_text())
    assert report["complete"] is False and report["status"] == "failed" and report["host_gpu"] is False
    assert report["jovin_receipt"] is False and Path(str(path) + ".sha256").exists()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(args)
    assert path.read_bytes() == before


def test_numerics_source_hash_failure_is_receipted(tmp_path, monkeypatch):
    def fail():
        raise OSError("CPU injected source read failure")
    monkeypatch.setattr(numerics, "source_hashes", fail)
    monkeypatch.setattr(numerics, "require_gpu", lambda: pytest.fail("no GPU prerequisite reached"))
    with pytest.raises(OSError, match="source read failure"):
        numerics.main(["--output", str(tmp_path / "source")])
    report = json.loads((tmp_path / "source/metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"


@pytest.mark.parametrize("missing", ["checkpoint", "metadata"])
def test_sensitivity_input_failure_is_receipted(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(sensitivity, "require_gpu", lambda: torch.device("cpu"))
    monkeypatch.setattr(sensitivity, "DATA", tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    if missing != "checkpoint":
        checkpoint.write_bytes(b"CPU fixture")
    (tmp_path / "tinystories_val.u16").write_bytes(b"\0\0")
    if missing != "metadata":
        (tmp_path / "meta.json").write_text('{"vocab": 7}')
    with pytest.raises(FileNotFoundError):
        sensitivity.main(["--output", str(tmp_path / "attempt"), "--checkpoint", str(checkpoint)])
    report = json.loads((tmp_path / "attempt/metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert "FileNotFoundError" in report["error"]


@pytest.mark.parametrize("exponent", [.25, .5, .75, 1.0])
def test_cpu_fp64_reference_gradient_has_independent_finite_difference_oracle(exponent):
    data = torch.tensor([[0., 0.], [1., .2], [-.3, .7], [.5, -.4]], dtype=torch.float64)
    direction = torch.tensor([[.3, -.2], [.1, .4], [-.2, .1], [.2, -.1]], dtype=torch.float64)
    loss, gradient = numerics.cpu_reference(data, exponent)
    epsilon = 1e-5
    left, _ = numerics.cpu_reference(data - epsilon * direction, exponent)
    right, _ = numerics.cpu_reference(data + epsilon * direction, exponent)
    assert loss.dtype == gradient.dtype == torch.float64 and loss.device.type == "cpu"
    assert float((gradient * direction).sum()) == pytest.approx(float((right - left) / (2 * epsilon)), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("autocast", [False, True])
def test_reference_comparison_rejects_finite_loss_or_gradient_error(autocast):
    loss, gradient = torch.tensor(1.), torch.ones(2, 2)
    assert numerics.compare_reference(loss, gradient, loss, gradient, autocast)["passed"]
    wrong_loss = numerics.compare_reference(loss + 1, gradient, loss, gradient, autocast)
    wrong_gradient = numerics.compare_reference(loss, gradient + 1, loss, gradient, autocast)
    assert wrong_loss["finite"] and not wrong_loss["passed"] and wrong_loss["loss_abs_error"] == 1
    assert wrong_gradient["finite"] and not wrong_gradient["passed"] and wrong_gradient["gradient_max_abs_error"] == 1
    assert not numerics.compare_reference(loss, gradient * float("nan"), loss, gradient, autocast)["passed"]
    zero = numerics.compare_reference(torch.tensor(0.), torch.zeros(2, 2), torch.tensor(0.), torch.zeros(2, 2), autocast)
    assert zero["passed"] and zero["gradient_relative_l2_error"] is None


def test_cpu_fake_numeric_mismatch_has_failed_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(numerics, "require_gpu", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fake, no hardware execution")
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(numerics, "spectral_penalties", lambda output, *args, **kwargs:
                        {"fractional": output["pooled"][0].square().sum() + 1})
    with pytest.raises(RuntimeError, match="CPU FP64 reference mismatch"):
        numerics.main(["--output", str(tmp_path / "mismatch")])
    report = json.loads((tmp_path / "mismatch/metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert len(report["cases"]) == 1
    assert report["cases"][0]["finite"] is True
    assert report["cases"][0]["cpu_fp64_comparison"]["passed"] is False


def test_default_sensitivity_calls_production_fractional_only_under_no_grad(monkeypatch):
    value = torch.tensor(3., requires_grad=True)
    def production(output, layers, exponent, *, weights):
        assert not torch.is_grad_enabled()
        assert weights == {"fractional": 1., "manifold": 0., "hyper": 0., "dirichlet": 0.}
        return {"fractional": torch.tensor(3.)}
    monkeypatch.setattr(sensitivity, "spectral_penalties", production)
    assert sensitivity.production_default_error({"pooled": [value]}, value, .5) == 0


def test_cpu_fake_sensitivity_mismatch_is_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(sensitivity, "require_gpu", lambda: torch.device("cpu"))
    monkeypatch.setattr(sensitivity, "DATA", tmp_path)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model": {}})
    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.states = torch.nn.Parameter(torch.zeros(4, 4))
        def load_state_dict(self, state):
            pass
        def forward(self, inputs):
            return {"pooled": [self.states]}
    class Store:
        def __init__(self, *args, **kwargs):
            pass
        def batch(self, *args):
            return torch.zeros(8, 2, dtype=torch.long), None
    monkeypatch.setattr(sensitivity, "SpectralTransformer", Model)
    monkeypatch.setattr(sensitivity, "TokenStore", Store)
    monkeypatch.setattr(sensitivity, "spectral_penalties", lambda *args, **kwargs: {"fractional": torch.tensor(1.)})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"CPU fake checkpoint")
    (tmp_path / "tinystories_val.u16").write_bytes(b"\0\0")
    (tmp_path / "meta.json").write_text('{"vocab": 7}')
    with pytest.raises(RuntimeError, match="differs from production"):
        sensitivity.main(["--output", str(tmp_path / "mismatch"), "--checkpoint", str(checkpoint)])
    report = json.loads((tmp_path / "mismatch/metrics.json").read_text())
    assert report["complete"] is False and report["status"] == "failed"
    assert report["failed_default_check"]["production_abs_error"] == 1.0
