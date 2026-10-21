"""A benchmark must reject numerical regressions, even when timing improves."""

import importlib.util
from pathlib import Path

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
