"""CPU diagnostic receipts must remain immutable and fail closed."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fractional_cutoff_cpu_test", ROOT / "scripts/check_fractional_cutoff_cpu.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_complete_cpu_sweep_and_immutable_receipt(tmp_path):
    output = tmp_path / "attempt"
    report = diagnostic.run(output)
    assert report["complete"] is True
    assert report["device"] == "cpu"
    assert report["summary"]["case_count"] == 144
    assert all(row["finite"] and row["oracle_passed"] for row in report["cases"])
    assert report["source_sha256"] == report["source_sha256_after"]
    for relative, digest in report["source_sha256"].items():
        assert hashlib.sha256((output / "source_snapshot" / relative).read_bytes()).hexdigest() == digest
    metrics = output / "metrics.json"
    before = metrics.read_bytes()
    assert hashlib.sha256(metrics.read_text().encode()).hexdigest() == (output / "metrics.json.sha256").read_text().strip()
    with pytest.raises(FileExistsError):
        diagnostic.run(output)
    assert metrics.read_bytes() == before


def test_operator_failure_writes_incomplete_receipt_and_restores_cpu_threads(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()

    def fail(*args, **kwargs):
        raise RuntimeError("injected operator failure")

    monkeypatch.setattr(diagnostic, "fractional_matrix", fail)
    output = tmp_path / "failed_attempt"
    with pytest.raises(RuntimeError, match="injected operator failure"):
        diagnostic.run(output)
    report = json.loads((output / "metrics.json").read_text())
    assert report["complete"] is False
    assert "injected operator failure" in report["error"]
    assert report["cases"] == []
    assert torch.get_num_threads() == previous_threads
    assert (output / "metrics.json.sha256").is_file()
