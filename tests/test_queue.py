"""Correctness receipts must belong to the source actually checked."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def queue(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_queue.py"
    spec = importlib.util.spec_from_file_location("queue_freshness_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(module, "DONE", tmp_path / "runs" / "QUEUE_DONE")
    monkeypatch.setattr(module, "LEDGER", tmp_path / "runs" / "harness-ledger.jsonl")
    for relative in ("scripts/run_queue.py", "src/laplacian/losses.py", "tests/test_loss.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n", encoding="utf-8")
    return module


def gate_item():
    return {"id": "pytest", "kind": "gate", "gpu": False,
            "cmd": ["python", "-m", "pytest"], "metrics": "runs/pytest/metrics.json"}


def record_current(queue, item):
    queue.write_metrics(queue.ROOT / item["metrics"], {
        "complete": True, "verification_source": queue.gate_source_snapshot(item),
    })
    return queue.metrics_state(queue.ROOT / item["metrics"])


def test_legacy_gate_requires_a_fresh_check(queue):
    item = gate_item()
    queue.write_metrics(queue.ROOT / item["metrics"], {"complete": True})
    state = queue.metrics_state(queue.ROOT / item["metrics"])
    assert state["ok"]
    assert queue.gate_is_stale(item, state)


def test_content_change_invalidates_with_preserved_timestamp(queue):
    item = gate_item()
    state = record_current(queue, item)
    path = queue.ROOT / "src/laplacian/losses.py"
    stamp = path.stat()
    path.write_text("modified\n", encoding="utf-8")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert queue.gate_is_stale(item, state)


@pytest.mark.parametrize("action", ["add", "delete"])
def test_test_inventory_changes_invalidate(queue, action):
    item = gate_item()
    state = record_current(queue, item)
    if action == "add":
        (queue.ROOT / "tests/test_added.py").write_text("new\n", encoding="utf-8")
    else:
        (queue.ROOT / "tests/test_loss.py").unlink()
    assert queue.gate_is_stale(item, state)


def test_reference_outputs_do_not_invalidate_reference_inputs(queue):
    item = {**gate_item(), "id": "verify_reference"}
    before = queue.gate_source_snapshot(item)
    receipt = queue.ROOT / "reference/receipts/verify-hypergraph.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"passed": true}', encoding="utf-8")
    assert queue.gate_source_snapshot(item) == before


def test_benchmark_script_change_invalidates_pytest_gate(queue):
    item = gate_item()
    state = record_current(queue, item)
    (queue.ROOT / "scripts/benchmark_optimization.py").write_text("new benchmark\n", encoding="utf-8")
    assert queue.gate_is_stale(item, state)


def test_successful_execution_attests_verified_sources(queue, monkeypatch):
    item = gate_item()
    monkeypatch.setattr(queue, "stream_command", lambda command: 0)
    assert queue.execute(item) == 0
    state = queue.metrics_state(queue.ROOT / item["metrics"])
    assert state["ok"]
    assert not queue.gate_is_stale(item, state)
    assert state["payload"]["jovin_receipt"] is False


def test_sources_modified_during_gate_cannot_receive_attestation(queue, monkeypatch):
    item = gate_item()

    def edit_during_check(command):
        (queue.ROOT / "src/laplacian/losses.py").write_text("edited during run\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(queue, "stream_command", edit_during_check)
    assert queue.execute(item) == 1
    state = queue.metrics_state(queue.ROOT / item["metrics"])
    assert queue.gate_is_stale(item, state)


def test_done_marker_revalidates_gates_without_retraining(queue, monkeypatch):
    gate = gate_item()
    train = {"id": "arm0", "kind": "run", "gpu": True,
             "cmd": ["python", "train"], "metrics": "runs/arm0/metrics.json"}
    queue.write_metrics(queue.ROOT / train["metrics"], {"complete": True, "historical": True})
    training_before = (queue.ROOT / train["metrics"]).read_bytes()
    queue.DONE.write_text('{"complete": true}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(queue, "apply_env", lambda: None)
    monkeypatch.setattr(queue, "build_items", lambda: [gate, train])
    monkeypatch.setattr(queue, "foreign_pids", lambda: [])
    monkeypatch.setattr(queue, "preflight_wsl", lambda: ("chain", 12))
    monkeypatch.setattr(queue, "backfill", lambda *args: None)
    monkeypatch.setattr(queue, "stream_command", lambda command: calls.append(command) or 0)
    monkeypatch.setattr(queue, "gpu_is_busy", lambda: pytest.fail("historical training must not restart"))

    assert queue.main([]) == 0
    assert calls == [gate["cmd"]]
    assert json.loads(queue.DONE.read_text(encoding="utf-8"))["complete"]
    assert queue.main([]) == 0
    assert calls == [gate["cmd"]]
    (queue.ROOT / "tests/test_loss.py").write_text("new checks\n", encoding="utf-8")
    assert queue.main([]) == 0
    assert calls == [gate["cmd"], gate["cmd"]]
    assert (queue.ROOT / train["metrics"]).read_bytes() == training_before


def test_failed_gate_does_not_attest_old_success(queue, monkeypatch):
    item = gate_item()
    record_current(queue, item)
    (queue.ROOT / "src/laplacian/losses.py").write_text("new\n", encoding="utf-8")
    monkeypatch.setattr(queue, "stream_command", lambda command: 1)
    assert queue.execute(item) == 1
    assert queue.gate_is_stale(item, queue.metrics_state(queue.ROOT / item["metrics"]))


@pytest.mark.parametrize("done_exists", [False, True])
def test_gates_only_never_runs_training_or_changes_done_marker(queue, monkeypatch, done_exists):
    gate = gate_item()
    train = {"id": "arm1", "kind": "run", "gpu": True,
             "cmd": ["python", "train"], "metrics": "runs/arm1/metrics.json"}
    if done_exists:
        queue.DONE.parent.mkdir(parents=True)
        queue.DONE.write_text("historical queue completion\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(queue, "apply_env", lambda: None)
    monkeypatch.setattr(queue, "build_items", lambda: [gate, train])
    monkeypatch.setattr(queue, "foreign_pids", lambda: [])
    monkeypatch.setattr(queue, "preflight_wsl", lambda: ("chain", 12))
    monkeypatch.setattr(queue, "backfill", lambda *args: None)
    monkeypatch.setattr(queue, "stream_command", lambda command: calls.append(command) or 0)
    monkeypatch.setattr(queue, "gpu_is_busy", lambda: pytest.fail("must not run GPU training"))
    assert queue.main(["--gates-only"]) == 0
    assert calls == [gate["cmd"]]
    assert queue.DONE.exists() is done_exists
    if done_exists:
        assert queue.DONE.read_text(encoding="utf-8") == "historical queue completion\n"


def test_reference_ledger_uses_chain_after_new_receipts(queue, monkeypatch):
    reference = {**gate_item(), "id": "verify_reference", "metrics": "runs/verify_reference/metrics.json"}
    monkeypatch.setattr(queue, "apply_env", lambda: None)
    monkeypatch.setattr(queue, "build_items", lambda: [reference])
    monkeypatch.setattr(queue, "foreign_pids", lambda: [])
    monkeypatch.setattr(queue, "preflight_wsl", lambda: ("old", 35))
    monkeypatch.setattr(queue, "verify_receipt_chain", lambda: ("fresh", 47))
    monkeypatch.setattr(queue, "backfill", lambda *args: None)
    monkeypatch.setattr(queue, "stream_command", lambda command: 0)
    assert queue.main(["--gates-only"]) == 0
    row = json.loads(queue.LEDGER.read_text(encoding="utf-8").splitlines()[-1])
    assert row["jovin_head"] == "fresh"
    assert row["jovin_count"] == 47
