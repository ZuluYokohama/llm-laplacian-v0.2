"""Real CPU subprocesses exercise accounting, containment, and crash recovery."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "study_budget.py"
SPEC = importlib.util.spec_from_file_location("study_budget", SCRIPT)
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    decisions = tmp_path / "decisions.json"
    ledger = tmp_path / "study" / "budget.jsonl"
    decisions.write_text(json.dumps({"recommended_core_gpu_hours": {"approved": True, "hard_cap": 0.01, "ledger": str(ledger)}}))
    return decisions, ledger


def cli(paths, *args):
    decisions, ledger = paths
    if args and args[0] == "run":
        args = ("run", "--cuda-device", "", *args[1:])
    return [sys.executable, str(SCRIPT), "--decisions", str(decisions), "--ledger", str(ledger), *args]


def run(paths, run_id, code, reserve=2.0):
    return subprocess.run(cli(paths, "run", "--run-id", run_id, "--reserve-seconds", str(reserve),
                              "--", sys.executable, "-c", code), capture_output=True, text=True, timeout=10)


def wait_until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition did not become true")


def test_success_is_durable_measured_and_cpu_only(paths):
    result = run(paths, "ok", "import os; print(repr(os.environ['CUDA_VISIBLE_DEVICES']))")
    assert result.returncode == 0, result.stderr + result.stdout
    records = budget.read_ledger(paths[1])
    assert [row["event"] for row in records] == ["reserved", "finished"]
    assert 0 < records[-1]["charge_seconds"] < 2
    directory = Path(records[0]["directory"])
    assert (directory / "stdout.log").read_text().strip() == "''"
    assert budget.accounting(records)["ok"]["settled"]


def test_failure_keeps_entire_reservation(paths):
    result = run(paths, "failed", "raise SystemExit(7)")
    assert result.returncode == 7, result.stderr + result.stdout
    last = budget.read_ledger(paths[1])[-1]
    assert last["status"] == "failed"
    assert last["charge_seconds"] == 2


def test_timeout_kills_grandchild_and_keeps_reservation(paths, tmp_path):
    marker = tmp_path / "grandchild.pid"
    child = "import time,pathlib,os; pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(30)"
    proc = subprocess.Popen(cli(paths, "run", "--run-id", "timeout", "--reserve-seconds", "2", "--",
                                sys.executable, "-c", code), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_until(lambda: marker.exists() and marker.stat().st_size > 0)
        grandchild_pid = int(marker.read_text())
        assert budget.process_alive(grandchild_pid), "Handshake must observe the running grandchild"
        assert proc.wait(timeout=5) != 0
        wait_until(lambda: not budget.process_alive(grandchild_pid))
        record = budget.read_ledger(paths[1])[-1]
        assert record["status"] in ("timeout", "uncertain")
        assert record["charge_seconds"] >= 2
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_cap_and_run_id_cannot_be_reused(paths):
    paths[0].write_text(json.dumps({"recommended_core_gpu_hours": {"approved": True, "hard_cap": 2.5/3600, "ledger": str(paths[1])}}))
    assert run(paths, "failure", "raise SystemExit(1)", reserve=2).returncode != 0
    result = run(paths, "overcap", "print('must not launch')", reserve=1)
    assert result.returncode == 2
    assert "exceeds approved" in result.stderr
    assert len(budget.read_ledger(paths[1])) == 2
    result = run(paths, "failure", "print('must not launch')", reserve=0.5)
    assert result.returncode == 2
    assert "immutable" in result.stderr


@pytest.mark.parametrize("grant", [{"approved": False, "hard_cap": 181}, {"approved": True, "hard_cap": 182}, {"approved": True, "hard_cap_proposed": 181}])
def test_unapproved_or_unsupported_cap_blocks_before_launch(paths, grant):
    paths[0].write_text(json.dumps({"recommended_core_gpu_hours": grant}))
    result = run(paths, "blocked", "print('must not launch')")
    assert result.returncode == 2
    assert not paths[1].exists()


def test_chain_corruption_and_partial_write_fail_closed(paths):
    assert run(paths, "ok", "pass").returncode == 0
    original = paths[1].read_bytes()
    paths[1].write_bytes(original.replace(b'"charge_seconds":', b'"charge_seconds":9', 1))
    with pytest.raises(budget.BudgetError):
        budget.read_ledger(paths[1])
    paths[1].write_bytes(original[:-1])
    with pytest.raises(budget.BudgetError, match="Truncated"):
        budget.read_ledger(paths[1])


def test_missing_ledger_cannot_reset_existing_study(paths):
    assert run(paths, "ok", "pass").returncode == 0
    paths[1].unlink()
    result = run(paths, "reset", "pass")
    assert result.returncode == 2
    assert "existing run artifacts" in result.stderr


def test_valid_ledger_prefix_cannot_hide_removed_reservation(paths):
    assert run(paths, "ok", "pass").returncode == 0
    (paths[1].parent / "runs" / "orphaned-reservation").mkdir()
    with pytest.raises(budget.BudgetError, match="inventory disagree"):
        budget.read_ledger(paths[1])


def test_live_lock_rejects_concurrent_admission(paths):
    proc = subprocess.Popen(cli(paths, "run", "--run-id", "first", "--reserve-seconds", "4", "--",
                                sys.executable, "-c", "import time; time.sleep(1)"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_until(lambda: paths[1].exists() and paths[1].stat().st_size > 0)
        result = run(paths, "second", "pass")
        assert result.returncode == 2
        assert "holds the study lock" in result.stderr
        assert proc.wait(timeout=6) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_crashed_wrapper_retains_charge_until_explicit_recovery(paths, tmp_path):
    marker = tmp_path / "grandchild-crash.pid"
    child = "import time,pathlib,os; pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(30)"
    proc = subprocess.Popen(cli(paths, "run", "--run-id", "crashed", "--reserve-seconds", "2", "--",
                                sys.executable, "-c", code), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    directory = paths[1].parent / "runs" / "crashed"
    try:
        wait_until(lambda: marker.exists() and marker.stat().st_size > 0)
        grandchild_pid = int(marker.read_text())
        assert budget.process_alive(grandchild_pid), "Crash only after grandchild startup handshake"
        proc.kill()
        proc.wait(timeout=2)
        records = budget.read_ledger(paths[1])
        assert len(records) == 1
        assert budget.accounting(records)["crashed"]["charge_seconds"] == 2
        assert run(paths, "premature", "pass").returncode == 2
        wait_until(lambda: not budget.process_alive(grandchild_pid))
        time.sleep(max(0, records[0]["deadline_unix"] - time.time()) + 0.05)
        result = subprocess.run(cli(paths, "recover", "--run-id", "crashed"), capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr + result.stdout
        last = budget.read_ledger(paths[1])[-1]
        assert last["event"] == "recovered" and last["charge_seconds"] == 2
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_command_and_environment_provenance_omit_secrets(monkeypatch):
    monkeypatch.setenv("PRIVATE_API_TOKEN", "secret-env-value")
    value = budget.provenance(["python", "-c", "print('secret-code-value')", "--api-key", "secret-arg-value"], "")
    encoded = json.dumps(value)
    assert "secret-env-value" not in encoded
    assert "secret-code-value" not in encoded
    assert "secret-arg-value" not in encoded


def test_logged_output_redacts_known_credentials(paths, monkeypatch):
    monkeypatch.setenv("PRIVATE_API_TOKEN", "secret-env-value")
    result = run(paths, "redaction", "import os; print(os.environ['PRIVATE_API_TOKEN'])")
    assert result.returncode == 0, result.stderr + result.stdout
    assert (paths[1].parent / "runs" / "redaction" / "stdout.log").read_text().strip() == "<redacted>"


def test_recovery_cannot_credit_unused_uncertain_time():
    with pytest.raises(budget.BudgetError, match="refund"):
        budget.accounting([{"event": "reserved", "run_id": "x", "reserve_seconds": 10},
                           {"event": "recovered", "run_id": "x", "status": "abandoned", "charge_seconds": 1}])


def test_approval_cannot_be_reused_with_another_ledger(paths, tmp_path):
    other = tmp_path / "duplicate-allowance" / "budget.jsonl"
    result = run((paths[0], other), "bypass", "print('must not launch')")
    assert result.returncode == 2
    assert "differs from the approved ledger" in result.stderr
    assert not other.exists()
