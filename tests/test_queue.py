"""Correctness receipts must belong to the source actually checked."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("gate", ["verify_reference", "pytest"])
def test_reference_outputs_do_not_invalidate_reference_inputs(queue, gate):
    item = {**gate_item(), "id": gate}
    before = queue.gate_source_snapshot(item)
    receipt = queue.ROOT / "reference/receipts/verify-hypergraph.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"passed": true}', encoding="utf-8")
    assert queue.gate_source_snapshot(item) == before


@pytest.mark.parametrize("relative", [
    "scripts/verify_reference.ps1", "paper/fetch_references.py", "paper/planning/FRACTIONAL_POLICY.md",
    "runs/optimization_20261021/baseline_losses.py", "paper/planning/timing_schedule.json",
])
def test_static_test_inputs_invalidate_pytest_gate(queue, relative):
    item = gate_item()
    path = queue.ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original\n", encoding="utf-8")
    state = record_current(queue, item)
    stamp = path.stat()
    path.write_text("changed\n", encoding="utf-8")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert queue.gate_is_stale(item, state)


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
    monkeypatch.setattr(queue, "stream_command", lambda command, **kwargs: 0)
    assert queue.main(["--gates-only"]) == 0
    row = json.loads(queue.LEDGER.read_text(encoding="utf-8").splitlines()[-1])
    assert row["jovin_head"] == "fresh"
    assert row["jovin_count"] == 47


def test_failed_reference_records_failure_without_refreshing_chain(queue, monkeypatch):
    reference = {**gate_item(), "id": "verify_reference", "metrics": "runs/verify_reference/metrics.json"}
    monkeypatch.setattr(queue, "apply_env", lambda: None)
    monkeypatch.setattr(queue, "build_items", lambda: [reference])
    monkeypatch.setattr(queue, "foreign_pids", lambda: [])
    monkeypatch.setattr(queue, "preflight_wsl", lambda: ("old", 35))
    monkeypatch.setattr(queue, "verify_receipt_chain", lambda: pytest.fail("failed reference must not refresh chain"))
    monkeypatch.setattr(queue, "backfill", lambda *args: None)
    monkeypatch.setattr(queue, "stream_command", lambda command, **kwargs: 7)
    assert queue.main(["--gates-only"]) == 7
    row = json.loads(queue.LEDGER.read_text(encoding="utf-8").splitlines()[-1])
    assert row["id"] == "verify_reference"
    assert row["exit"] == 7 and row["skipped"] is False
    assert row["jovin_head"] == "old" and row["jovin_count"] == 35
    assert not queue.metrics_state(queue.ROOT / reference["metrics"])["ok"]


@pytest.mark.parametrize("exit_code,output,message", [
    (1, '{"valid": true}', "receipt chain is not valid"),
    (0, '{"valid": false}', "receipt chain is not valid"),
    (0, '{"valid": 1}', "receipt chain is not valid"),
    (0, '{"valid": "true"}', "receipt chain is not valid"),
    (0, 'no JSON returned', "no JSON object"),
])
def test_real_receipt_chain_rejects_invalid_response(queue, monkeypatch, exit_code, output, message):
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=exit_code, stdout=output, stderr=""))
    with pytest.raises(SystemExit, match=message):
        queue.verify_receipt_chain()


def test_real_receipt_chain_accepts_boolean_success_and_has_outer_timeout(queue, monkeypatch):
    def run(command, **kwargs):
        assert command[-2:] == ["receipts", "verify"]
        assert kwargs["timeout"] == queue.HARNESS_CHECK_TIMEOUT_S
        return SimpleNamespace(returncode=0, stdout='{"valid": true, "head_sha256": "head", "count": 47}', stderr="")

    monkeypatch.setattr(queue.subprocess, "run", run)
    assert queue.verify_receipt_chain() == ("head", 47)


@pytest.mark.parametrize("check,message", [
    ("preflight_wsl", "jovin doctor timed out after 120 seconds"),
    ("verify_receipt_chain", "jovin receipts verify timed out after 120 seconds"),
])
def test_harness_checks_fail_clearly_on_timeout(queue, monkeypatch, check, message):
    monkeypatch.setattr(queue, "stream_command", lambda command, **kwargs: 0)

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == queue.HARNESS_CHECK_TIMEOUT_S
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(queue.subprocess, "run", timeout)
    with pytest.raises(SystemExit, match=message):
        getattr(queue, check)()


@pytest.mark.parametrize("ready", [False, 1, "true", None])
def test_real_doctor_requires_boolean_ready(queue, monkeypatch, ready):
    monkeypatch.setattr(queue, "stream_command", lambda command, **kwargs: 0)
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"ready": ready}), stderr=""))
    monkeypatch.setattr(queue, "verify_receipt_chain", lambda: pytest.fail("unready doctor must stop preflight"))
    with pytest.raises(SystemExit, match="doctor is not ready"):
        queue.preflight_wsl()


def test_docker_preflight_has_outer_limit_and_timeout_stops_preflight(queue, monkeypatch):
    def timeout(command, *, timeout_s):
        assert command[-2:] == ["docker", "version"]
        assert timeout_s == 120
        raise subprocess.TimeoutExpired(command, timeout_s)
    monkeypatch.setattr(queue, "stream_command", timeout)
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: pytest.fail("Doctor ran after Docker timeout"))
    with pytest.raises(SystemExit, match="WSL Docker preflight timed out after 120 seconds; gate not completed"):
        queue.preflight_wsl()


def test_reference_timeout_cannot_attest_gate_or_continue_queue(queue, monkeypatch):
    reference = {**gate_item(), "id": "verify_reference", "metrics": "runs/verify_reference/metrics.json"}
    def timeout(command, *, timeout_s):
        assert timeout_s == 900
        raise subprocess.TimeoutExpired(command, timeout_s)
    monkeypatch.setattr(queue, "stream_command", timeout)
    monkeypatch.setattr(queue, "apply_env", lambda: None)
    monkeypatch.setattr(queue, "build_items", lambda: [reference])
    monkeypatch.setattr(queue, "foreign_pids", lambda: [])
    monkeypatch.setattr(queue, "preflight_wsl", lambda: ("old", 35))
    monkeypatch.setattr(queue, "verify_receipt_chain", lambda: pytest.fail("Timed-out reference cannot refresh chain"))
    monkeypatch.setattr(queue, "backfill", lambda *args: None)
    assert queue.main(["--gates-only"]) == 124
    row = json.loads(queue.LEDGER.read_text(encoding="utf-8").splitlines()[-1])
    assert row["exit"] == 124 and row["sha256"] is None and row["skipped"] is False
    assert not queue.metrics_state(queue.ROOT / reference["metrics"])["ok"]
    assert not queue.DONE.exists()


def test_generic_training_command_receives_no_arbitrary_timeout(queue, monkeypatch):
    item = {"id": "arm0", "kind": "run", "gpu": True, "cmd": ["synthetic-training"], "metrics": "runs/arm0/metrics.json"}
    calls = []
    def stream(command, **kwargs):
        calls.append((command, kwargs))
        return 7
    monkeypatch.setattr(queue, "stream_command", stream)
    assert queue.execute(item) == 7
    assert calls == [(item["cmd"], {})]


def test_bounded_real_subprocess_streams_short_output_and_succeeds(queue, capsys):
    assert queue.stream_command([sys.executable, "-u", "-c", "import sys; sys.stdout.write('short without newline'); sys.stdout.flush()"],
                                timeout_s=5) == 0
    assert "short without newline" in capsys.readouterr().out


def test_bounded_real_subprocess_cannot_block_on_unterminated_line(queue, capsys):
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        queue.stream_command([sys.executable, "-u", "-c", "import sys,time; sys.stdout.write('ready without newline'); sys.stdout.flush(); time.sleep(30)"],
                             timeout_s=.5)
    assert time.monotonic() - started < 8
    assert "ready without newline" in capsys.readouterr().out


def test_containment_failure_prevents_child_launch_and_queue_entry(queue, monkeypatch):
    def unavailable_job():
        raise OSError("synthetic job setup failure")
    monkeypatch.setitem(sys.modules, "study_budget", SimpleNamespace(windows_job=unavailable_job))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Uncontained child was launched"))
    monkeypatch.setattr(queue, "main", lambda *args: pytest.fail("Helper entered ordinary queue"))
    with pytest.raises(OSError, match="synthetic job setup failure"):
        queue.contained_queue_command({"command": ["must-not-run"], "deadline_monotonic": time.monotonic() + 10})


def test_job_implementation_change_invalidates_reference_gate(queue):
    item = {**gate_item(), "id": "verify_reference"}
    state = record_current(queue, item)
    (queue.ROOT / "scripts/study_budget.py").write_text("new containment code\n", encoding="utf-8")
    assert queue.gate_is_stale(item, state)


def _process_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_uint32()
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    if sys.platform == "linux":
        try:
            # comm can contain spaces and parentheses; state follows its last ')'.
            identity, fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)
            stat_pid, _ = identity.split(" (", 1)
            state, parent_pid, *_ = fields.split()
            if int(stat_pid) == pid and state == "Z" and int(parent_pid) >= 0:
                return False  # Exited but not yet reaped; kill(pid, 0) still succeeds.
        except (OSError, UnicodeError, ValueError):
            pass  # Missing/unreadable/malformed proc data: retain the kill check.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture
def posix_process_probe(monkeypatch):
    calls = []
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "os", SimpleNamespace(
        name="posix", kill=lambda pid, sig: calls.append((pid, sig))))
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    return calls


@pytest.mark.parametrize("state,expected", [("Z", False), ("S", True)])
def test_process_alive_checks_linux_state_after_comm(monkeypatch, posix_process_probe, state, expected):
    def stat(path, **kwargs):
        assert path.as_posix() == "/proc/321/stat"
        return f"321 (worker ) with spaces) {state} 1 2 3\n"

    monkeypatch.setattr(Path, "read_text", stat)
    assert _process_alive(321) is expected
    assert posix_process_probe == ([(321, 0)] if expected else [])


@pytest.mark.parametrize("stat_result", [
    FileNotFoundError(), PermissionError(), "invalid stat", "321 (worker) Z",
])
@pytest.mark.parametrize("alive", [False, True])
def test_process_alive_falls_back_for_unavailable_linux_stat(
    monkeypatch, posix_process_probe, stat_result, alive,
):
    def stat(path, **kwargs):
        if isinstance(stat_result, Exception):
            raise stat_result
        return stat_result

    def kill(pid, sig):
        posix_process_probe.append((pid, sig))
        if not alive:
            raise ProcessLookupError

    monkeypatch.setattr(Path, "read_text", stat)
    monkeypatch.setattr(os, "kill", kill)
    assert _process_alive(321) is alive
    assert posix_process_probe == [(321, 0)]


def test_process_alive_nonlinux_uses_kill_without_proc(monkeypatch, posix_process_probe):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("Non-Linux proc read"))
    assert _process_alive(321)
    assert posix_process_probe == [(321, 0)]


def test_timeout_stops_live_child_and_handshaken_grandchild(queue, tmp_path):
    handshake = tmp_path / "actual-grandchild.json"
    grandchild = ("import json,os,sys,time; from pathlib import Path; "
                  "Path(sys.argv[1]).write_text(json.dumps({'parent':os.getppid(),'child':os.getpid()})); "
                  "print('grandchild running',flush=True); time.sleep(30)")
    parent = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable,'-u','-c',{grandchild!r},sys.argv[1]]); "
              "print('parent running',flush=True); time.sleep(30)")
    pids = []
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            queue.stream_command([sys.executable, "-u", "-c", parent, str(handshake)], timeout_s=2)
        assert handshake.is_file(), "Grandchild must actually launch before the timeout test is meaningful"
        pids = list(json.loads(handshake.read_text(encoding="utf-8")).values())
        assert len(pids) == 2 and len(set(pids)) == 2
        deadline = time.monotonic() + 2
        while any(_process_alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not any(_process_alive(pid) for pid in pids)
    finally:
        for pid in pids:
            if _process_alive(pid):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=5, check=False)
                else:
                    import signal
                    os.kill(pid, signal.SIGKILL)


def test_powershell_guards_require_literal_boolean_success(tmp_path):
    """Execute the real script's parsed guards, with no WSL/harness launch."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required for its Boolean coercion regression")
    source = Path(__file__).resolve().parents[1] / "scripts/verify_reference.ps1"
    runner = tmp_path / "check-guards.ps1"
    runner.write_text(r'''
param([string]$Source)
$ErrorActionPreference = "Stop"
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw "PowerShell parse failed" }
$ifs = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.IfStatementAst] }, $true)
$doctorGuard = @($ifs | Where-Object { $_.Clauses[0].Item1.Extent.Text.StartsWith('$doctorExit -ne 0') })
$chainGuard = @($ifs | Where-Object { $_.Clauses[0].Item1.Extent.Text.StartsWith('$chainExit -ne 0') })
$operatorGuard = @($ifs | Where-Object { $_.Clauses[0].Item1.Extent.Text.Contains('$parsed.passed') })
if ($doctorGuard.Count -ne 1 -or $chainGuard.Count -ne 1 -or $operatorGuard.Count -ne 1) { throw "Missing or ambiguous guard" }
$result = @()
foreach ($json in @('true', 'false', '1', '"true"', 'null')) {
    $doctorExit = 0; $chainExit = 0
    $doctor = ('{"ready":' + $json + '}') | ConvertFrom-Json
    $chain = ('{"valid":' + $json + '}') | ConvertFrom-Json
    $parsed = ('{"passed":' + $json + ',"integrity":"process_isolated"}') | ConvertFrom-Json
    $result += @{
        input_json = $json
        doctor_rejected = [bool](Invoke-Expression $doctorGuard[0].Clauses[0].Item1.Extent.Text)
        chain_rejected = [bool](Invoke-Expression $chainGuard[0].Clauses[0].Item1.Extent.Text)
        operator_rejected = [bool](Invoke-Expression $operatorGuard[0].Clauses[0].Item1.Extent.Text)
    }
}
$result | ConvertTo-Json -Compress
''', encoding="utf-8")
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(runner),
                             "-Source", str(source)], capture_output=True, text=True, timeout=30, check=True)
    rows = json.loads(result.stdout)
    assert len(rows) == 5
    for row in rows:
        expected_rejection = row["input_json"] != "true"
        assert row["doctor_rejected"] is expected_rejection
        assert row["chain_rejected"] is expected_rejection
        assert row["operator_rejected"] is expected_rejection
