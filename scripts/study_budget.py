"""Fail-closed serial wall-time accounting for the approved publication study.

No work is scheduled by this module. Every run reserves its entire timeout
before starting; crashed/uncertain runs keep that entire charge.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_APPROVED_HOURS = 181.0
DEFAULT_DECISIONS = ROOT / "paper" / "planning" / "decisions.json"
DEFAULT_LEDGER = ROOT / "runs" / "publication_study" / "budget.jsonl"
SECRET_NAME = re.compile(r"token|secret|password|passwd|credential|api[-_]?key|authorization", re.I)


class BudgetError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def approval(path, ledger):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    grant = data.get("recommended_core_gpu_hours", {})
    cap = grant.get("hard_cap")
    if grant.get("approved") is not True or isinstance(cap, bool) or not isinstance(cap, (int, float)):
        raise BudgetError("Core GPU budget has no explicit approved hard_cap")
    if not math.isfinite(cap) or not 0 < cap <= MAX_APPROVED_HOURS:
        raise BudgetError("Approved cap must be positive and cannot exceed 181 GPU-hours")
    approved_ledger = grant.get("ledger")
    if not isinstance(approved_ledger, str) or not approved_ledger:
        raise BudgetError("Approval must bind exactly one ledger path")
    bound = Path(approved_ledger)
    if not bound.is_absolute():
        bound = ROOT / bound
    if bound.resolve() != Path(ledger).resolve():
        raise BudgetError("Requested ledger differs from the approved ledger; no separate allowance")
    return float(cap) * 3600, hashlib.sha256(raw).hexdigest()


@contextlib.contextmanager
def ledger_lock(path):
    """OS-released lock; never delete/recreate its inode to break a live lock."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BudgetError("Another budget invocation holds the study lock") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def read_ledger(path):
    path = Path(path)
    if not path.exists():
        artifacts = path.parent / "runs"
        if artifacts.exists() and any(artifacts.iterdir()):
            raise BudgetError("Ledger missing beside existing run artifacts; refusing a fresh allowance")
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise BudgetError("Truncated ledger: preserve it and audit before any further run")
    records, previous = [], None
    try:
        for index, line in enumerate(raw.splitlines()):
            record = json.loads(line)
            digest = record.pop("sha256")
            if record["sequence"] != index or record["previous_sha256"] != previous:
                raise ValueError("sequence or chain mismatch")
            if hashlib.sha256(canonical(record)).hexdigest() != digest:
                raise ValueError("event hash mismatch")
            record["sha256"] = digest
            previous = digest
            records.append(record)
    except (ValueError, KeyError, TypeError) as exc:
        raise BudgetError(f"Invalid budget ledger: {exc}") from exc
    runs = accounting(records)
    artifacts = path.parent / "runs"
    inventory = {item.name for item in artifacts.iterdir()} if artifacts.exists() else set()
    if inventory != set(runs):
        raise BudgetError("Run artifacts and reservation inventory disagree; refusing a reset or missing evidence")
    return records


def append_event(path, records, event, **fields):
    record = {"sequence": len(records), "previous_sha256": records[-1]["sha256"] if records else None,
              "event": event, "utc": utc_now(), **fields}
    record["sha256"] = hashlib.sha256(canonical(record)).hexdigest()
    with Path(path).open("ab") as handle:
        handle.write(canonical(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    records.append(record)
    return record


def accounting(records):
    runs = {}
    for record in records:
        event, run_id = record["event"], record.get("run_id")
        if event == "reserved":
            reserve = record.get("reserve_seconds")
            if run_id in runs or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve <= 0:
                raise BudgetError("Invalid or duplicate reservation")
            runs[run_id] = {"reservation": record, "charge_seconds": reserve, "settled": False}
        elif event in ("finished", "recovered"):
            if run_id not in runs or runs[run_id]["settled"]:
                raise BudgetError("Settlement without one unique open reservation")
            charge = record.get("charge_seconds")
            reserve = runs[run_id]["reservation"]["reserve_seconds"]
            if not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0:
                raise BudgetError("Invalid settlement charge")
            if event == "recovered" and charge < reserve:
                raise BudgetError("Recovery cannot refund uncertain time")
            if record.get("status") != "completed" and charge < reserve:
                raise BudgetError("Failed/uncertain runs retain their whole reservation")
            runs[run_id].update(charge_seconds=charge, settled=True, settlement=record)
        else:
            raise BudgetError(f"Unknown ledger event: {event}")
    return runs


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def public_command(command):
    result, hide_next = [], False
    for part in command:
        if hide_next:
            result.append("<redacted>")
            hide_next = False
        elif SECRET_NAME.search(part.split("=", 1)[0]) and part.startswith("-"):
            result.append(part.split("=", 1)[0] + "=<redacted>" if "=" in part else part)
            hide_next = "=" not in part
        elif part in ("-c", "--command", "-Command", "-EncodedCommand"):
            result.append(part)
            hide_next = True
        elif "://" in part and "@" in part:
            result.append("<redacted-url>")
        else:
            result.append(part)
    return result


def provenance(command, device):
    # Deliberately exclude the complete environment, PATH, and credential values.
    allowed = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
    packages = {}
    for name in ("torch", "numpy", "transformers", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages,
            "platform": platform.platform(), "cwd": str(ROOT), "command": public_command(command),
            "cuda_visible_devices": device, "environment": {key: os.environ[key] for key in allowed if key in os.environ},
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def secret_values(command):
    values = {value for key, value in os.environ.items() if SECRET_NAME.search(key) and len(value) >= 4}
    hide_next = False
    for part in command:
        if hide_next:
            values.add(part)
            hide_next = False
        elif part.startswith("-") and SECRET_NAME.search(part.split("=", 1)[0]):
            if "=" in part:
                values.add(part.split("=", 1)[1])
            else:
                hide_next = True
    return sorted((value for value in values if value), key=len, reverse=True)


def copy_redacted(stream, target, secrets):
    try:
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            for value in secrets:
                text = text.replace(value, "<redacted>")
            target.write(text)
            target.flush()
    except (OSError, ValueError):
        pass  # Timeout teardown can close a pipe while its reader is active.


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied is uncertainty, not evidence of termination.
            return ctypes.get_last_error() != 87
        code = ctypes.c_uint32()
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return not ok or code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def windows_job():
    """Place supervisor itself in a non-inheritable kill-on-close job.

    Descendants inherit membership; the OS closes our sole job handle even if
    the supervisor is killed. Failure to establish containment prevents launch.
    """
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("per_process", ctypes.c_int64), ("per_job", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("active_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("basic", BASIC), ("io", IO), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    limits = EXTENDED()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
    if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise OSError(ctypes.get_last_error(), "Cannot establish Windows job limits")
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "Cannot contain supervisor in Windows job")
    return job  # Kept open until process exit; never inherited by descendants.


def supervise(directory):
    directory = Path(directory)
    try:
        _job = windows_job() if os.name == "nt" else None
        atomic_json(directory / "supervisor.json", {"pid": os.getpid(), "utc": utc_now(), "contained": True})
        request = json.loads(sys.stdin.readline())
        remaining = request["deadline_monotonic"] - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Reservation expired before child launch")
        with (directory / "stdout.log").open("w", encoding="utf-8") as out, (directory / "stderr.log").open("w", encoding="utf-8") as err:
            child = subprocess.Popen(request["command"], cwd=request["cwd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     stdin=subprocess.DEVNULL, shell=False)
            secrets = secret_values(request["command"])
            readers = [threading.Thread(target=copy_redacted, args=(stream, target, secrets), daemon=True)
                       for stream, target in ((child.stdout, out), (child.stderr, err))]
            for reader in readers:
                reader.start()
            try:
                code = child.wait(timeout=max(0.0, request["deadline_monotonic"] - time.monotonic()))
                result = {"status": "completed" if code == 0 else "failed", "returncode": code}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "returncode": None}
            for reader in readers:
                reader.join(timeout=0.02)
            result["logs_complete"] = all(not reader.is_alive() for reader in readers)
            if result["status"] == "completed" and not result["logs_complete"]:
                result["status"] = "log_incomplete"
        atomic_json(directory / "result.json", result)
        return 0
    except BaseException as exc:
        atomic_json(directory / "result.json", {"status": "supervisor_failure", "error_type": type(exc).__name__})
        return 1
    finally:
        if os.name != "nt":
            # This also executes after the accounting parent crashes. Results
            # are durable before terminating this supervisor and its group.
            os.killpg(os.getpgrp(), signal.SIGKILL)
    # Windows kernel closes the job handle on exit, killing any descendants.


def stop_supervisor(proc):
    if os.name == "nt":
        if proc.poll() is None:
            proc.kill()  # Closing its job kills the complete contained tree.
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_study(args):
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise BudgetError("A child command is required after --")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        raise BudgetError("run-id must be a short filename-safe unique identifier")
    if not math.isfinite(args.reserve_seconds) or args.reserve_seconds < 0.5:
        raise BudgetError("reserve-seconds must be finite and at least 0.5")
    if "," in args.cuda_device or not (args.cuda_device == "" or args.cuda_device.isdigit() or args.cuda_device.startswith("GPU-")):
        raise BudgetError("Select exactly one CUDA device (or empty for CPU verification)")
    ledger = Path(args.ledger)
    with ledger_lock(ledger):
        cap, approval_hash = approval(args.decisions, ledger)
        records = read_ledger(ledger)
        runs = accounting(records)
        if any(not run["settled"] for run in runs.values()):
            raise BudgetError("Unfinished reservation exists; explicit recovery is required")
        if args.run_id in runs:
            raise BudgetError("Run ID is immutable and already exists")
        used = sum(run["charge_seconds"] for run in runs.values())
        if used + args.reserve_seconds > cap:
            raise BudgetError(f"Reservation exceeds approved remaining budget: {max(0, cap-used):.3f} seconds")
        directory = ledger.parent / "runs" / args.run_id
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "provenance.json", provenance(command, args.cuda_device))
        started, started_unix = time.monotonic(), time.time()
        grace = min(5.0, args.reserve_seconds * 0.2)
        append_event(ledger, records, "reserved", run_id=args.run_id, reserve_seconds=args.reserve_seconds,
                     cap_seconds=cap, approval_sha256=approval_hash, started_unix=started_unix,
                     deadline_unix=started_unix + args.reserve_seconds, directory=str(directory.resolve()),
                     supervisor_cutoff_unix=started_unix + args.reserve_seconds - grace,
                     supervisor_cutoff_monotonic=started + args.reserve_seconds - grace)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        env["PYTHONUNBUFFERED"] = "1"
        env["STUDY_BUDGET_RUN_ID"] = args.run_id
        env["STUDY_BUDGET_LEDGER"] = str(ledger.resolve())
        env["STUDY_BUDGET_DEADLINE_UNIX"] = str(started_unix + args.reserve_seconds - grace)
        kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        proc, status, code = None, "launch_failure", None
        try:
            with (directory / "supervisor.log").open("wb") as output:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_supervise", str(directory.resolve())],
                                        stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, env=env, **kwargs)
                request = {"command": command, "cwd": str(ROOT),
                           "deadline_monotonic": started + args.reserve_seconds - grace}
                proc.stdin.write(canonical(request) + b"\n")
                proc.stdin.close()
                proc.wait(timeout=max(0.01, args.reserve_seconds - grace / 2 - (time.monotonic() - started)))
            stop_supervisor(proc)  # POSIX descendants may remain after command exit.
            proc.wait(timeout=max(0.01, grace / 2))
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            status, code = result["status"], result.get("returncode")
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "uncertain"
            if proc is not None:
                stop_supervisor(proc)
                try:
                    proc.wait(timeout=max(0.01, grace / 2))
                except subprocess.TimeoutExpired:
                    # Do not settle or refund if containment termination is unconfirmed.
                    raise BudgetError("Termination unconfirmed; reservation remains fully charged") from exc
        elapsed = time.monotonic() - started
        # Only a successful, reaped, contained tree earns a measured-time refund.
        charge = elapsed if status == "completed" else max(args.reserve_seconds, elapsed)
        append_event(ledger, records, "finished", run_id=args.run_id, status=status, returncode=code,
                     elapsed_seconds=elapsed, charge_seconds=charge,
                     overrun_seconds=max(0.0, elapsed - args.reserve_seconds))
        print(json.dumps({"run_id": args.run_id, "status": status, "charge_seconds": charge,
                          "remaining_seconds": max(0.0, cap - used - charge), "logs": str(directory)}, indent=2))
        return 0 if status == "completed" else (code if isinstance(code, int) and 0 < code < 126 else 1)


def status_or_recover(args):
    ledger = Path(args.ledger)
    with ledger_lock(ledger):
        cap, approval_hash = approval(args.decisions, ledger)
        records = read_ledger(ledger)
        runs = accounting(records)
        if args.action == "recover":
            run = runs.get(args.run_id)
            if run is None or run["settled"]:
                raise BudgetError("Recovery requires an unfinished reservation")
            reservation = run["reservation"]
            if time.time() <= reservation["deadline_unix"]:
                raise BudgetError("Recovery cannot precede the original reservation deadline")
            identity_path = Path(reservation["directory"]) / "supervisor.json"
            if identity_path.exists():
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                if not identity.get("contained") or process_alive(identity["pid"]):
                    raise BudgetError("Supervisor termination cannot be proven; retain reservation")
                if os.name != "nt":
                    try:
                        os.killpg(identity["pid"], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise BudgetError("Original process group remains; manual termination audit required")
            elif (Path(reservation["directory"]) / "result.json").exists():
                # A setup failure before containment cannot have launched the command.
                result = json.loads((Path(reservation["directory"]) / "result.json").read_text())
                if result.get("status") != "supervisor_failure":
                    raise BudgetError("Missing supervisor identity; cannot establish containment")
            else:
                raise BudgetError("Missing supervisor identity; preserve reservation for manual audit")
            append_event(ledger, records, "recovered", run_id=args.run_id, status="abandoned",
                         charge_seconds=reservation["reserve_seconds"], approval_sha256=approval_hash)
            runs = accounting(records)
        used = sum(run["charge_seconds"] for run in runs.values())
        print(json.dumps({"cap_hours": cap / 3600, "charged_hours": used / 3600,
                          "remaining_hours": max(0.0, cap - used) / 3600,
                          "unfinished": [name for name, run in runs.items() if not run["settled"]],
                          "runs": len(runs), "ledger": str(ledger)}, indent=2))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_supervise"]:
        return supervise(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--reserve-seconds", type=float, required=True)
    run.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    run.add_argument("command", nargs=argparse.REMAINDER)
    recover = sub.add_parser("recover")
    recover.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        return run_study(args) if args.action == "run" else status_or_recover(args)
    except (BudgetError, OSError, ValueError) as exc:
        print(f"BUDGET BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
