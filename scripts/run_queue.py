"""Serial RTX 5070 queue.

One child at a time. The next item starts only after exit 0 and a complete
metrics file whose sibling sha256 matches the file text. Aliases copy the
source metrics. runs/QUEUE_DONE is written only after the last item.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import math
import queue as output_queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import codecs
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

from laplacian.arms import all_configs, get_config  # noqa: E402
from laplacian.runtime import apply_env, write_metrics  # noqa: E402

PY = ROOT / ".venv" / "Scripts" / "python.exe"
EXE = Path(os.environ.get("JOVIN_HARNESS_EXE") or shutil.which("jovin-harness") or
           r"D:\JovinHarness_Artifacts\environments\windows\Scripts\jovin-harness.exe")
RUNS = ROOT / "runs"
LEDGER = RUNS / "harness-ledger.jsonl"
DONE = RUNS / "QUEUE_DONE"
# Queue-level limits cover the Docker preflight and whole reference script,
# beyond internal per-operation timeouts. Timeout cleanup has its own bounds;
# stopping a Windows WSL client does not attest remote Linux process cleanup.
HARNESS_CHECK_TIMEOUT_S = 120
REFERENCE_GATE_TIMEOUT_S = 900
CHILD_REAP_TIMEOUT_S = 5


def child_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TEMP"] = str(ROOT / "cache" / "tmp")
    env["TMP"] = str(ROOT / "cache" / "tmp")
    env["PIP_CACHE_DIR"] = str(ROOT / "cache" / "pip")
    env["HF_HOME"] = str(ROOT / "cache" / "hf")
    env["HF_DATASETS_CACHE"] = str(ROOT / "cache" / "hf" / "datasets")
    env["TORCH_HOME"] = str(ROOT / "cache" / "torch")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["JOVIN_HARNESS_EXE"] = str(EXE)
    env["JOVIN_HARNESS_RUNTIME"] = "wsl:JovinHarness"
    env.setdefault("JOVIN_HARNESS_TIMEOUT_S", "30")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def heartbeat(line):
    path = RUNS / "heartbeat.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")


DATA_MEMMAPS = (
    "data/meta.json",
    "data/tinystories_train.u16",
    "data/tinystories_val.u16",
    "data/tinystories_smol_train.u16",
    "data/tinystories_smol_val.u16",
)


def data_files_ready():
    for relative in DATA_MEMMAPS:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def metrics_state(path):
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    if not path.exists() or not side.exists():
        return {"ok": False, "reason": "missing", "sha256": None, "logical_sha256": None}
    logical = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()
    recorded = side.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(logical)
    except json.JSONDecodeError:
        return {"ok": False, "reason": "json", "sha256": recorded, "logical_sha256": digest}
    ok = payload.get("complete") is True and recorded == digest
    reason = "ok" if ok else "incomplete_or_hash"
    return {
        "ok": ok,
        "reason": reason,
        "sha256": recorded,
        "logical_sha256": digest,
        "payload": payload,
    }


def append_ledger(row):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def json_document(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise SystemExit("jovin returned no JSON object")
    return json.loads(text[start : end + 1])


def _stop_timed_out_tree(proc):
    """Bound teardown itself; uncertain cleanup never turns a timeout into pass."""
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            print("Timeout process-group cleanup is uncertain", flush=True)
    # On Windows proc is the dedicated real-Python helper. Its sole Job Object
    # handle closes on process death, which kills its contained descendants.
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=CHILD_REAP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print("Timeout cleanup is uncertain: direct child did not exit within reap grace", flush=True)


def contained_queue_command(request):
    """Internal Windows helper; establish containment before any requested child.

    This entry point never runs ordinary queue main, WSL preflight or GPU checks.
    It also owns the deadline if the calling queue disappears unexpectedly.
    """
    from study_budget import windows_job
    _job = windows_job()  # Non-inheritable handle; keep alive until helper exit.
    remaining = request["deadline_monotonic"] - time.monotonic()
    if remaining <= 0:
        return 124
    child = subprocess.Popen(request["command"], stdin=subprocess.DEVNULL, shell=False)
    try:
        return child.wait(timeout=max(0, request["deadline_monotonic"] - time.monotonic()))
    except subprocess.TimeoutExpired:
        return 124  # Returning exits helper; the OS closes its job and tree.


def _bounded_stream(command, timeout_s):
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Command timeout must be finite and positive")
    deadline = time.monotonic() + timeout_s
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    actual_command = command
    if os.name == "nt":
        # Bypass Windows venv launcher shims so Popen.pid identifies the actual
        # process holding the Job Object, not a launcher waiting for that process.
        executable = getattr(sys, "_base_executable", sys.executable)
        actual_command = [executable, str(Path(__file__).resolve()), "_contained_queue_command",
                          json.dumps({"command": command, "deadline_monotonic": deadline})]
    proc = subprocess.Popen(actual_command, cwd=str(ROOT), env=child_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=0, **options)
    assert proc.stdout is not None
    chunks = output_queue.Queue(maxsize=64)
    finished = object()

    def read_pipe():
        try:
            while True:
                block = os.read(proc.stdout.fileno(), 4096)
                if not block:
                    break
                chunks.put(block)
        except OSError as exc:
            chunks.put(exc)
        finally:
            chunks.put(finished)

    reader = threading.Thread(target=read_pipe, name="queue-command-output", daemon=True)
    reader.start()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            try:
                block = chunks.get(timeout=remaining)
            except output_queue.Empty as exc:
                raise subprocess.TimeoutExpired(command, timeout_s) from exc
            if block is finished:
                print(decoder.decode(b"", final=True), end="", flush=True)
                return proc.wait(timeout=max(0, deadline - time.monotonic()))
            if isinstance(block, OSError):
                raise block
            print(decoder.decode(block), end="", flush=True)
    except BaseException:
        _stop_timed_out_tree(proc)
        raise
    finally:
        reader.join(timeout=1)
        # A detached descendant can keep a pipe open. Never wait unboundedly for
        # its reader; timeout already fails the gate and reports uncertain cleanup.
        if not reader.is_alive():
            proc.stdout.close()


def stream_command(command, *, timeout_s=None):
    if timeout_s is not None:
        return _bounded_stream(command, timeout_s)
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            print(line, end="", flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
    return proc.wait()


def foreign_pids():
    """Other queues. The venv python.exe shim stays alive as this process's parent."""
    me = os.getpid()
    parent = os.getppid()
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_queue.py*' } | "
        "ForEach-Object { $_.ProcessId }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    found = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.isdigit() and int(text) not in (me, parent):
            found.append(int(text))
    return found


def gpu_is_busy():
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        print(listing.stderr, flush=True)
        raise SystemExit("nvidia-smi failed before a GPU item")
    uuid = None
    for line in listing.stdout.splitlines():
        if "5070" in line:
            uuid = line.split(",")[-1].strip()
    if not uuid:
        raise SystemExit("RTX 5070 was not found by name")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if apps.returncode != 0:
        print(apps.stderr, flush=True)
        raise SystemExit("nvidia-smi compute-apps failed")
    for line in apps.stdout.splitlines():
        if uuid.lower() in line.lower():
            print(f"5070 compute app: {line}", flush=True)
            return True
    return False


def preflight_wsl():
    wsl = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "wsl.exe"
    print(f"WSL initiate: {wsl} --distribution JovinHarness --exec docker version", flush=True)
    try:
        code = stream_command([str(wsl), "--distribution", "JovinHarness", "--exec", "docker", "version"],
                              timeout_s=HARNESS_CHECK_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"WSL Docker preflight timed out after {HARNESS_CHECK_TIMEOUT_S} seconds; gate not completed") from exc
    if code != 0:
        raise SystemExit(code)
    print(f"WSL initiate: {EXE} doctor", flush=True)
    try:
        proc = subprocess.run([str(EXE), "doctor"], capture_output=True, text=True,
                              env=child_env(), check=False, timeout=HARNESS_CHECK_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"jovin doctor timed out after {HARNESS_CHECK_TIMEOUT_S} seconds") from exc
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    report = json_document(proc.stdout)
    if proc.returncode != 0 or report.get("ready") is not True:
        raise SystemExit("jovin doctor is not ready")
    return verify_receipt_chain()


def verify_receipt_chain():
    print(f"WSL initiate: {EXE} receipts verify", flush=True)
    try:
        chain = subprocess.run(
            [str(EXE), "receipts", "verify"],
            capture_output=True,
            text=True,
            env=child_env(),
            check=False,
            timeout=HARNESS_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"jovin receipts verify timed out after {HARNESS_CHECK_TIMEOUT_S} seconds") from exc
    print(chain.stdout, flush=True)
    if chain.stderr:
        print(chain.stderr, flush=True)
    receipt = json_document(chain.stdout)
    if chain.returncode != 0 or receipt.get("valid") is not True:
        raise SystemExit("jovin receipt chain is not valid")
    return receipt.get("head_sha256"), receipt.get("count")


GATE_SOURCES = {
    "pytest": ("src/**/*.py", "scripts/**/*.py", "tests/**/*.py", "reference/operators/**/*.py", "reference/tests/**/*.py",
               "pyproject.toml", "scripts/verify_reference.ps1", "paper/fetch_references.py", "paper/planning/FRACTIONAL_POLICY.md",
               "runs/optimization_20261021/baseline_losses.py", "paper/planning/timing_schedule.json"),
    "verify_reference": ("reference/operators/**/*.py", "reference/tests/**/*.py", "scripts/verify_reference.ps1", "scripts/study_budget.py"),
    "host_invariants": ("src/**/*.py", "reference/operators/fiedler.py"),
    "mcp_stdio": ("src/**/*.py",),
}


def gate_source_snapshot(item):
    """Content identity of correctness inputs; generated receipts are not inputs.

    Include paths as well as bytes so added/deleted tests invalidate old passes,
    even when a checkout or copy preserves filesystem timestamps. Historical
    training measurements intentionally retain their original resume behavior.
    """
    if item["id"] not in GATE_SOURCES:
        return None
    paths = {ROOT / "scripts" / "run_queue.py"}
    for pattern in GATE_SOURCES[item["id"]]:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    }
    identity = {"files": files, "command": item["cmd"]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return {"sha256": digest, **identity}


def gate_is_stale(item, state):
    expected = gate_source_snapshot(item)
    return expected is not None and state.get("payload", {}).get("verification_source") != expected


def correctness_gates_current(items):
    for item in items:
        if item["id"] in GATE_SOURCES:
            state = metrics_state(ROOT / item["metrics"])
            if not state["ok"] or gate_is_stale(item, state):
                return False
    return True


def write_gate(item_id, proof):
    return write_metrics(
        RUNS / item_id / "metrics.json",
        {"run_id": item_id, "complete": True, "host_gpu": False, "jovin_receipt": False,
         "kind": "host_gate", "proof": proof[-400:]},
    )


def run_mcp():
    message = {
        "id": "nilpotency-zero",
        "tool": "verification_gate",
        "arguments": {"kind": "nilpotency", "payload": {"residual_l1": 0}},
    }
    proc = subprocess.run(
        [str(PY), "-u", "-m", "laplacian.mcp_server"],
        input=json.dumps(message),
        capture_output=True,
        text=True,
        env=child_env(),
        cwd=str(ROOT),
        check=False,
    )
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode != 0:
        return proc.returncode
    body = json.loads(proc.stdout)
    if not body.get("ok") or not body.get("result", {}).get("pass"):
        return 1
    write_gate("mcp_stdio", proc.stdout)
    return 0


def copy_alias(item):
    source = RUNS / item["source"] / "metrics.json"
    state = metrics_state(source)
    if not state["ok"]:
        raise SystemExit(f"alias {item['id']} has no complete source {item['source']}")
    payload = dict(state["payload"])
    payload["run_id"] = item["id"]
    payload["alias"] = True
    payload["measurement_source"] = item["source"]
    payload["measurement_source_sha256"] = state["sha256"]
    payload["complete"] = True
    write_metrics(ROOT / item["metrics"], payload)


def grid_ids():
    names = []
    for exponent in (0.25, 0.5, 0.75):
        for layers in ([1, 5], [2, 5], [1, 3, 5]):
            if exponent == 0.5 and layers == [2, 5]:
                continue
            tag = str(exponent).replace(".", "p")
            layer_tag = "".join(str(layer) for layer in layers)
            names.append(f"grid_s{tag}_l{layer_tag}")
    return names


def run_item(name, command, metrics, gpu):
    return {"id": name, "kind": "run", "gpu": gpu, "cmd": command, "metrics": metrics}


def train_item(name):
    get_config(name)
    return run_item(name, [str(PY), "-u", "-m", "laplacian.train", "--arm", name], f"runs/{name}/metrics.json", True)


def build_items():
    configs = all_configs()
    grids = grid_ids()
    if set(grids) != {name for name in configs if name.startswith("grid_")}:
        raise SystemExit("grid ids drifted from arms.all_configs")
    items = [
        {
            "id": "pytest",
            "kind": "gate",
            "gpu": False,
            "cmd": [str(PY), "-m", "pytest", "tests", "-q", "--tb=line"],
            "metrics": "runs/pytest/metrics.json",
        },
        {
            "id": "verify_reference",
            "kind": "gate",
            "gpu": False,
            "cmd": [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "verify_reference.ps1"),
            ],
            "metrics": "runs/verify_reference/metrics.json",
        },
        run_item(
            "host_invariants",
            [str(PY), "-u", "-m", "laplacian.host_invariants"],
            "runs/host_invariants/metrics.json",
            False,
        ),
        {"id": "mcp_stdio", "kind": "mcp", "gpu": False, "cmd": [], "metrics": "runs/mcp_stdio/metrics.json"},
        run_item(
            "guidance_b",
            [str(PY), "-u", str(ROOT / "experiments" / "b_schur.py")],
            "runs/guidance_b/metrics.json",
            True,
        ),
        run_item(
            "guidance_c",
            [str(PY), "-u", str(ROOT / "experiments" / "c_morse.py")],
            "runs/guidance_c/metrics.json",
            True,
        ),
        run_item(
            "guidance_d",
            [str(PY), "-u", str(ROOT / "experiments" / "d_tropical.py")],
            "runs/guidance_d/metrics.json",
            True,
        ),
        run_item("data", [str(PY), "-u", "-m", "laplacian.data"], "runs/data/metrics.json", False),
        train_item("smoke_arm0"),
        train_item("smoke_pulse"),
        run_item(
            "eta",
            [str(PY), "-u", "-m", "laplacian.train", "--arm", "write_eta"],
            "runs/eta/metrics.json",
            False,
        ),
    ]
    for name in ("arm0", "arm1", "arm2"):
        items.append(train_item(name))
        items.append(
            run_item(
                f"eval_{name}",
                [str(PY), "-u", "-m", "laplacian.eval_run", "--arm", name],
                f"runs/eval_{name}/metrics.json",
                True,
            )
        )
    for name in grids:
        items.append(train_item(name))
    items.append(
        {
            "id": "grid_s0p5_l25",
            "kind": "alias",
            "gpu": False,
            "cmd": [],
            "source": "arm2",
            "metrics": "runs/grid_s0p5_l25/metrics.json",
        }
    )
    items.append(train_item("arm3"))
    items.append(train_item("arm3_masks"))
    items.append(train_item("arm4"))
    items.append(
        run_item(
            "eval_arm4",
            [str(PY), "-u", "-m", "laplacian.eval_run", "--arm", "arm4"],
            "runs/eval_arm4/metrics.json",
            True,
        )
    )
    for name in ("scale_m2", "scale_m4", "scale_m16"):
        items.append(train_item(name))
    items.append(
        {
            "id": "scale_m8",
            "kind": "alias",
            "gpu": False,
            "cmd": [],
            "source": "arm4",
            "metrics": "runs/scale_m8/metrics.json",
        }
    )
    items.append(train_item("distill_plain"))
    items.append(train_item("distill_manifold"))
    items.append(
        run_item(
            "arm5",
            [str(PY), "-u", "-m", "laplacian.smol", "--train"],
            "runs/arm5/metrics.json",
            True,
        )
    )
    items.append(
        run_item(
            "arm5_eval",
            [str(PY), "-u", "-m", "laplacian.smol", "--eval"],
            "runs/arm5_eval/metrics.json",
            True,
        )
    )
    items.append(
        run_item(
            "fetch_references",
            [str(PY), "-u", str(ROOT / "paper" / "fetch_references.py")],
            "runs/fetch_references/metrics.json",
            False,
        )
    )
    return items


def backfill(head, count):
    if LEDGER.exists() and '"id": "guidance_a"' in LEDGER.read_text(encoding="utf-8"):
        return
    path = RUNS / "guidance_a" / "metrics.json"
    state = metrics_state(path)
    append_ledger(
        {
            "id": "guidance_a",
            "exit": 0,
            "metrics_path": "runs/guidance_a/metrics.json",
            "sha256": state.get("sha256"),
            "logical_sha256": state.get("logical_sha256"),
            "hash_match": state.get("sha256") == state.get("logical_sha256") and bool(state.get("sha256")),
            "backfill": True,
            "complete_flag": False,
            "jovin_head": head,
            "jovin_count": count,
            "note": "Experiment A finished before the queue. This line does not re-run it.",
        }
    )


def ledger_row(item, head, count, exit_code, state, skipped):
    return {
        "id": item["id"],
        "exit": exit_code,
        "metrics_path": item["metrics"],
        "sha256": state.get("sha256"),
        "jovin_head": head,
        "jovin_count": count,
        "skipped": skipped,
        "alias_of": item.get("source"),
        "host_gpu": None if state.get("payload") is None else state["payload"].get("host_gpu"),
    }


def execute(item):
    snapshot = gate_source_snapshot(item)
    if item["kind"] == "alias":
        copy_alias(item)
        return 0
    if item["kind"] == "mcp":
        code = run_mcp()
    else:
        if item["id"] == "verify_reference":
            try:
                code = stream_command(item["cmd"], timeout_s=REFERENCE_GATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"FAILED reference gate timed out after {REFERENCE_GATE_TIMEOUT_S} seconds; no completion attested", flush=True)
                return 124
        else:
            code = stream_command(item["cmd"])
        if code == 0 and item["kind"] == "gate":
            write_gate(item["id"], item["id"])
    if code == 0 and snapshot is not None:
        if snapshot != gate_source_snapshot(item):
            print(f"FAILED {item['id']} source changed during verification; rerun required", flush=True)
            return 1
        state = metrics_state(ROOT / item["metrics"])
        if not state["ok"]:
            return 1
        payload = dict(state["payload"])
        payload["verification_source"] = snapshot
        write_metrics(ROOT / item["metrics"], payload)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates-only", action="store_true",
                        help="Run/check correctness gates only; never start training or write QUEUE_DONE.")
    args = parser.parse_args(argv)
    apply_env()
    RUNS.mkdir(parents=True, exist_ok=True)
    items = build_items()
    if args.gates_only:
        items = [item for item in items if item["id"] in GATE_SOURCES]
    if not args.gates_only and DONE.exists() and correctness_gates_current(items):
        print("SKIP queue runs/QUEUE_DONE exists", flush=True)
        return 0
    others = foreign_pids()
    if others:
        print(f"FAILED another run_queue.py is alive: {others}", flush=True)
        return 1
    if not args.gates_only and DONE.exists():
        DONE.unlink()
        print("RECHECK queue correctness inputs changed", flush=True)
    heartbeat(f"QUEUE preflight pid={os.getpid()}")
    print(f"QUEUE pid={os.getpid()}", flush=True)
    head, count = preflight_wsl()
    print(f"jovin head {head} count {count}", flush=True)
    backfill(head, count)
    for item in items:
        state = metrics_state(ROOT / item["metrics"])
        stale = gate_is_stale(item, state)
        # The data metrics hash can be present on a clean checkout while the
        # gitignored token memmaps are not. Skipping then fails the first train.
        needs_memmaps = item["id"] == "data" and not data_files_ready()
        if state["ok"] and not stale and not needs_memmaps:
            append_ledger(ledger_row(item, head, count, 0, state, True))
            print(f"SKIP {item['id']} sha256={state['sha256']}", flush=True)
            continue
        if item["gpu"] and gpu_is_busy():
            print(f"FAILED {item['id']} RTX 5070 already has a compute process", flush=True)
            return 1
        heartbeat(f"QUEUE start {item['id']}")
        print(f"QUEUE start {item['id']}", flush=True)
        code = execute(item)
        if code == 0 and item["id"] == "verify_reference":
            # The verification command appends twelve receipts. Record the
            # verified chain after that run, not the preflight's older head.
            head, count = verify_receipt_chain()
        state = metrics_state(ROOT / item["metrics"])
        append_ledger(ledger_row(item, head, count, code, state, False))
        if code != 0 or not state["ok"]:
            print(
                f"FAILED {item['id']} exit={code} metrics_ok={state['ok']} reason={state.get('reason')}",
                flush=True,
            )
            return code or 1
        print(f"QUEUE pass {item['id']} sha256={state['sha256']}", flush=True)
    if args.gates_only:
        heartbeat("QUEUE correctness gates done")
        print("DONE correctness gates; training was not started", flush=True)
        return 0
    DONE.write_text(
        json.dumps({"complete": True, "items": len(items), "jovin_head_at_start": head}, indent=2) + "\n",
        encoding="utf-8",
    )
    heartbeat("QUEUE done")
    print("DONE queue runs/QUEUE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["_contained_queue_command"]:
        raise SystemExit(contained_queue_command(json.loads(sys.argv[2])))
    raise SystemExit(main())
