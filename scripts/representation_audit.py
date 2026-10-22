"""Prepare/run a bounded CPU-only representation audit outside frozen training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import concept_status as status
import study_budget as budget

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "representation_audit_manifest_v1"
STEPS = [0, 5000, 25000, 50000]
LIMITS = {"threads": 1, "elapsed_seconds": 7200, "memory_bytes": 2 * 1024**3}
SAMPLE_SEED = 20261022
SCOPE = "host_cpu_training_only_diagnostics"
OWN_FILES = ["scripts/representation_audit.py", "scripts/concept_status.py", "scripts/study_budget.py",
             "research/representation_audit/worker.py", "research/representation_audit/limits.py",
             "research/representation_audit/metrics.py", "research/representation_audit/contracts.json"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return budget.canonical(value)


def config_hash(config):
    # Match the frozen trainer's identity convention exactly: run_id is excluded.
    return hashlib.sha256(canonical({key: value for key, value in config.items() if key != "run_id"})).hexdigest()


def record(path):
    path = status.local(ROOT, path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def check(row):
    require(record(row["path"]) == {k: row[k] for k in ("path", "bytes", "sha256")}, "Bound audit input changed: " + Path(row["path"]).name)


def immutable(path, value):
    raw = canonical(value) + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii")


def offsets(token_count, context=256):
    require(type(token_count) is int and token_count > context + 1, "Invalid training token length")
    return [[int.from_bytes(hashlib.sha256(f"representation-audit-v1:{SAMPLE_SEED}:{batch}:{row}".encode()).digest(), "big")
             % (token_count - context - 1) for row in range(8)] for batch in range(8)]


def source_records(parent):
    paths = list(dict.fromkeys([str(ROOT / name) for name in OWN_FILES]
                              + [row["path"] for row in parent["source_records"]]))
    return [record(path) for path in paths]


def verify_receipt(path, sources):
    verified = status.document(ROOT, path)
    require(verified.get("kind") == "representation_audit_verification_v1" and verified.get("complete") is True,
            "Audit verification is not complete")
    for row in verified["sources"] + verified["outputs"]:
        check(row)
    require(all(row in verified["sources"] for row in sources), "Verification does not bind every audit implementation source")


def locate_continuation(stage):
    candidates = []
    for path in (ROOT / "runs/publication_study").glob("continuation_ce_*/manifest.json"):
        doc = status.document(ROOT, path)
        if Path(doc["parent_stage"]["path"]).resolve() == stage:
            candidates.append(path)
    require(len(candidates) == 1, "Exactly one bound CE continuation is required; use --continuation to disambiguate")
    return candidates[0]


def prepare(stage_path, output, continuation_path=None, verification_path=None):
    stage_path, output = status.local(ROOT, stage_path), status.local(ROOT, output)
    require(not output.exists(), "Audit preparation never overwrites a prior output")
    parent = status.document(ROOT, stage_path)
    continuation_path = status.local(ROOT, continuation_path) if continuation_path else locate_continuation(stage_path)
    continuation = status.document(ROOT, continuation_path)
    require(continuation["schema"] == "ce_parent_continuation_v1", "Unknown CE continuation")
    check(continuation["parent_stage"])
    require(Path(continuation["parent_stage"]["path"]).resolve() == stage_path, "Different original stage")
    for row in parent["source_records"] + continuation["configs"] + [continuation["self"], continuation["python"]]:
        check(row)
    admission_path = continuation_path.parent / "seed0_admission.json"
    admission = status.document(ROOT, admission_path)
    require(admission["parent"]["seed"] == 0 and admission["parent"]["updates"] == 50000, "Seed 0 is not admitted")
    config_record = next(row["config"] for row in parent["ce_parents"] if row["seed"] == 0)
    check(config_record)
    config_doc = json.loads(Path(config_record["path"]).read_bytes())
    config = config_doc["config"]
    require(config_hash(config) == config_doc["config_sha256"], "Config identity differs")
    require(config["study"]["stage"] == "ce_pretrain" and config["seed"] == 0
            and config["context"] == 256 and config["microbatch"] == 8 and config["steps"] == 50000,
            "Audit v1 requires the admitted seed0 CE B8/context256/50k configuration")
    snapshots = admission["parent"]["snapshots"]
    require([Path(r["path"]).name for r in snapshots] == [f"state_step{s:07d}.pt" for s in STEPS], "Only the four immutable retained snapshots are admitted")
    for row in snapshots:
        require(Path(row["path"]).parent == Path(config_record["path"]).parent, "Checkpoint outside the seed0 folder")
        check(row)
    token_manifest = config["study"]["token_manifest"]
    check(token_manifest)
    folder = Path(token_manifest["path"]).parent
    train_inputs = []
    for name in ("train.u16", "train.stories.jsonl"):
        row = {"path": str(folder / name), **config["study"]["data_files"][name]}
        check(row)
        train_inputs.append(row)
    require(train_inputs[0]["bytes"] % 2 == 0, "Malformed uint16 training stream")
    require(verification_path is not None, "Prepare requires a completed, source-bound --verification receipt")
    verification_path = status.local(ROOT, verification_path)
    sources = source_records(parent)
    verify_receipt(verification_path, sources)
    manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "original_stage": record(stage_path), "continuation": record(continuation_path),
                "seed0_admission": record(admission_path), "config": config_record, "snapshots": snapshots,
                "steps": STEPS, "train_inputs": train_inputs, "sample_offsets": offsets(train_inputs[0]["bytes"] // 2),
                "sampling": {"split": "train", "seed": SAMPLE_SEED, "batches": 8, "microbatch": 8, "context": 256,
                             "policy": "sha256_mod_train_window_v1", "same_windows_at_each_checkpoint": True},
                "limits": LIMITS, "python": record(sys.executable), "sources": sources,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy")},
                "verification": record(verification_path), "scope": SCOPE,
                "device": "cpu", "confirmatory_claim_permitted": False, "publication_admitted": False}
    output.mkdir(parents=True)
    for row in sources:
        target = output / "source_archive" / Path(row["path"]).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    immutable(output / "manifest.json", manifest)
    return output / "manifest.json"


def checked_manifest(path):
    path = status.local(ROOT, path)
    m = status.document(ROOT, path)
    require(m["schema"] == SCHEMA and m["scope"] == SCOPE and m["device"] == "cpu"
            and m["confirmatory_claim_permitted"] is False and m["publication_admitted"] is False,
            "Invalid audit scope/schema")
    require(m["limits"] == LIMITS and m["steps"] == STEPS and m["sampling"] == {
        "split": "train", "seed": SAMPLE_SEED, "batches": 8, "microbatch": 8, "context": 256,
        "policy": "sha256_mod_train_window_v1", "same_windows_at_each_checkpoint": True}, "Audit resource/sample policy changed")
    require([Path(r["path"]).name for r in m["train_inputs"]] == ["train.u16", "train.stories.jsonl"], "Nontraining data is forbidden")
    require(m["sample_offsets"] == offsets(m["train_inputs"][0]["bytes"] // 2), "Diagnostic sample offsets changed")
    require([Path(r["path"]).name for r in m["snapshots"]] == [f"state_step{s:07d}.pt" for s in STEPS], "Checkpoint roles differ")
    parent = status.document(ROOT, m["original_stage"]["path"])
    continuation = status.document(ROOT, m["continuation"]["path"])
    admission = status.document(ROOT, m["seed0_admission"]["path"])
    require(continuation["parent_stage"] == m["original_stage"] and admission["manifest"] == m["continuation"],
            "Audit stage/admission lineage differs")
    expected_config = next(row["config"] for row in parent["ce_parents"] if row["seed"] == 0)
    require(m["config"] == expected_config and m["snapshots"] == admission["parent"]["snapshots"], "Checkpoint/config lineage differs")
    cfg_doc = json.loads(Path(m["config"]["path"]).read_bytes())
    cfg = cfg_doc["config"]
    require(config_hash(cfg) == cfg_doc["config_sha256"] and cfg["seed"] == 0
            and cfg["steps"] == 50000 and cfg["context"] == 256 and cfg["microbatch"] == 8
            and cfg["study"]["stage"] == "ce_pretrain", "Invalid frozen seed0 config")
    token_folder = Path(cfg["study"]["token_manifest"]["path"]).parent
    expected_train = [{"path": str(token_folder / name), **cfg["study"]["data_files"][name]}
                      for name in ("train.u16", "train.stories.jsonl")]
    require(m["train_inputs"] == expected_train, "Training data lineage differs")
    for row in parent["source_records"] + continuation["configs"] + [continuation["self"], continuation["python"], cfg["study"]["token_manifest"]]:
        check(row)
    require(m["sources"] == source_records(parent), "Implementation source set differs")
    verify_receipt(m["verification"]["path"], m["sources"])
    for row in [m[k] for k in ("original_stage", "continuation", "seed0_admission", "config", "python", "verification")] + m["sources"]:
        check(row)
    require(m["python"] == record(sys.executable), "Interpreter differs")
    require(m["packages"] == {name: importlib.metadata.version(name) for name in ("torch", "numpy")}, "Runtime packages changed")
    # Hash large binaries in streaming chunks; do not deserialize checkpoints here.
    for row in m["snapshots"] + m["train_inputs"]:
        check(row)
    return m


def readiness(m):
    continuation = Path(m["continuation"]["path"])
    current = json.loads((continuation.parent / "status.json").read_bytes())
    if current["state"] != "completed":
        return {"ready": False, "reason": "CE continuation has not completed", "controller_state": current["state"]}
    mapping = {"execution": {"continuation_manifest": str(continuation), "ledger": str(budget.DEFAULT_LEDGER),
                             "grant": str(budget.DEFAULT_DECISIONS)}}
    observed = status.execution(ROOT, mapping)
    require(observed["completed_blocks"] == 50 and observed["completed_ce_parents"] == 5
            and not observed["open_reservations"], "Incomplete CE/accounting evidence")
    return {"ready": True, "ledger_head": observed["ledger_head"]}


def contain_worker(proc):
    """Own the child's kill-on-close job before permitting model/runtime work."""
    sys.path.insert(0, str(ROOT / "research/representation_audit"))
    import limits
    held = None
    try:
        require(os.name == "nt", "Parent-owned audit containment currently requires Windows")
        held = limits.establish(LIMITS["memory_bytes"], process_handle=proc._handle)
        proc.stdin.write(b"AUDIT_CONTAINED_V1\n")
        proc.stdin.flush()
        proc.stdin.close()
        return held
    except Exception:
        proc.kill()
        proc.wait()
        limits.close(held)
        raise


def release_worker(held):
    import limits
    limits.close(held)


def validate_outputs(directory, manifest, expected_layers=5):
    result = status.document(ROOT, directory / "worker_result.json")
    require(result.get("state") == "completed" and result.get("complete") is True
            and result.get("checkpoints") == 4 and result.get("microbatches") == 32
            and result.get("scope") == SCOPE and result.get("device") == "cpu"
            and result.get("torch_threads") == 1 and result.get("gpu_invocations") == 0
            and result.get("held_out_examples") == 0 and result.get("confirmatory_claim_permitted") is False,
            "Incomplete worker completion evidence")
    lines = (directory / "measurements.jsonl").read_bytes()
    require(lines.endswith(b"\n"), "Incomplete measurements file")
    rows = [json.loads(line) for line in lines.splitlines()]
    require(len(rows) == 32, "Expected exactly 32 checkpoint/microbatch observations")
    window_hashes = {}
    for row, (step, batch) in zip(rows, [(step, b) for step in STEPS for b in range(8)]):
        canonical(row)  # Reject nonfinite JSON values, including nested metrics.
        require(row["step"] == step and row["batch"] == batch
                and row["starts"] == manifest["sample_offsets"][batch], "Diagnostic sample order differs")
        value = row["window_sha256"]
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "Invalid window identity")
        require(window_hashes.setdefault(batch, value) == value, "Checkpoint windows differ")
        metrics = row["metrics"]
        require(metrics["schema"] == "representation_metrics_v1" and metrics["device"] == "cpu"
                and [layer["layer"] for layer in metrics["layers"]] == list(range(expected_layers)), "Diagnostic layer inventory differs")
    proposal = status.document(ROOT, directory / "next_experiment.json")
    require(proposal["state"] == "proposal_only" and proposal["proposed_budget"]["gpu_hours"] == 0
            and proposal["confirmatory_claim_permitted"] is False, "Invalid follow-on proposal scope")


def run(manifest_path):
    manifest_path = status.local(ROOT, manifest_path)
    m = checked_manifest(manifest_path)
    ready = readiness(m)
    if not ready["ready"]:
        return ready
    directory = manifest_path.parent / "attempt"
    # Keep canonical lock throughout CPU execution: no new GPU reservation may
    # race admission. The ledger bytes and training files remain unchanged.
    with budget.ledger_lock(budget.DEFAULT_LEDGER):
        ready = readiness(m)
        if not ready["ready"]:
            return ready
        directory.mkdir(exist_ok=False)
        immutable(directory / "started.json", {"manifest": record(manifest_path), "admission": ready,
                  "utc": datetime.now(timezone.utc).isoformat(), "retry": "forbidden"})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
        started = time.monotonic()
        code, timed_out, state, error, held = None, False, "failed", None, None
        try:
            with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                proc = subprocess.Popen([sys.executable, str(ROOT / "research/representation_audit/worker.py"),
                                         str(manifest_path), str(directory)], cwd=ROOT, env=env,
                                        stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, shell=False)
                held = contain_worker(proc)
                try:
                    code = proc.wait(timeout=LIMITS["elapsed_seconds"])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc.kill()
                    proc.wait()
            worker_path = directory / "worker_result.json"
            worker = json.loads(worker_path.read_bytes()) if worker_path.exists() else {}
            state = "resource_exhausted" if timed_out else worker.get("state", "failed")
            require(state in ("completed", "failed", "resource_exhausted"), "Unknown worker result state")
            if state == "completed":
                require(code == 0, "Completed worker returned a failure code")
                validate_outputs(directory, m)
                checked_manifest(manifest_path)
                require(readiness(m)["ready"], "Completion admission changed")
        except Exception as exc:
            state, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            if held is not None:
                release_worker(held)
        outputs = [record(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix != ".sha256"]
        receipt = {"kind": "representation_audit_receipt_v1", "state": state, "complete": state == "completed",
                   "manifest": record(manifest_path), "sources": m["sources"], "outputs": outputs,
                   "scope": SCOPE, "device": "cpu", "limits": LIMITS, "returncode": code,
                   "elapsed_seconds": time.monotonic() - started, "timeout": timed_out,
                   "error": error,
                   "checkpoint_and_input_integrity": "passed" if state == "completed" else "not_completed",
                   "confirmatory_claim_permitted": False, "publication_admitted": False}
        immutable(directory / "receipt.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--continuation", type=Path)
    p.add_argument("--verification", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = {"manifest": str(prepare(args.stage, args.output, args.continuation, args.verification)), "state": "prepared"}
        else:
            result = run(args.manifest)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("ready") is False:
            return 3
        return 0 if result.get("state") not in ("failed", "resource_exhausted") else 2
    except (ValueError, KeyError, TypeError, OSError, budget.BudgetError) as exc:
        print(json.dumps({"state": "rejected", "error": str(exc), "publication_admitted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
