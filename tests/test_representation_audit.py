"""CPU controller checks; dispatcher tests use an explicitly fake worker process."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import representation_audit as audit


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    interpreter = tmp_path / "python.exe"
    interpreter.write_bytes(b"fixture interpreter: no execution")
    monkeypatch.setattr(audit.sys, "executable", str(interpreter))
    monkeypatch.setattr(audit.budget, "DEFAULT_LEDGER", tmp_path / "budget.jsonl")
    monkeypatch.setattr(audit.budget, "DEFAULT_DECISIONS", tmp_path / "grant.json")
    for name in audit.OWN_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source " + name, encoding="utf-8")
    stage = tmp_path / "stage"
    seed = stage / "ce_seed0"
    seed.mkdir(parents=True)
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    (tokens / "train.u16").write_bytes(b"\0\0" * 1000)
    (tokens / "train.stories.jsonl").write_text('{"split":"train"}\n')
    audit.immutable(tokens / "manifest.json", {"fixture": True})
    data_files = {name: {k: v for k, v in audit.record(tokens / name).items() if k != "path"}
                  for name in ("train.u16", "train.stories.jsonl")}
    config = {"seed": 0, "context": 256, "microbatch": 8, "steps": 50000,
              "study": {"stage": "ce_pretrain", "token_manifest": audit.record(tokens / "manifest.json"),
                        "data_files": data_files}}
    audit.immutable(seed / "study_config.json", {"config": config, "config_sha256": hashlib.sha256(audit.canonical(config)).hexdigest()})
    audit.immutable(stage / "stage.json", {"source_records": [], "ce_parents": [{"seed": 0, "config": audit.record(seed / "study_config.json")}]})
    continuation = tmp_path / "continuation"
    continuation.mkdir()
    audit.immutable(continuation / "manifest.json", {
        "schema": "ce_parent_continuation_v1", "parent_stage": audit.record(stage / "stage.json"),
        "configs": [], "self": audit.record(tmp_path / audit.OWN_FILES[0]), "python": audit.record(interpreter)})
    snapshots = []
    for step in audit.STEPS:
        path = seed / f"state_step{step:07d}.pt"
        path.write_bytes(f"fixture checkpoint {step}".encode())
        snapshots.append(audit.record(path))
    audit.immutable(continuation / "seed0_admission.json", {
        "manifest": audit.record(continuation / "manifest.json"),
        "parent": {"seed": 0, "updates": 50000, "snapshots": snapshots}})
    (continuation / "status.json").write_text(json.dumps({"state": "training"}))
    verification = tmp_path / "verification"
    verification.mkdir()
    (verification / "pytest.log").write_text("fixture verification, not an actual test receipt")
    audit.immutable(verification / "receipt.json", {
        "kind": "representation_audit_verification_v1", "complete": True,
        "sources": [audit.record(tmp_path / p) for p in audit.OWN_FILES],
        "outputs": [audit.record(verification / "pytest.log")]})
    manifest = audit.prepare(stage / "stage.json", tmp_path / "audit", continuation / "manifest.json", verification / "receipt.json")
    return tmp_path, manifest, continuation


def rewrite(path, value):
    """Explicit fixture mutation, including a new sidecar to test semantic checks."""
    raw = audit.canonical(value) + b"\n"
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest())


def test_diagnostic_offsets_are_fixed_training_windows():
    offsets = audit.offsets(1000)
    assert len(offsets) == 8 and all(len(batch) == 8 for batch in offsets)
    assert offsets == audit.offsets(1000)
    assert all(0 <= start < 1000 - 256 - 1 for batch in offsets for start in batch)
    assert len(set(x for batch in offsets for x in batch)) > 50


@pytest.mark.parametrize("token_count", [0, 257, True, 258.0])
def test_invalid_training_lengths_rejected(token_count):
    with pytest.raises(ValueError):
        audit.offsets(token_count)


def test_preparation_binds_fixed_roles_and_archives_sources(prepared):
    root, path, _ = prepared
    manifest = audit.checked_manifest(path)
    assert manifest["sampling"]["split"] == "train"
    assert manifest["steps"] == [0, 5000, 25000, 50000]
    assert manifest["limits"] == {"threads": 1, "elapsed_seconds": 7200, "memory_bytes": 2 * 1024**3}
    assert manifest["device"] == "cpu"
    assert manifest["publication_admitted"] is False
    for source in manifest["sources"]:
        archived = path.parent / "source_archive" / Path(source["path"]).relative_to(root)
        assert archived.read_bytes() == Path(source["path"]).read_bytes()


@pytest.mark.parametrize("change", ["source", "checkpoint", "tokens", "sidecar", "offset", "test_split", "last_checkpoint", "resource"])
def test_drift_and_sealed_split_substitution_rejected(prepared, change):
    root, path, _ = prepared
    manifest = json.loads(path.read_bytes())
    if change == "source":
        Path(manifest["sources"][0]["path"]).write_text("source changed")
    elif change == "checkpoint":
        Path(manifest["snapshots"][1]["path"]).write_bytes(b"changed checkpoint")
    elif change == "tokens":
        Path(manifest["train_inputs"][0]["path"]).write_bytes(b"changed train stream")
    elif change == "sidecar":
        path.with_suffix(".sha256").write_text("0" * 64)
    else:
        if change == "offset": manifest["sample_offsets"][0][0] += 1
        if change == "test_split": manifest["train_inputs"][0]["path"] = str(root / "tokens/test.u16")
        if change == "last_checkpoint": manifest["snapshots"][0]["path"] = str(root / "stage/ce_seed0/last.pt")
        if change == "resource": manifest["limits"]["memory_bytes"] *= 2
        rewrite(path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.checked_manifest(path)


def test_prepare_never_overwrites(prepared):
    _, path, _ = prepared
    with pytest.raises(ValueError, match="never overwrites"):
        audit.prepare("does-not-matter", path.parent)


@pytest.mark.parametrize("role", ["checkpoint", "training_stream"])
def test_manifest_cannot_rebind_same_named_inputs_outside_lineage(prepared, role):
    root, path, _ = prepared
    manifest = json.loads(path.read_bytes())
    index = "snapshots" if role == "checkpoint" else "train_inputs"
    original = Path(manifest[index][0]["path"])
    substitute = root / "substitute" / original.name
    substitute.parent.mkdir()
    substitute.write_bytes(original.read_bytes())
    manifest[index][0] = audit.record(substitute)
    rewrite(path, manifest)
    with pytest.raises(ValueError):
        audit.checked_manifest(path)


def test_active_training_refuses_dispatch_without_attempt(prepared, monkeypatch):
    _, path, _ = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError("A worker must not start while CE is active")
    monkeypatch.setattr(audit.subprocess, "Popen", forbidden)
    observed = audit.run(path)
    assert observed["ready"] is False
    assert observed["controller_state"] == "training"
    assert not (path.parent / "attempt").exists()


@pytest.mark.parametrize("state", ["failed", "paused"])
def test_failed_or_paused_training_remains_pending(prepared, state):
    _, path, continuation = prepared
    (continuation / "status.json").write_text(json.dumps({"state": state}))
    observed = audit.readiness(audit.checked_manifest(path))
    assert observed["ready"] is False
    assert observed["controller_state"] == state


@pytest.mark.parametrize("observed", [
    {"completed_blocks": 49, "completed_ce_parents": 5, "open_reservations": [], "ledger_head": "fixture"},
    {"completed_blocks": 50, "completed_ce_parents": 4, "open_reservations": [], "ledger_head": "fixture"},
    {"completed_blocks": 50, "completed_ce_parents": 5, "open_reservations": ["still-open"], "ledger_head": "fixture"},
])
def test_completion_requires_all_receipts_and_no_open_reservation(prepared, monkeypatch, observed):
    _, path, continuation = prepared
    (continuation / "status.json").write_text('{"state":"completed"}')
    monkeypatch.setattr(audit.status, "execution", lambda *args: observed)
    with pytest.raises(ValueError, match="Incomplete CE/accounting"):
        audit.readiness(audit.checked_manifest(path))


def test_complete_receipt_inventory_admits_cpu_audit(prepared, monkeypatch):
    _, path, continuation = prepared
    (continuation / "status.json").write_text('{"state":"completed"}')
    monkeypatch.setattr(audit.status, "execution", lambda *args: {
        "completed_blocks": 50, "completed_ce_parents": 5, "open_reservations": [], "ledger_head": "fixture-chain-head"})
    assert audit.readiness(audit.checked_manifest(path)) == {"ready": True, "ledger_head": "fixture-chain-head"}


def fake_worker(monkeypatch, path, state="failed", timeout=False):
    """Dispatcher-only fake: no model, binary checkpoint, or real subprocess runs."""
    observed = {"kills": 0, "calls": 0}
    monkeypatch.setattr(audit, "readiness", lambda _: {"ready": True, "ledger_head": "fixture"})
    monkeypatch.setattr(audit, "contain_worker", lambda _: None)
    class Process:
        def __init__(self, command, **kwargs):
            observed["calls"] += 1
            observed["command"] = command
            observed["env"] = kwargs["env"]
            directory = path.parent / "attempt"
            audit.immutable(directory / "worker_result.json", {"state": state})
            self.waits = 0
        def wait(self, timeout=None):
            self.waits += 1
            if timeout is not None and self.waits == 1 and observed.get("timeout"):
                raise subprocess.TimeoutExpired("fixture-worker", timeout)
            return 0 if state == "completed" else 1
        def kill(self):
            observed["kills"] += 1
    observed["timeout"] = timeout
    monkeypatch.setattr(audit.subprocess, "Popen", Process)
    return observed


def test_failed_worker_produces_failed_receipt_and_forbids_retry(prepared, monkeypatch):
    _, path, _ = prepared
    observed = fake_worker(monkeypatch, path)
    receipt = audit.run(path)
    assert receipt["state"] == "failed" and receipt["complete"] is False
    assert receipt["checkpoint_and_input_integrity"] == "not_completed"
    assert receipt["confirmatory_claim_permitted"] is receipt["publication_admitted"] is False
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert observed["env"]["OMP_NUM_THREADS"] == "1"
    with pytest.raises(FileExistsError):
        audit.run(path)
    assert observed["calls"] == 1


def test_claimed_completion_without_measurements_is_failed(prepared, monkeypatch):
    _, path, _ = prepared
    fake_worker(monkeypatch, path, state="completed")
    receipt = audit.run(path)
    assert receipt["state"] == "failed" and not receipt["complete"]


def test_timeout_kills_worker_and_preserves_resource_failure(prepared, monkeypatch):
    _, path, _ = prepared
    observed = fake_worker(monkeypatch, path, timeout=True)
    receipt = audit.run(path)
    assert observed["kills"] == 1
    assert receipt["state"] == "resource_exhausted" and receipt["complete"] is False
    assert receipt["timeout"] is True
    assert (path.parent / "attempt/started.json").exists()
    assert (path.parent / "attempt/receipt.sha256").exists()


def test_launch_failure_has_durable_failed_receipt(prepared, monkeypatch):
    _, path, _ = prepared
    monkeypatch.setattr(audit, "readiness", lambda _: {"ready": True, "ledger_head": "fixture"})
    def failed_launch(*args, **kwargs):
        raise OSError("fixture launch denied")
    monkeypatch.setattr(audit.subprocess, "Popen", failed_launch)
    receipt = audit.run(path)
    assert receipt["state"] == "failed" and receipt["complete"] is False
    assert (path.parent / "attempt/receipt.json").exists()


def test_malformed_worker_result_has_durable_failed_receipt(prepared, monkeypatch):
    _, path, _ = prepared
    observed = fake_worker(monkeypatch, path)
    process = audit.subprocess.Popen
    def corrupted_result(*args, **kwargs):
        child = process(*args, **kwargs)
        (path.parent / "attempt/worker_result.json").write_bytes(b"not-json")
        return child
    monkeypatch.setattr(audit.subprocess, "Popen", corrupted_result)
    receipt = audit.run(path)
    assert observed["calls"] == 1
    assert receipt["state"] == "failed" and receipt["complete"] is False
    assert (path.parent / "attempt/receipt.json").exists()


def test_runtime_verification_must_be_complete_and_bind_sources(prepared):
    _, path, _ = prepared
    manifest = audit.status.document(audit.ROOT, path)
    verification = Path(manifest["verification"]["path"])
    for value in [{"kind": "unrelated", "complete": True, "sources": manifest["sources"], "outputs": []},
                  {"kind": "representation_audit_verification_v1", "complete": True, "sources": [], "outputs": []}]:
        rewrite(verification, value)
        manifest["verification"] = audit.record(verification)
        rewrite(path, manifest)
        with pytest.raises(ValueError, match="verification|Verification"):
            audit.checked_manifest(path)


def test_config_identity_uses_frozen_run_id_exclusion():
    assert audit.config_hash({"seed": 0, "run_id": "first"}) == audit.config_hash({"seed": 0, "run_id": "relocated"})
    assert audit.config_hash({"seed": 0}) != audit.config_hash({"seed": 1})
