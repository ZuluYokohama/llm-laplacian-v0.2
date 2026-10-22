"""CPU fixtures exercise reporting without starting training or loading a model."""
import copy
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import concept_status as report


def write(path, value, sidecar=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    if sidecar:
        path.with_suffix(".sha256").write_text(report.digest(raw))
    return {"path": str(path), "sha256": report.digest(raw)}


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(report.budget, "ROOT", tmp_path)
    parent, continuation = tmp_path / "stage", tmp_path / "continuation"
    ledger = tmp_path / "budget.jsonl"
    ledger.touch()
    grant = tmp_path / "grant.json"
    write(grant, {"recommended_core_gpu_hours": {"approved": True, "hard_cap": 181, "ledger": str(ledger)}})
    evidence = write(tmp_path / "evidence.json", {"complete": True})
    original = [{"seed": 0, "from_step": i * 5000, "to_step": (i + 1) * 5000} for i in range(10)]
    jobs = [{"number": n, "seed": n // 10, "from_step": n % 10 * 5000, "to_step": (n % 10 + 1) * 5000} for n in range(10, 50)]
    p = write(parent / "stage.json", {"segments": original, "source_records": [], "ce_parents": []}, True)
    source = write(tmp_path / "source.json", {"fixed": True})
    write(continuation / "manifest.json", {"schema": "ce_parent_continuation_v1", "parent_stage": p,
          "configs": [], "self": source, "python": source, "jobs": jobs, "ceiling_seconds": 108000}, True)
    mapping = {"schema_version": 1, "sources": [], "artifacts": {"fixture": {**evidence, "checks": [{"selector": ["complete"], "equals": True}]}},
               "regions": [{"id": "quality", "name": "Quality", "status": "not_evaluated", "implementation": [], "evidence": [], "next_gate": "Held-out comparison"}],
               "scientific_gates": {"RQ3": "not_evaluated"}, "execution": {"continuation_manifest": str(continuation / "manifest.json"), "ledger": str(ledger), "grant": str(grant)}}
    records, markers = [], []

    def advance(count):
        for n in range(count):
            directory = parent if n < 10 else continuation
            job = original[n] if n < 10 else jobs[n - 10]
            rid = f"{directory.name}-b{n:03d}"
            (tmp_path / "runs" / rid).mkdir(parents=True)
            report.budget.append_event(ledger, records, "reserved", run_id=rid, reserve_seconds=3600)
            settlement = report.budget.append_event(ledger, records, "finished", run_id=rid, charge_seconds=1, status="completed")
            receipt = write(tmp_path / f"receipt{n}.json", {"complete": n % 10 == 9})
            marker = write(directory / f"block_{n:03d}.json", {"job": job, "budget_run_id": rid, "charge_seconds": 1,
                          "settlement_sha256": settlement["sha256"], "training_receipt": receipt}, n >= 10)
            markers.append(marker)
            if n == 9:
                write(continuation / "seed0_admission.json", {"parent": {"updates": 50000, "block_receipts": markers[:10]}}, True)
            if n >= 10 and n % 10 == 9:
                write(continuation / f"parent_seed{n // 10}_complete.json", {"seed": n // 10, "updates": 50000, "block_receipts": markers[n-9:n+1]}, True)
        if count < 50:
            rid = f"{continuation.name}-b{count:03d}"
            (tmp_path / "runs" / rid).mkdir(parents=True)
            report.budget.append_event(ledger, records, "reserved", run_id=rid, reserve_seconds=3600)
            write(continuation / "status.json", {"state": "training", "budget_run_id": rid, "job": jobs[count - 10]})
        else:
            write(continuation / "status.json", {"state": "completed"})
        return write(tmp_path / "map.json", mapping)

    return tmp_path, mapping, advance, continuation, ledger


def test_running_snapshot_preserves_inputs_and_does_not_admit_quality(study):
    root, mapping, advance, _, _ = study
    advance(10)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    observed = report.snapshot(root, root / "map.json")
    assert observed["execution"]["completed_ce_parents"] == 1
    assert observed["execution"]["completed_blocks"] == 10
    assert observed["execution"]["continuation_charged_or_reserved_hours"] == 1
    assert observed["scientific_gates"]["RQ3"] == "not_evaluated"
    assert observed["publication_admitted"] is False
    assert before == {p: p.read_bytes() for p in before}


def test_all_parents_complete_still_does_not_admit_quality(study):
    root, _, advance, _, _ = study
    advance(50)
    observed = report.snapshot(root, root / "map.json")
    assert observed["execution"]["completed_ce_parents"] == 5
    assert observed["execution"]["open_reservations"] == []
    assert observed["scientific_gates"]["RQ3"] == "not_evaluated"
    assert observed["publication_admitted"] is False


@pytest.mark.parametrize("state", ["paused", "failed"])
def test_controller_state_and_pause_are_reported_not_success(study, state):
    root, _, advance, folder, _ = study
    advance(10)
    write(folder / "status.json", {"state": state})
    (folder / "PAUSE").touch()
    observed = report.snapshot(root, root / "map.json")
    assert observed["execution"]["controller_reported"]["state"] == state
    assert observed["execution"]["pause_requested"]


@pytest.mark.parametrize("mutation", ["receipt", "source", "sidecar", "ledger", "active", "completed", "budget"])
def test_inconsistent_evidence_fails_closed(study, mutation):
    root, _, advance, folder, ledger = study
    advance(10)
    if mutation == "receipt":
        (root / "receipt3.json").write_text('{}')
    elif mutation == "source":
        (root / "source.json").write_text('{}')
    elif mutation == "sidecar":
        (folder / "manifest.sha256").write_text('0' * 64)
    elif mutation == "ledger":
        ledger.write_bytes(ledger.read_bytes().replace(b'"charge_seconds":1', b'"charge_seconds":2', 1))
    elif mutation == "active":
        write(folder / "status.json", {"state": "training", "budget_run_id": "unknown", "job": {}})
    elif mutation == "completed":
        write(folder / "status.json", {"state": "completed"})
    else:
        write(root / "grant.json", {"recommended_core_gpu_hours": {"approved": True, "hard_cap": .001, "ledger": str(ledger)}})
    with pytest.raises((ValueError, report.budget.BudgetError)):
        report.snapshot(root, root / "map.json")


def test_map_types_references_and_paths_are_strict(study):
    root, mapping, _, _, _ = study
    for alteration in ["bool", "hash", "state", "duplicate", "reference", "escape"]:
        m = copy.deepcopy(mapping)
        if alteration == "bool": m["artifacts"]["fixture"]["checks"][0]["equals"] = 1
        if alteration == "hash": m["artifacts"]["fixture"]["hash_convention"] = "normalized"
        if alteration == "state": m["regions"][0]["status"] = "proved"
        if alteration == "duplicate": m["regions"].append(m["regions"][0])
        if alteration == "reference": m["regions"][0]["evidence"] = ["invented"]
        if alteration == "escape": m["artifacts"]["fixture"]["path"] = "../outside.json"
        with pytest.raises(ValueError): report.inspect_map(root, m)


def test_changed_pdf_bytes_rejected(tmp_path):
    source = write(tmp_path / "source.pdf", {"source": True})
    mapping = {"schema_version": 1, "sources": [{**source, "id": "pdf", "pages": 1}], "artifacts": {}, "regions": []}
    (tmp_path / "source.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed bound file"):
        report.inspect_map(tmp_path, mapping)


def test_output_is_immutable_and_hash_bound(study):
    root, _, advance, _, _ = study
    advance(10)
    observed = report.snapshot(root, root / "map.json")
    output = root / "snapshot"
    report.save(output, observed)
    assert (output / "status.json.sha256").read_text().strip() == report.digest((output / "status.json").read_bytes())
    with pytest.raises(FileExistsError): report.save(output, observed)


def test_unstable_view_is_rejected(study, monkeypatch):
    root, _, advance, folder, _ = study
    advance(10)
    count = []
    def changing(*args):
        count.append(1)
        write(folder / "status.json", {"state": "transition", "revision": len(count)})
        return {}
    monkeypatch.setattr(report, "execution", changing)
    with pytest.raises(ValueError, match="all three"):
        report.snapshot(root, root / "map.json")


def test_source_page_and_identity_must_exist(study):
    root, mapping, _, _, _ = study
    source = write(root / "pdf.json", {})
    mapping["sources"] = [{**source, "id": "pdf", "pages": 1}]
    for reference in [{"document": "missing", "pages": [1]}, {"document": "pdf", "pages": [2]}]:
        mapping["regions"][0]["source_refs"] = [reference]
        with pytest.raises(ValueError): report.inspect_map(root, mapping)
