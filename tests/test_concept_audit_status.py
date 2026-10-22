"""Audit receipt integration does not promote descriptive work to study success."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import concept_status as report


def write(path, value, sidecar=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    if sidecar:
        path.with_suffix(".sha256").write_text(report.digest(raw))
    return {"path": str(path), "bytes": len(raw), "sha256": report.digest(raw)}


@pytest.fixture
def receipt(tmp_path):
    row = {"kind": "representation_audit_receipt_v1", "state": "completed", "complete": True,
           "manifest": write(tmp_path / "manifest.json", {"frozen": True}),
           "sources": [write(tmp_path / "source.py", {"source": True})],
           "outputs": [write(tmp_path / "metrics.json", {"descriptive": True})],
           "scope": "host_cpu_training_only_diagnostics", "device": "cpu",
           "confirmatory_claim_permitted": False, "publication_admitted": False}
    path = tmp_path / "receipt.json"
    write(path, row, True)
    return tmp_path, path, row


def existing_report():
    return {"regions": [{"id": "CXM-GEOM", "status": "finite_checks"}, {"id": "CXM-EXEC", "status": "partial"},
                        {"id": "CXM-QUALITY", "status": "not_evaluated"}],
            "scientific_gates": {"RQ3": "not_evaluated"}, "confirmatory_claim_permitted": False, "publication_admitted": False}


def test_completed_audit_attaches_only_to_relevant_regions(receipt):
    root, path, _ = receipt
    before = existing_report()
    original = copy.deepcopy(before)
    result = report.attach_audits(before, report.inspect_audits(root, [path]))
    assert before == original
    assert result["regions"][0]["diagnostic_state"] == "completed_descriptive"
    assert result["regions"][1]["diagnostic_state"] == "completed_descriptive"
    assert result["regions"][2] == original["regions"][2]
    assert result["scientific_gates"] == original["scientific_gates"]
    assert all(a["status"] == b["status"] for a, b in zip(result["regions"], original["regions"]))
    assert result["publication_admitted"] is result["confirmatory_claim_permitted"] is False


@pytest.mark.parametrize("state", ["failed", "resource_exhausted"])
def test_failure_receipt_is_not_diagnostic_success(receipt, state):
    root, path, row = receipt
    row.update(state=state, complete=False)
    write(path, row, True)
    result = report.attach_audits(existing_report(), report.inspect_audits(root, [path]))
    assert result["regions"][0]["diagnostic_state"] == state
    assert result["audits"][0]["state"] == state
    assert result["publication_admitted"] is False


@pytest.mark.parametrize("field,value", [("device", "cuda:0"), ("scope", "held_out"),
    ("confirmatory_claim_permitted", True), ("publication_admitted", True), ("complete", 1),
    ("state", "passed"), ("complete", False)])
def test_unsupported_scope_claim_or_state_rejected(receipt, field, value):
    root, path, row = receipt
    row[field] = value
    write(path, row, True)
    with pytest.raises(ValueError): report.inspect_audits(root, [path])


@pytest.mark.parametrize("name", ["source.py", "metrics.json", "manifest.json", "receipt.sha256"])
def test_corruption_fails_closed(receipt, name):
    root, path, _ = receipt
    (root / name).write_text("changed")
    with pytest.raises(ValueError): report.inspect_audits(root, [path])


def test_missing_receipt_sidecar_rejected(receipt):
    root, path, _ = receipt
    path.with_suffix(".sha256").unlink()
    with pytest.raises(OSError): report.inspect_audits(root, [path])


def test_path_escape_rejected(receipt):
    root, path, row = receipt
    row["outputs"][0]["path"] = "../outside.json"
    write(path, row, True)
    with pytest.raises(ValueError, match="escapes"): report.inspect_audits(root, [path])


def test_receipt_aliases_and_duplicate_manifest_rejected(receipt):
    root, path, row = receipt
    with pytest.raises(ValueError, match="Duplicate audit receipt"):
        report.inspect_audits(root, [path, root / "directory" / ".." / "receipt.json"])
    other = root / "other_receipt.json"
    write(other, row, True)
    with pytest.raises(ValueError, match="Duplicate audit manifest"):
        report.inspect_audits(root, [path, other])
    copied_manifest = root / "copied_manifest.json"
    copied_manifest.write_bytes(Path(row["manifest"]["path"]).read_bytes())
    row["manifest"]["path"] = str(copied_manifest)
    write(other, row, True)
    with pytest.raises(ValueError, match="Duplicate audit manifest"):
        report.inspect_audits(root, [path, other])


@pytest.mark.parametrize("field", ["sources", "outputs"])
def test_completed_audit_needs_inventory(receipt, field):
    root, path, row = receipt
    row[field] = []
    write(path, row, True)
    with pytest.raises(ValueError): report.inspect_audits(root, [path])


def test_no_audits_preserves_old_behavior():
    before = existing_report()
    result = report.attach_audits(before, [])
    assert result == {**before, "audits": []}


def test_multiple_receipts_keep_failures_visible(receipt):
    root, path, row = receipt
    other = root / "second_receipt.json"
    row.update(state="failed", complete=False, manifest=write(root / "other_manifest.json", {}))
    write(other, row, True)
    result = report.attach_audits(existing_report(), report.inspect_audits(root, [path, other]))
    assert [r["diagnostic_state"] for r in result["audits"]] == ["completed_descriptive", "failed"]
    assert result["regions"][0]["diagnostic_state"] == "failed"


def test_markdown_labels_audit_scope(receipt):
    root, path, _ = receipt
    data = existing_report()
    data.update(utc="now", execution={"completed_ce_parents": 5, "completed_blocks": 50, "controller_reported": {"state": "completed"}})
    for region in data["regions"]:
        region.update(name=region["id"], next_gate="separate study gate")
    text = report.markdown(report.attach_audits(data, report.inspect_audits(root, [path])))
    assert "completed_descriptive" in text and "host_cpu_training_only_diagnostics" in text
    assert "scientific gates remain unchanged" in text
