import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_claims", ROOT / "scripts/validate_claims.py")
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)


def fixture(tmp_path):
    path = tmp_path / "runs" / "nested" / "metrics.json"
    path.parent.mkdir(parents=True)
    text = json.dumps({"complete": True, "rate": 1.25}) + "\n"
    path.write_bytes(text.replace("\n", "\r\n").encode())
    path.with_name("metrics.json.sha256").write_text(hashlib.sha256(text.encode()).hexdigest())
    manifest = {"schema_version": 1, "artifacts": {"timing": {
        "path": "runs/nested/metrics.json", "kind": "metrics",
        "sha256": claims.file_hash(path), "hash_convention": "raw_bytes",
    }}, "claims": [{"id": "pilot", "status": "observed", "statement": "Pilot rate was 1.25.",
                     "scope": "A fixture", "limitations": "Not a real experiment",
                     "evidence": [{"artifact": "timing", "selector": ["rate"], "equals": 1.25}]}]}
    return path, manifest


def test_nested_metrics_and_explicit_hash_conventions(tmp_path):
    path, manifest = fixture(tmp_path)
    assert claims.validate_manifest(manifest, tmp_path)["integrity_passed"]
    assert claims.file_hash(path) != claims.file_hash(path, "lf_normalized_utf8")


@pytest.mark.parametrize("mutation", ["file", "sidecar", "value", "incomplete", "nonfinite", "missing", "directory", "escape"])
def test_tampering_and_invalid_evidence_fail(tmp_path, mutation):
    path, manifest = fixture(tmp_path)
    if mutation == "file":
        path.write_text('{"complete": true, "rate": 9.9}')
    elif mutation == "sidecar":
        path.with_name("metrics.json.sha256").write_text("0" * 64)
    elif mutation == "value":
        manifest["claims"][0]["evidence"][0]["equals"] = 9.9
    elif mutation in ("incomplete", "nonfinite"):
        path.write_text(json.dumps({"complete": mutation != "incomplete", "rate": float("nan") if mutation == "nonfinite" else 1.25}))
        manifest["artifacts"]["timing"]["sha256"] = claims.file_hash(path)
    else:
        manifest["artifacts"]["timing"]["path"] = {"missing": "absent.json", "directory": "runs", "escape": "../escape.json"}[mutation]
    with pytest.raises((ValueError, OSError)):
        claims.validate_manifest(manifest, tmp_path)


def test_publication_requires_review_and_exact_manuscript(tmp_path):
    path, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="human-reviewed"):
        claims.validate_manifest(manifest, tmp_path, publication=True)
    manuscript = tmp_path / "main.tex"
    manuscript.write_text("A reviewed test manuscript")
    manifest["claims"][0]["human_review"] = {"approved": True, "reviewer": "test fixture", "date": "2026-10-21"}
    manifest["manuscript"] = {"path": "main.tex", "sha256": claims.file_hash(manuscript)}
    assert claims.validate_manifest(manifest, tmp_path, publication=True)["integrity_passed"]
    manuscript.write_text("An unreviewed change")
    with pytest.raises(ValueError, match="exact manuscript"):
        claims.validate_manifest(manifest, tmp_path, publication=True)


def test_planned_claims_are_not_results(tmp_path):
    _, manifest = fixture(tmp_path)
    future = copy.deepcopy(manifest["claims"][0])
    future.update(id="future", status="planned", evidence=[])
    manifest["claims"].append(future)
    assert claims.validate_manifest(manifest, tmp_path)["claim_count"] == 2
    with pytest.raises(ValueError):
        claims.validate_manifest(manifest, tmp_path, publication=True)
