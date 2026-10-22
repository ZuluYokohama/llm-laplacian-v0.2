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
    text = json.dumps({"complete": True, "rate": 1.25, "samples": [1.25]}) + "\n"
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
    report = claims.validate_manifest(manifest, tmp_path)
    assert report["integrity_passed"]
    assert "Host-side SHA-256" in report["limitation"]
    assert "not a Jovin isolated validation receipt" in report["limitation"]
    assert claims.file_hash(path) != claims.file_hash(path, "lf_normalized_utf8")
    manifest["artifacts"]["timing"].update(hash_convention="lf_normalized_utf8", sha256=claims.file_hash(path, "lf_normalized_utf8"))
    assert claims.validate_manifest(manifest, tmp_path)["integrity_passed"]


@pytest.mark.parametrize("mutation, reason", [
    ("file", "artifact hash mismatch: timing"),
    ("sidecar", "metrics sidecar mismatch: timing"),
    ("value", "evidence value mismatch: pilot/timing"),
    ("incomplete", "incomplete metrics: timing"),
    ("nonfinite", "nonfinite JSON value"),
    ("missing", "evidence must be an existing file within the repository: absent.json"),
    ("directory", "evidence must be an existing file within the repository: runs"),
    ("escape", "evidence must be an existing file within the repository: ../escape.json"),
])
def test_tampering_and_invalid_evidence_fail(tmp_path, mutation, reason):
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
    with pytest.raises(ValueError) as caught:
        claims.validate_manifest(manifest, tmp_path)
    assert str(caught.value) == reason


@pytest.mark.parametrize("mutation, reason", [
    ("expected_type", "evidence value mismatch: pilot/timing"),
    ("missing_equals", "evidence value mismatch: pilot/timing"),
    ("missing_hash", "artifact hash mismatch: timing"),
    ("missing_kind", "unknown artifact kind: timing"),
    ("unknown_kind", "unknown artifact kind: timing"),
    ("hash_convention", "unknown hash convention: unsupported"),
    ("duplicate_id", "missing or duplicate claim id"),
    ("missing_id", "missing or duplicate claim id"),
    ("missing_scope", "claim pilot missing scope"),
    ("scope_type", "claim pilot missing scope"),
    ("bool_selector", "invalid selector component: False"),
])
def test_claim_schema_and_selector_mutations_have_specific_failures(tmp_path, mutation, reason):
    _, manifest = fixture(tmp_path)
    artifact = manifest["artifacts"]["timing"]
    claim = manifest["claims"][0]
    binding = claim["evidence"][0]
    if mutation == "expected_type":
        binding.update(selector=["complete"], equals=1)  # True equals1 in Python; schema types must differ.
    elif mutation == "missing_equals":
        del binding["equals"]
    elif mutation == "missing_hash":
        del artifact["sha256"]
    elif mutation == "missing_kind":
        del artifact["kind"]
    elif mutation == "unknown_kind":
        artifact["kind"] = "unsupported"
    elif mutation == "hash_convention":
        artifact["hash_convention"] = "unsupported"
    elif mutation == "duplicate_id":
        manifest["claims"].append(copy.deepcopy(claim))
    elif mutation == "missing_id":
        del claim["id"]
    elif mutation == "missing_scope":
        del claim["scope"]
    elif mutation == "scope_type":
        claim["scope"] = 0
    else:
        binding["selector"] = ["samples", False]
    with pytest.raises(ValueError) as caught:
        claims.validate_manifest(manifest, tmp_path)
    assert str(caught.value) == reason


@pytest.mark.parametrize("path", ["C:/outside/metrics.json", "runs\\nested\\metrics.json"])
def test_absolute_and_backslash_evidence_paths_rejected(tmp_path, path):
    _, manifest = fixture(tmp_path)
    manifest["artifacts"]["timing"]["path"] = path
    with pytest.raises(ValueError) as caught:
        claims.validate_manifest(manifest, tmp_path)
    assert str(caught.value) == f"evidence path must be repository relative: {path}"


def test_metrics_sidecar_is_read_as_utf8(tmp_path, monkeypatch):
    _, manifest = fixture(tmp_path)
    original = Path.read_text
    seen = []
    def inspect_read(path, *args, **kwargs):
        if path.name == "metrics.json.sha256":
            seen.append(kwargs.get("encoding"))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", inspect_read)
    assert claims.validate_manifest(manifest, tmp_path)["integrity_passed"]
    assert seen == ["utf-8"]


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
    # The already observed claim is reviewed, so the planned row itself must fail.
    manifest["claims"][0]["human_review"] = {"approved": True, "reviewer": "fixture", "date": "2026-10-21"}
    with pytest.raises(ValueError) as caught:
        claims.validate_manifest(manifest, tmp_path, publication=True)
    assert str(caught.value) == "publication claim unresolved or not human-reviewed: future"
