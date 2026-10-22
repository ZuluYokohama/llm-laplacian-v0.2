import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/audit_study_tokens.py"
_spec = importlib.util.spec_from_file_location("token_audit", _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def fixture_bundle(tmp_path):
    folder, parts = tmp_path / "tokens", tmp_path / "partitions"
    folder.mkdir()
    parts.mkdir()
    manifest = {"partition_manifest": {"path": str(parts / "manifest.json")},
                "tokenizer": {"vocab_size": 10, "eos_id": 9}, "story_counts": {}, "token_counts": {},
                "partition_seed": 123, "sources": [{}]}
    for i, split in enumerate(audit.SPLITS):
        identity = next(identity for k in range(100000)
                        if audit.assigned_split(identity := hashlib.sha256(f"{i}-{k}".encode()).hexdigest(), 123) == split)
        row = {"story_id": identity, "source_index": 0, "row_index": i, "text_sha256": "a" * 64}
        (parts / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (folder / f"{split}.stories.jsonl").write_text(json.dumps({**row, "offset": 0, "length": 3}) + "\n", encoding="utf-8")
        np.array([1, 2, 9], dtype="<u2").tofile(folder / f"{split}.u16")
        manifest["story_counts"][split] = 1
        manifest["token_counts"][split] = 3
    return folder, parts, manifest


def test_every_split_audited(tmp_path):
    folder, _, manifest = fixture_bundle(tmp_path)
    result = audit.audit_structure(folder / "manifest.json", manifest)
    assert result["disjoint_unique_story_ids"] == 4
    assert all(row["tokens"] == 3 for row in result["splits"].values())
    assert all(row["no_interior_eos"] for row in result["splits"].values())


@pytest.mark.parametrize("field,value,match", [
    ("offset", 1, "offset"), ("length", 4, "length"), ("length", True, "length"),
    ("text_sha256", "b" * 64, "partition row"),
    ("source_index", False, "partition row"), ("row_index", 0.0, "partition row"),
])
def test_changed_story_record_rejected(tmp_path, field, value, match):
    folder, _, manifest = fixture_bundle(tmp_path)
    path = folder / "train.stories.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=match):
        audit.audit_structure(folder / "manifest.json", manifest)


@pytest.mark.parametrize("tokens,match", [([1, 2, 8], "EOS"), ([1, 10, 9], "vocabulary"),
                                         ([1, 2, 9, 9], "accounting"), ([1, 9, 9], "In-story EOS"),
                                         ([9, 2, 9], "In-story EOS")])
def test_token_corruption_rejected(tmp_path, tokens, match):
    folder, _, manifest = fixture_bundle(tmp_path)
    np.array(tokens, dtype="<u2").tofile(folder / "train.u16")
    with pytest.raises(ValueError, match=match):
        audit.audit_structure(folder / "manifest.json", manifest)


def test_duplicate_identity_rejected_even_if_rows_match(tmp_path):
    folder, parts, manifest = fixture_bundle(tmp_path)
    identity = json.loads((parts / "train.jsonl").read_text())["story_id"]
    for path in [parts / "test.jsonl", folder / "test.stories.jsonl"]:
        row = json.loads(path.read_text())
        row["story_id"] = identity
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        audit.audit_structure(folder / "manifest.json", manifest)


def test_missing_partition_row_rejected(tmp_path):
    folder, parts, manifest = fixture_bundle(tmp_path)
    (parts / "calibration.jsonl").write_text("")
    with pytest.raises(ValueError, match="row count"):
        audit.audit_structure(folder / "manifest.json", manifest)


@pytest.mark.parametrize("field", ["source_index", "row_index", "text_sha256"])
def test_rehashed_missing_reference_fields_do_not_pass(tmp_path, field):
    folder, parts, manifest = fixture_bundle(tmp_path)
    for path in [parts / "train.jsonl", folder / "train.stories.jsonl"]:
        row = json.loads(path.read_text())
        row.pop(field)
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="schema"):
        audit.audit_structure(folder / "manifest.json", manifest)


def test_multiple_stories_contiguous_and_gap_rejected(tmp_path):
    folder, parts, manifest = fixture_bundle(tmp_path)
    identity = next(identity for k in range(10000)
                    if audit.assigned_split(identity := hashlib.sha256(f"second-{k}".encode()).hexdigest(), 123) == "train")
    row = {"story_id": identity, "source_index": 0, "row_index": 10, "text_sha256": "b" * 64}
    with (parts / "train.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    indexed = folder / "train.stories.jsonl"
    with indexed.open("a") as handle:
        handle.write(json.dumps({**row, "offset": 3, "length": 3}) + "\n")
    np.array([1, 2, 9, 3, 4, 9], dtype="<u2").tofile(folder / "train.u16")
    manifest["story_counts"]["train"], manifest["token_counts"]["train"] = 2, 6
    assert audit.audit_structure(folder / "manifest.json", manifest)["splits"]["train"]["stories"] == 2
    lines = indexed.read_text().splitlines()
    second = json.loads(lines[1])
    second["offset"] = 4
    indexed.write_text(lines[0] + "\n" + json.dumps(second) + "\n")
    with pytest.raises(ValueError, match="offset"):
        audit.audit_structure(folder / "manifest.json", manifest)
