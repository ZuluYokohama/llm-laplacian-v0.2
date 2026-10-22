import json

import pytest

from laplacian.study_data import assigned_split, story_identity, write_partitions, sha256_file
from laplacian import study_data


def test_duplicate_normalization_and_deterministic_assignment():
    first = story_identity("A  FOX\nfound a boat.")
    assert first == story_identity("a fox found a boat.")
    assert assigned_split(first, 17) == assigned_split(story_identity("a fox found a boat."), 17)
    with pytest.raises(ValueError):
        story_identity(" \n")


def test_story_groups_never_overlap_and_original_rows_are_traceable(tmp_path):
    rows = [(0, i, f"Story number {i}.") for i in range(10000)]
    rows += [(1, 0, "STORY NUMBER 7."), (1, 1, " ")]
    output = tmp_path / "partitions"
    result = write_partitions(rows, output, [{"path": "test fixture"}], seed=10)
    assert result["unique_stories"] == 10000
    assert result["duplicates_removed"] == 1
    assert result["empty_rows_removed"] == 1
    assert result["complete"] is False  # No source stability validation yet.
    found = set()
    for split, count in result["story_counts"].items():
        path = output / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert count == len(records) > 0
        assert result["files"][path.name] == sha256_file(path)
        identities = {row["story_id"] for row in records}
        assert not found.intersection(identities)
        found.update(identities)
        for row in records:
            assert row["source_index"] == 0
            assert story_identity(rows[row["row_index"]][2]) == row["story_id"]
    assert len(found) == 10000
    with pytest.raises(FileExistsError):
        write_partitions(rows, output, [])


def test_prepare_refuses_changed_source_and_retains_incomplete_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source.arrow"
    source.write_bytes(b"original")
    def rows(paths):
        yield 0, 0, "First story"
        source.write_bytes(b"changed during read")
    monkeypatch.setattr(study_data, "arrow_rows", rows)
    with pytest.raises(RuntimeError, match="source changed"):
        study_data.prepare([source], tmp_path / "result")
    assert json.loads((tmp_path / "result/manifest.json").read_text())["complete"] is False


def test_prepare_records_reader_failure_and_rejects_empty_split(tmp_path, monkeypatch):
    source = tmp_path / "source.arrow"
    source.write_bytes(b"fixture")
    def broken(paths):
        yield 0, 0, "First story"
        raise ValueError("malformed Arrow stream")
    monkeypatch.setattr(study_data, "arrow_rows", broken)
    with pytest.raises(ValueError, match="malformed"):
        study_data.prepare([source], tmp_path / "failure")
    assert json.loads((tmp_path / "failure/failure.json").read_text())["complete"] is False
    monkeypatch.setattr(study_data, "arrow_rows", lambda paths: [(0, 0, "Only story")])
    with pytest.raises(RuntimeError, match="empty study partition"):
        study_data.prepare([source], tmp_path / "empty")
    assert json.loads((tmp_path / "empty/manifest.json").read_text())["complete"] is False
