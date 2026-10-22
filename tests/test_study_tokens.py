import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from laplacian import study_data, study_tokens

REAL_DEPENDENCY_IDENTITY = study_tokens.dependency_identity
REAL_TOKENIZER_JSON = Path(os.environ.get("LAPLACIAN_TOKENIZER_JSON", study_tokens.DEFAULT_TOKENIZER_JSON))


class TinyTokenizer:
    padding = truncation = None

    def __init__(self, callback=None, ids=None):
        self.callback, self.ids, self.seen = callback, ids, []

    def get_vocab_size(self):
        return 50257

    def token_to_id(self, token):
        assert token == "<|endoftext|>"
        return 50256

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.seen.append(text)
        if self.callback:
            callback, self.callback = self.callback, None
            callback()
        return SimpleNamespace(ids=self.ids if self.ids is not None else [len(text) % 100 + 1, ord(text[-1]) % 100 + 1])


def write_arrow(path, texts):
    schema = pa.schema([("text", pa.string())])
    with path.open("wb") as handle, ipc.new_stream(handle, schema) as writer:
        for start in range(0, len(texts), 2):
            writer.write_batch(pa.record_batch([texts[start:start + 2]], schema=schema))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    groups = {split: [] for split in study_tokens.SPLITS}
    index = 0
    while any(len(texts) < 2 for texts in groups.values()):
        text = f"CPU tokenization fixture story {index}."
        split = study_data.assigned_split(study_data.story_identity(text), 23)
        if len(groups[split]) < 2:
            groups[split].append(text)
        index += 1
    paths, selected = [], []
    for source_index in range(2):
        texts = [groups[split][source_index] for split in study_tokens.SPLITS]
        selected.extend(texts)
        if source_index == 1:
            texts.extend([groups["train"][0].upper(), " \n"])
        path = tmp_path / f"source{source_index}.arrow"
        write_arrow(path, texts)
        paths.append(path)
    partitions = tmp_path / "partitions"
    study_data.prepare(paths, partitions, seed=23)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"synthetic tokenizer fixture; never a production tokenizer")
    dependencies = {"packages": {}, "files": []}
    for name in ("pyarrow", "tokenizers"):
        dependency_path = tmp_path / f"{name}_dependency_fixture.txt"
        dependency_path.write_bytes(f"Synthetic {name} dependency for CPU unit tests".encode())
        record = study_tokens.file_record(dependency_path)
        dependencies["files"].append(record)
        dependencies["packages"][name] = {"version": "synthetic-fixture",
                                         "files": {dependency_path.name: {key: record[key] for key in ("bytes", "sha256")}}}
    dependencies["sha256"] = hashlib.sha256(study_tokens.canonical(dependencies["packages"])).hexdigest()
    monkeypatch.setattr(study_tokens, "dependency_identity", lambda: dependencies)
    return SimpleNamespace(partitions=partitions, output=tmp_path / "tokens", tokenizer=tokenizer,
                           sources=paths, selected=selected, dependencies=dependencies)


def prepare(fixture, tokenizer=None):
    return study_tokens.prepare(fixture.partitions, fixture.output, fixture.tokenizer,
                                tokenizer=tokenizer or TinyTokenizer())


def update_partition(fixture, split, transform):
    path = fixture.partitions / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = fixture.partitions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = study_data.sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))


def assert_failed(fixture):
    assert json.loads((fixture.output / "manifest.json").read_text())["complete"] is False
    assert json.loads((fixture.output / "failure.json").read_text())["complete"] is False
    assert not (fixture.output / "manifest.complete.tmp").exists()


def test_streamed_selection_offsets_eos_and_raw_hashes(fixture):
    tokenizer = TinyTokenizer()
    manifest = prepare(fixture, tokenizer)
    assert manifest["complete"] is True and manifest["production_ready"] is False
    assert manifest["tokenizer"]["mode"] == "injected_test_double"
    assert tokenizer.seen == fixture.selected  # Global Arrow order, not split order.
    assert manifest["source_rows"] == 10
    for split in study_tokens.SPLITS:
        data = (fixture.output / f"{split}.u16").read_bytes()
        tokens = list(struct.unpack(f"<{len(data) // 2}H", data))
        records = [json.loads(line) for line in (fixture.output / f"{split}.stories.jsonl").read_text().splitlines()]
        selected = [json.loads(line) for line in (fixture.partitions / f"{split}.jsonl").read_text().splitlines()]
        offset = 0
        for row, source in zip(records, selected, strict=True):
            assert {key: row[key] for key in source} == source
            assert row["offset"] == offset and row["length"] == 3
            story_tokens = tokens[offset:offset + row["length"]]
            assert story_tokens[-1] == 50256
            assert len(story_tokens) == row["length"]
            offset += row["length"]
        assert offset == manifest["token_counts"][split] == 6
        for filename in (f"{split}.u16", f"{split}.stories.jsonl"):
            assert manifest["files"][filename]["sha256"] == study_data.sha256_file(fixture.output / filename)


@pytest.mark.parametrize("ids", [[65536], [-1], [50257], [True], [1.5], []])
def test_invalid_or_overflow_token_ids_are_rejected_without_completion(fixture, ids):
    with pytest.raises(ValueError, match="overflow token"):
        prepare(fixture, TinyTokenizer(ids=ids))
    assert_failed(fixture)


@pytest.mark.parametrize("ids", [[50256], [1, 50256, 2], [1, 50256]])
def test_in_story_eos_is_rejected_without_promoting_a_bundle(fixture, ids):
    with pytest.raises(ValueError, match="in-story EOS token is forbidden"):
        prepare(fixture, TinyTokenizer(ids=ids))
    assert_failed(fixture)


@pytest.mark.parametrize("target", ["source", "jsonl"])
def test_preexisting_input_tampering_is_rejected(fixture, target):
    path = fixture.sources[0] if target == "source" else fixture.partitions / "train.jsonl"
    with path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="hash"):
        prepare(fixture)
    assert_failed(fixture)


def test_missing_referenced_arrow_row_is_not_silently_skipped(fixture):
    update_partition(fixture, "test", lambda rows: rows[-1].update(row_index=1000000))
    with pytest.raises(ValueError, match="missing Arrow row"):
        prepare(fixture)
    assert_failed(fixture)


@pytest.mark.parametrize("field", ["text_sha256", "story_id"])
def test_story_content_and_normalized_identity_are_checked_before_encode(fixture, field):
    def mutate(rows):
        if field == "text_sha256":
            rows[0][field] = "0" * 64
        else:
            index = 0
            while True:
                identity = study_data.story_identity(f"Another unrelated story {index}")
                if study_data.assigned_split(identity, 23) == "train":
                    rows[0][field] = identity
                    break
                index += 1
    update_partition(fixture, "train", mutate)
    tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="hash mismatch|identity mismatch"):
        prepare(fixture, tokenizer)
    assert tokenizer.seen == []
    assert_failed(fixture)


def test_unsorted_or_repeated_row_references_fail(fixture):
    update_partition(fixture, "train", lambda rows: rows.reverse())
    with pytest.raises(ValueError, match="strictly sorted"):
        prepare(fixture)
    assert_failed(fixture)


@pytest.mark.parametrize("target", ["source", "partition", "tokenizer", "code", "dependency"])
def test_mutations_during_encoding_never_promote_complete(fixture, monkeypatch, tmp_path, target):
    paths = {"source": fixture.sources[-1], "partition": fixture.partitions / "manifest.json", "tokenizer": fixture.tokenizer}
    if target == "code":
        paths[target] = tmp_path / "tokenizer_code_fixture.py"
        paths[target].write_bytes(b"# Code identity fixture\n")
        monkeypatch.setattr(study_tokens, "__file__", str(paths[target]))
    if target == "dependency":
        paths[target] = tmp_path / "dependency_fixture.bin"
        paths[target].write_bytes(b"dependency before tokenization")
        fixture.dependencies["files"].append(study_tokens.file_record(paths[target]))
    def mutate():
        with paths[target].open("ab") as handle:
            handle.write(b" ")
    with pytest.raises(RuntimeError, match="changed while tokenizing"):
        prepare(fixture, TinyTokenizer(callback=mutate))
    assert_failed(fixture)


def test_successful_and_failed_output_directories_are_immutable(fixture):
    prepare(fixture)
    before = {path.name: path.read_bytes() for path in fixture.output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(fixture)
    assert before == {path.name: path.read_bytes() for path in fixture.output.iterdir()}
    fixture.output = fixture.output.with_name("failed")
    with pytest.raises(ValueError):
        prepare(fixture, TinyTokenizer(ids=[65536]))
    before = {path.name: path.read_bytes() for path in fixture.output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(fixture)
    assert before == {path.name: path.read_bytes() for path in fixture.output.iterdir()}


def test_pinned_loader_rejects_another_local_tokenizer(fixture):
    with pytest.raises(ValueError, match="pinned GPT-2"):
        study_tokens.load_tokenizer(fixture.tokenizer)


def production_fixture(fixture, monkeypatch):
    manifest = prepare(fixture)
    # Explicit synthetic fixture for consumer validation; no real dataset is promoted.
    monkeypatch.setattr(study_tokens, "TOKENIZER_SHA256", manifest["tokenizer"]["file"]["sha256"])
    manifest["production_ready"] = True
    manifest["tokenizer"].update(mode="pinned_gpt2", revision=study_tokens.TOKENIZER_REVISION)
    path = fixture.output / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_consumer_validator_rejects_injected_tokenizer_by_default(fixture):
    prepare(fixture)
    with pytest.raises(ValueError, match="production token bundle"):
        study_tokens.validate_bundle(fixture.output / "manifest.json")


@pytest.mark.parametrize("mutation,message", [
    ("historical_policy", "complete production token bundle"),
    ("interior_eos_policy", "pinned GPT-2 policy"),
])
def test_consumer_requires_explicit_single_eos_version(fixture, monkeypatch, mutation, message):
    path = production_fixture(fixture, monkeypatch)
    manifest = json.loads(path.read_text())
    if mutation == "historical_policy":
        manifest["policy"] = "gpt2_story_eos_u16le_v1"
    else:
        manifest["tokenizer"].pop("interior_eos")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        study_tokens.validate_bundle(path)


@pytest.mark.parametrize("tamper", [None, "tokens", "offsets", "partition", "tokenizer"])
def test_consumer_validator_binds_bundle_and_upstream_inputs(fixture, monkeypatch, tamper):
    path = production_fixture(fixture, monkeypatch)
    if tamper is None:
        assert study_tokens.validate_bundle(path)["token_counts"]["train"] == 6
        return
    target = {"tokens": fixture.output / "train.u16", "offsets": fixture.output / "train.stories.jsonl",
              "partition": fixture.partitions / "manifest.json", "tokenizer": fixture.tokenizer}[tamper]
    with target.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises((ValueError, RuntimeError), match="mismatch|changed"):
        study_tokens.validate_bundle(path)


@pytest.mark.parametrize("field", ["code", "dependencies"])
def test_consumer_validator_requires_nonempty_provenance(fixture, monkeypatch, field):
    path = production_fixture(fixture, monkeypatch)
    manifest = json.loads(path.read_text())
    if field == "code":
        manifest[field] = []
    else:
        manifest[field] = {"packages": {}, "files": [], "sha256": hashlib.sha256(b"{}").hexdigest()}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance|source-code"):
        study_tokens.validate_bundle(path)


@pytest.mark.skipif(not os.environ.get("LAPLACIAN_TOKENIZER_JSON") and not REAL_TOKENIZER_JSON.is_file(),
                    reason="pinned local GPT-2 snapshot is unavailable")
def test_tiny_real_pinned_cpu_tokenizer_and_dependency_inventory(fixture, monkeypatch):
    monkeypatch.setattr(study_tokens, "dependency_identity", REAL_DEPENDENCY_IDENTITY)
    manifest = study_tokens.prepare(fixture.partitions, fixture.output, tokenizer_json=REAL_TOKENIZER_JSON)
    assert manifest["production_ready"] is True
    assert manifest["tokenizer"]["file"]["sha256"] == study_tokens.TOKENIZER_SHA256
    assert manifest["token_counts"]["train"] > manifest["story_counts"]["train"] * 2
    assert study_tokens.validate_bundle(fixture.output / "manifest.json") == manifest


@pytest.mark.skipif(not REAL_TOKENIZER_JSON.is_file(), reason="pinned local GPT-2 snapshot is unavailable")
def test_literal_special_token_is_rejected_with_real_gpt2(fixture):
    texts = list(fixture.selected[:4])
    texts[0] = "A story containing literal <|endoftext|> text."
    write_arrow(fixture.sources[0], texts)
    partitions = fixture.partitions.with_name("literal_partitions")
    study_data.prepare(fixture.sources, partitions, seed=23)
    with pytest.raises(ValueError, match="in-story EOS token is forbidden"):
        study_tokens.prepare(partitions, fixture.output, tokenizer_json=REAL_TOKENIZER_JSON)
    assert_failed(fixture)
