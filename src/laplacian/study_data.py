"""Deterministic story-level partitions for the new publication study.

This CPU-only stage indexes existing local Arrow sources. It does not tokenize,
train, or claim that the source corpus was historically unseen.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import unicodedata

POLICY = "story_content_hash_partition_v1"
SPLIT_WEIGHTS = {"train": 9800, "development": 100, "calibration": 50, "test": 50}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def story_identity(text):
    if not isinstance(text, str):
        raise ValueError("story text must be a string")
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not canonical:
        raise ValueError("empty story")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assigned_split(identity, seed):
    digest = hashlib.sha256(f"{POLICY}:{seed}:{identity}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest, "big") % sum(SPLIT_WEIGHTS.values())
    cursor = 0
    for split, weight in SPLIT_WEIGHTS.items():
        cursor += weight
        if bucket < cursor:
            return split
    raise AssertionError("invalid split weights")


def arrow_rows(paths):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    for source_index, path in enumerate(paths):
        row_index = 0
        with pa.memory_map(str(path), "r") as source, ipc.open_stream(source) as reader:
            for batch in reader:
                for text in batch.column("text").to_pylist():
                    yield source_index, row_index, text
                    row_index += 1


def write_partitions(rows, output, sources, seed=20261021):
    """Persist a unique normalized-content group once; duplicates cannot cross splits."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(SPLIT_WEIGHTS, 0)
    seen = set()
    duplicates = empty = total = 0
    with ExitStack() as stack:
        handles = {split: stack.enter_context((output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")) for split in counts}
        for source_index, row_index, text in rows:
            total += 1
            if not isinstance(text, str):
                raise ValueError(f"invalid text at source {source_index}, row {row_index}")
            try:
                identity = story_identity(text)
            except ValueError:
                empty += 1
                continue
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            split = assigned_split(identity, seed)
            row = {"story_id": identity, "source_index": source_index, "row_index": row_index,
                   "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            handles[split].write(json.dumps(row, sort_keys=True) + "\n")
            counts[split] += 1
    manifest = {
        "schema_version": 1, "policy": POLICY, "seed": seed,
        "weights_per_10000": SPLIT_WEIGHTS, "sources": sources,
        "source_rows": total, "unique_stories": len(seen), "duplicates_removed": duplicates,
        "empty_rows_removed": empty, "story_counts": counts,
        "files": {f"{split}.jsonl": sha256_file(output / f"{split}.jsonl") for split in counts},
        "normalization": "Unicode NFKC, casefold, whitespace collapse; first source occurrence retained",
        "historical_exposure": "Source corpus may have been used by historical models; new training must exclude the new held-out groups.",
        "limitations": "Exact normalized-content grouping only; no claim of semantic/paraphrase deduplication or historical novelty.",
        "tokenized": False, "complete": False,
    }
    # Promotion to complete belongs to prepare(), after checking source stability.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def prepare(paths, output, seed=20261021):
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("nonempty distinct source list required")
    sources = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths]
    # Never overwrite an earlier attempt, including an incomplete one.
    if Path(output).exists():
        raise FileExistsError(output)
    try:
        manifest = write_partitions(arrow_rows(paths), output, sources, seed)
    except Exception as exc:
        if Path(output).is_dir():
            failure = {"complete": False, "sources": sources, "seed": seed,
                       "error": f"{type(exc).__name__}: {exc}"}
            (Path(output) / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    for source in sources:
        if sha256_file(source["path"]) != source["sha256"]:
            raise RuntimeError("source changed while partitioning; incomplete output retained")
    if any(count == 0 for count in manifest["story_counts"].values()):
        raise RuntimeError("empty study partition; incomplete output retained")
    manifest["complete"] = True
    (Path(output) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    report = prepare(args.source, args.output, args.seed)
    print(json.dumps({"complete": report["complete"], "story_counts": report["story_counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
