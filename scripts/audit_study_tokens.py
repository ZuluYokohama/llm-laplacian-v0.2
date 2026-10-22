"""CPU structural audit of a completed production study token bundle.

Checks every story boundary, partition row binding and token range. This does
not independently retokenize the original text or establish semantic deduplication.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laplacian.study_tokens import HEX_SHA, SPLITS, file_record, validate_bundle, write_json
from laplacian.study_data import assigned_split


def audit_structure(manifest_path, manifest):
    folder = Path(manifest_path).resolve().parent
    partitions = Path(manifest["partition_manifest"]["path"]).parent
    seen = set()
    results = {}
    for split in SPLITS:
        tokens = np.memmap(folder / f"{split}.u16", dtype="<u2", mode="r")
        for start in range(0, len(tokens), 1_000_000):
            if np.any(tokens[start:start + 1_000_000] >= manifest["tokenizer"]["vocab_size"]):
                raise ValueError(f"Token outside vocabulary in {split}")
        count, offset = 0, 0
        with (folder / f"{split}.stories.jsonl").open(encoding="utf-8") as indexed, \
                (partitions / f"{split}.jsonl").open(encoding="utf-8") as selected:
            for story_line, reference_line in itertools.zip_longest(indexed, selected):
                if story_line is None or reference_line is None:
                    raise ValueError(f"Story/partition row count mismatch in {split}")
                story, reference = json.loads(story_line), json.loads(reference_line)
                reference_keys = {"story_id", "source_index", "row_index", "text_sha256"}
                if (not isinstance(reference, dict) or set(reference) != reference_keys
                        or not isinstance(story, dict) or set(story) != reference_keys | {"offset", "length"}):
                    raise ValueError(f"Invalid story or partition record schema in {split}")
                for field in ("story_id", "text_sha256"):
                    if not isinstance(reference[field], str) or not HEX_SHA.fullmatch(reference[field]):
                        raise ValueError(f"Invalid partition {field} in {split}")
                if (type(reference["source_index"]) is not int
                        or not 0 <= reference["source_index"] < len(manifest["sources"])
                        or type(reference["row_index"]) is not int or reference["row_index"] < 0):
                    raise ValueError(f"Invalid partition source/row index in {split}")
                if (any(type(story[key]) is not type(reference[key]) for key in reference)
                        or {key: story[key] for key in reference} != reference):
                    raise ValueError(f"Story does not match its partition row in {split}")
                identity = story["story_id"]
                if identity in seen:
                    raise ValueError("Duplicate normalized story identity within or across splits")
                seen.add(identity)
                if assigned_split(identity, manifest["partition_seed"]) != split:
                    raise ValueError(f"Story is assigned to the wrong partition in {split}")
                if type(story.get("offset")) is not int or story["offset"] != offset:
                    raise ValueError(f"Noncontiguous or invalid story offset in {split}")
                length = story.get("length")
                if type(length) is not int or length < 2 or offset + length > len(tokens):
                    raise ValueError(f"Invalid story length in {split}")
                if int(tokens[offset + length - 1]) != manifest["tokenizer"]["eos_id"]:
                    raise ValueError(f"Missing terminal EOS in {split}")
                if np.any(tokens[offset:offset + length - 1] == manifest["tokenizer"]["eos_id"]):
                    raise ValueError(f"In-story EOS before the terminal token in {split}")
                offset += length
                count += 1
        if (count != manifest["story_counts"][split] or offset != len(tokens)
                or offset != manifest["token_counts"][split]):
            raise ValueError(f"Story/token accounting mismatch in {split}")
        results[split] = {"stories": count, "tokens": offset, "all_boundaries_have_eos": True,
                          "no_interior_eos": True, "all_tokens_in_vocabulary": True, "partition_rows_match": True}
        del tokens
    return {"splits": results, "disjoint_unique_story_ids": len(seen)}


def run(manifest_path, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {"complete": False, "scope": "CPU structural audit; no independent retokenization or semantic deduplication"}
    failure = None
    try:
        source = file_record(__file__)
        input_record = file_record(manifest_path)
        manifest = validate_bundle(manifest_path)
        record.update(audit_structure(manifest_path, manifest))
        # Recheck all bound bytes after streaming, including the producer source.
        validate_bundle(manifest_path)
        if file_record(manifest_path) != input_record or file_record(__file__) != source:
            raise RuntimeError("Audit source or manifest changed during the audit")
        record.update(complete=True, token_manifest=input_record, audit_source=source)
    except BaseException as exc:
        failure = exc
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = time.perf_counter() - started
    write_json(output / "audit.json", record)
    if failure is not None:
        raise failure
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), sort_keys=True))
