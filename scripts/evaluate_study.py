"""Proposed held-out, story-isolated NLL evaluation for a completed study run.

This script never trains, selects checkpoints, or changes the held-out partition.
It loads only trusted local full checkpoints (including Python/NumPy RNG objects).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from laplacian import study_train
from laplacian.device import enforce_guard
from laplacian.reproducibility import config_hash, environment
from laplacian.runtime import write_metrics
from laplacian.study_tokens import validate_bundle
from laplacian.train import build_model

POLICY = "story-isolated-context256-stride128-nll-v1-proposed"
SPLITS = ("development", "calibration", "test")


def window_bounds(length, context=256, stride=128):
    """Yield (input_start, target_end_inclusive, first_scored_target_index)."""
    if type(length) is not int or length < 1:
        raise ValueError("Story token length must be a positive integer")
    if type(context) is not int or type(stride) is not int or not 1 <= stride <= context:
        raise ValueError("Require positive context and 1 <= stride <= context")
    previous_end = 0
    while previous_end < length - 1:
        target_end = min(length - 1, context if previous_end == 0 else previous_end + stride)
        start = max(0, target_end - context)
        yield start, target_end, previous_end + 1
        previous_end = target_end


def evaluate_stories(model, tokens, stories, *, context=256, stride=128, batch_size=4, device="cpu", pad_id=0,
                     guard=None):
    """Sum each story's NLL in float64, masking context-only and padded targets.

    ``stories`` is a validated list of offset/length/identity records. No input
    window can leave its story, even when adjacent stories are in one token file.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    # Validate policy even for an empty list.
    list(window_bounds(1, context, stride))
    device = torch.device(device)
    totals = [0.0 for _ in stories]
    counts = [0 for _ in stories]
    windows = [0 for _ in stories]
    pending = []
    model.eval()

    def flush():
        if not pending:
            return
        width = max(end - start for _, start, end, _ in pending)
        inputs = torch.full((len(pending), width), int(pad_id), dtype=torch.long)
        targets = torch.zeros_like(inputs)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for row, (story_index, start, end, score_from) in enumerate(pending):
            offset = stories[story_index]["offset"]
            width_here = end - start
            inputs[row, :width_here] = torch.from_numpy(np.array(tokens[offset + start:offset + end], dtype=np.int64))
            targets[row, :width_here] = torch.from_numpy(np.array(tokens[offset + start + 1:offset + end + 1], dtype=np.int64))
            mask[row, score_from - start - 1:width_here] = True
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        logits = model(inputs)["logits"].float()
        if logits.shape[:2] != inputs.shape:
            raise ValueError("Model logits do not match batched input dimensions")
        # Index first: no padding/context-only token contributes to CE, even if
        # its logits are nonfinite. Causality protects valid prefixes from pads.
        selected = logits[mask]
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("Non-finite held-out scored logits")
        nll = F.cross_entropy(selected, targets[mask], reduction="none").double()
        if not bool(torch.isfinite(nll).all()):
            raise FloatingPointError("Non-finite held-out target NLL")
        row_counts = mask.sum(dim=1).cpu().tolist()
        offset = 0
        for (story_index, _, _, _), count in zip(pending, row_counts):
            totals[story_index] += float(nll[offset:offset + count].sum().cpu())
            counts[story_index] += count
            windows[story_index] += 1
            offset += count
        pending.clear()
        if guard is not None:
            guard()

    with torch.inference_mode():
        for index, story in enumerate(stories):
            for start, end, score_from in window_bounds(story["length"], context, stride):
                pending.append((index, start, end, score_from))
                if len(pending) == batch_size:
                    flush()
        flush()
    results = []
    for story, total, count, windows_count in zip(stories, totals, counts, windows):
        if count != story["length"] - 1:
            raise RuntimeError("Each story target must be scored exactly once")
        if not math.isfinite(total):
            raise FloatingPointError("Non-finite held-out story NLL")
        results.append({**story, "summed_nll": total, "valid_targets": count, "windows": windows_count})
    return results


def read_stories(manifest, manifest_path, split, tokens):
    path = Path(manifest_path).parent / f"{split}.stories.jsonl"
    stories, identities, offset = [], set(), 0
    vocab, eos = manifest["tokenizer"]["vocab_size"], manifest["tokenizer"]["eos_id"]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            required = {"story_id", "source_index", "row_index", "text_sha256", "offset", "length"}
            if not required.issubset(row) or type(row["length"]) is not int or row["length"] < 2:
                raise ValueError("Invalid story index record")
            if type(row["offset"]) is not int or row["offset"] != offset or row["story_id"] in identities:
                raise ValueError("Story index offsets must be contiguous and identities unique")
            end = offset + row["length"]
            if end > len(tokens) or int(tokens[end - 1]) != eos or bool((tokens[offset:end] >= vocab).any()):
                raise ValueError("Story token bounds, vocabulary or final EOS do not match the manifest")
            stories.append(row)
            identities.add(row["story_id"])
            offset = end
    if (offset != len(tokens) or offset != manifest["token_counts"][split]
            or len(stories) != manifest["story_counts"][split] or not stories):
        raise ValueError("Story index counts do not match the token manifest")
    return stories


def sources():
    return [*study_train.source_records(), study_train.file_record(Path(__file__).resolve())]


def observed_files(paths):
    """Retain failed/missing input observations without hiding the primary error."""
    observations = []
    for path in paths:
        try:
            observations.append(study_train.file_record(path))
        except (OSError, ValueError) as exc:
            observations.append({"path": str(Path(path).resolve()), "read_error": type(exc).__name__})
    return observations


def device_for(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or budgeted CUDA evaluation is supported")
    if device.type == "cuda":
        if (device.index not in (None, 0) or not os.environ.get("STUDY_BUDGET_RUN_ID")
                or Path(os.environ.get("STUDY_BUDGET_LEDGER", "")).resolve() != (ROOT / "runs/publication_study/budget.jsonl").resolve()):
            raise RuntimeError("CUDA evaluation requires visible device0 and the canonical study budget wrapper")
        if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
            raise RuntimeError("The study CUDA evaluator requires the approved RTX5070")
    return device


def bound_inputs(config_path, checkpoint_path, manifest_path):
    config_path, checkpoint_path, manifest_path = map(lambda p: Path(p).resolve(), (config_path, checkpoint_path, manifest_path))
    records = [study_train.file_record(path) for path in (config_path, checkpoint_path, manifest_path)]
    config = study_train.read_config(config_path)
    if str(manifest_path) != config["study"]["token_manifest"]["path"]:
        raise ValueError("Evaluation token manifest differs from the training manifest")
    manifest = validate_bundle(manifest_path)
    folder = study_train.folder_for(config)
    if checkpoint_path.parent != folder:
        raise ValueError("Evaluation checkpoint must belong to its bound study run")
    metrics_path = folder / "metrics.json"
    records.extend(study_train.file_record(path) for path in (metrics_path, metrics_path.with_name(metrics_path.name + ".sha256")))
    metrics = study_train.read_verified_metrics(metrics_path)
    if (metrics.get("complete") is not True or metrics.get("steps") != config["steps"]
            or metrics.get("config") != config):
        raise ValueError("Evaluation requires a completed study run with matching configuration")
    checkpoint_record = records[1]
    allowed = [metrics.get("study", {}).get("final_checkpoint"), *metrics.get("study", {}).get("timing_states", [])]
    if checkpoint_record not in allowed:
        raise ValueError("Checkpoint bytes are not bound in the completed study receipt")
    for record in records:
        study_train.check_record(record)
    return config, manifest, records


def evaluate(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {
        "schema_version": 1, "policy": POLICY, "metric_status": "proposed_not_confirmatory_frozen",
        "complete": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "context": args.context, "stride": args.stride, "batch_size": args.batch_size,
        "device_requested": args.device, "max_stories": args.max_stories,
        "checkpoint_requested": str(Path(args.checkpoint).resolve()), "config_requested": str(Path(args.config).resolve()),
        "manifest_requested": str(Path(args.token_manifest).resolve()),
        "budget": {key: os.environ.get(key) for key in ("STUDY_BUDGET_RUN_ID", "STUDY_BUDGET_LEDGER")},
    }
    requested_paths = [args.config, args.checkpoint, args.token_manifest]
    receipt["attempted_inputs_before"] = observed_files(requested_paths)
    receipt["sources_before"] = sources()
    write_metrics(output / "attempt.json", receipt)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        if args.split not in SPLITS:
            raise ValueError("Evaluation split must be development, calibration or test")
        if args.max_stories is not None and (type(args.max_stories) is not int or args.max_stories < 1 or args.split == "test"):
            raise ValueError("Positive max_stories is for exploratory development/calibration only")
        list(window_bounds(1, args.context, args.stride))
        device = device_for(args.device)  # Refuse missing budget before any CUDA initialization.
        before_sources = receipt["sources_before"]
        receipt["sources"] = before_sources
        config, manifest, inputs = bound_inputs(args.config, args.checkpoint, args.token_manifest)
        receipt["inputs"] = inputs
        for record in receipt["attempted_inputs_before"]:
            if "read_error" in record:
                raise ValueError("A requested input could not be read at attempt start")
            study_train.check_record(record)
        receipt["training_config_sha256"] = config_hash(config)
        receipt["tokenizer"] = manifest["tokenizer"]
        receipt["split_files"] = {name: manifest["files"][name] for name in (f"{args.split}.u16", f"{args.split}.stories.jsonl")}
        if args.context > config["context"]:
            raise ValueError("Evaluation context exceeds the trained model context")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # CPU deserialization only; this checkpoint is trusted local input.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build_model(config, manifest["tokenizer"]["vocab_size"], torch.device("cpu"))
        study_train.validate_checkpoint(checkpoint, config, model=model)
        model.load_state_dict(checkpoint["model"], strict=True)
        receipt["checkpoint_step"] = checkpoint["step"]
        del checkpoint
        model.to(device).eval()
        receipt["environment"] = environment(device)
        receipt["precision"] = "FP32 logits and CE; FP64 target-loss sum; no autocast; TF32 disabled"
        tokens = np.memmap(Path(args.token_manifest).parent / f"{args.split}.u16", dtype="<u2", mode="r")
        stories = read_stories(manifest, args.token_manifest, args.split, tokens)
        receipt["available_stories"] = len(stories)
        if args.max_stories is not None:
            stories = stories[:args.max_stories]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        scoring_started = time.perf_counter()
        rows = evaluate_stories(model, tokens, stories, context=args.context, stride=args.stride,
                                batch_size=args.batch_size, device=device, pad_id=manifest["tokenizer"]["eos_id"],
                                guard=enforce_guard if device.type == "cuda" else None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scoring_seconds = time.perf_counter() - scoring_started
        del tokens
        total_nll = math.fsum(row["summed_nll"] for row in rows)
        total_targets = sum(row["valid_targets"] for row in rows)
        if not total_targets:
            raise ValueError("Held-out evaluation produced no scored targets")
        ce = total_nll / total_targets
        results_path = output / "stories.jsonl"
        with results_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Revalidate originals and all token/input provenance after scoring.
        for record in inputs:
            study_train.check_record(record)
        study_train.validate_config(config)
        validate_bundle(args.token_manifest)
        if sources() != before_sources:
            raise ValueError("Evaluation source files changed during scoring")
        receipt.update(
            complete=True, evaluated_stories=len(rows), valid_targets=total_targets, summed_nll=total_nll,
            token_weighted_mean_nll=ce, perplexity=math.exp(ce) if ce < 700 else None,
            story_results=study_train.file_record(results_path), scoring_seconds=scoring_seconds,
            elapsed_seconds_before_receipt=time.perf_counter() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else None,
            freshness_checked_at_completion=True,
            attempted_inputs_after=observed_files(requested_paths), sources_after=sources(),
            limitations="Proposed finite-context metric. First token unscored; EOS scored once. No test-set model selection or generalization conclusion is established by this receipt.",
        )
        # Promote the success JSON last, so an interrupted sidecar write never
        # exposes an apparently complete final result without its digest.
        pending_metrics = output / "metrics.pending.json"
        write_metrics(pending_metrics, receipt)
        pending_metrics.with_name(pending_metrics.name + ".sha256").replace(output / "metrics.json.sha256")
        pending_metrics.replace(output / "metrics.json")
        return receipt
    except BaseException as exc:
        receipt.update(complete=False, error={"type": type(exc).__name__, "message": str(exc)},
                       attempted_inputs_after=observed_files(requested_paths),
                       sources_after=observed_files([record["path"] for record in receipt["sources_before"]]),
                       elapsed_seconds_before_receipt=time.perf_counter() - started)
        write_metrics(output / "error.json", receipt)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-stories", type=int, help="Exploratory first-N development/calibration subset; forbidden on test")
    args = parser.parse_args(argv)
    result = evaluate(args)
    print(json.dumps({"complete": result["complete"], "split": result["split"], "stories": result["evaluated_stories"],
                      "valid_targets": result["valid_targets"], "mean_nll": result["token_weighted_mean_nll"]}))


if __name__ == "__main__":
    main()
