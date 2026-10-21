"""Exploratory ordinary-update rate from a bound study checkpoint and train stream.

GPU execution requires the canonical study-budget wrapper. This probe mutates
only its in-memory state and new output directory; it never advances the study
run. Timing excludes validation, sampling/transfers, checkpoint IO and telemetry.
The cooperative limit cannot interrupt an in-flight CUDA call; the wrapper must
provide the hard deadline. This is a rate diagnostic, not confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "pyproject.toml"]
    paths += sorted((ROOT / "src/laplacian").glob("*.py"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


IMPORTED_SOURCE_HASHES = source_hashes()

from laplacian import study_train as study
from laplacian import train as training
from laplacian.device import GUARD_MIB
from laplacian.operators_torch import fractional_policy
from laplacian.reproducibility import config_hash, ensure_finite_state, environment
from laplacian.runtime import write_metrics


def require_gpu():
    ledger = ROOT / "runs/publication_study/budget.jsonl"
    if (not os.environ.get("STUDY_BUDGET_RUN_ID")
            or Path(os.environ.get("STUDY_BUDGET_LEDGER", "")).resolve() != ledger.resolve()):
        raise RuntimeError("Rate probe requires the canonical study-budget wrapper")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Rate probe requires the approved RTX 5070 as visible CUDA device zero")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Rate probe requires BF16 support")
    return device


def memory_guard(device):
    free, total = torch.cuda.mem_get_info(device)
    used = (total-free)/1024**2
    if used > GUARD_MIB:
        raise RuntimeError(f"Rate probe device usage {used:.0f} MiB exceeds {GUARD_MIB:.0f} MiB")
    return used


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_probe_config(config):
    expected = {"m": 4, "k": 0, "microbatch": 8, "accum": 8, "context": 256}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Rate probe requires production M4/k0/microbatch8/accum8/context256")
    if config["study"]["vocab_size"] != 50257:
        raise ValueError("Rate probe requires the pinned production GPT-2 vocabulary")
    weights = training.penalty_weights(config, 0)
    stage = config["study"]["stage"]
    if stage == "ce_pretrain":
        if any(weights.values()):
            raise ValueError("CE rate probe cannot activate spectral penalties")
    elif stage == "spectral_continuation":
        if config["study"].get("penalty_step") != 135000 or not all(value > 0 for value in weights.values()):
            raise ValueError("Continuation rate probe requires the fixed all-active step135000 weights")
    else:
        raise ValueError("Rate probe requires a CE or fixed-weight study stage")
    return weights


def reject_intervention_boundary(config, source_step, updates):
    if config["dead_units"] and any(step < 30000 and step % 1000 == 0
                                    for step in range(source_step+1, source_step+updates+1)):
        raise ValueError("Ordinary-update rate probe cannot cross a dead-unit resampling boundary")


def prepare_batches(config, generator, source_step, updates, output, device, deadline):
    store = study.StudyTokenStore(config)
    store.start_segment(output, source_step)
    batches, records = [], []
    start_rng = tensor_hash(generator.get_state())
    for index in range(updates):
        for micro in range(config["accum"]):
            deadline()
            store.set_position(source_step+index, micro)
            inputs, targets = store.batch(config["microbatch"], generator)
            records.append({"probe_update": index, "source_step_plus_offset": source_step+index,
                            "micro": micro, "shape": list(inputs.shape), "dtype": str(inputs.dtype),
                            "inputs_sha256": tensor_hash(inputs), "targets_sha256": tensor_hash(targets)})
            batches.append((inputs.to(device), targets.to(device)))
    store.flush()
    return batches, {"batches": records, "sample_log": study.file_record(store.sample_log),
                     "sampler_state_sha256_before": start_rng,
                     "sampler_state_sha256_after": tensor_hash(generator.get_state())}


def ordinary_update(model, optimizer, batches, config, step, device, fired_sums=None, fired_tokens=0):
    rate = training.learning_rate(step, config["steps"], config["warmup"], config["lr"], config["min_lr"])
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    loss_sum, ce_sum = 0., 0.
    for inputs, targets in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, ce, outputs, penalties = training.forward_loss(model, inputs, targets, config, step)
        if not config["spectral"] and penalties is not None:
            raise RuntimeError("CE probe unexpectedly evaluated penalties")
        loss_value, ce_value = training.finite_loss_values(loss, ce, step)
        (loss/config["accum"]).backward()
        loss_sum += loss_value
        ce_sum += ce_value
        if config["dead_units"] and step < 30000:
            fired_sums, fired_tokens = training._add_fired(fired_sums, fired_tokens, outputs)
    training.checked_optimizer_step(model, optimizer, config["clip"], step)
    return {"loss": loss_sum/config["accum"], "ce": ce_sum/config["accum"], "lr": rate}, fired_sums, fired_tokens


def run_probe(checkpoint, output, *, warmup=5, steps=20, max_seconds=120.):
    if type(warmup) is not int or type(steps) is not int or warmup < 1 or steps < 1:
        raise ValueError("Warmup and measured steps must be positive integers")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
        raise ValueError("Cooperative limit must be in (0, 120] seconds")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint = Path(checkpoint).resolve()
    report = {"complete": False, "status": "running", "run_id": output.name, "host_gpu": False,
              "scope": "Exploratory ordinary-update rate; in-memory updates beyond source checkpoint, no study advancement or quality claim",
              "timer_scope": "Synchronized host wall time: forward/loss, per-micro finite checks, backward, optional firing accumulation, clipping and AdamW; sampling, transfers, IO, validation and telemetry excluded",
              "source_checkpoint_requested": str(checkpoint), "source_sha256_before": IMPORTED_SOURCE_HASHES,
              "fractional_policy": fractional_policy(), "warmup_updates": warmup, "measured_updates": steps,
              "max_seconds_cooperative": max_seconds, "updates": [],
              "budget": {key: os.environ.get(key) for key in ("STUDY_BUDGET_RUN_ID", "STUDY_BUDGET_LEDGER")},
              "limitations": ["One checkpoint and short process; no confidence interval or sustained-rate guarantee",
                              "Checkpoint, export, validation, sampler and transfer overhead must be added to budget projections",
                              "No dead-unit resampling event, pruning, checkpoint or model export is executed",
                              "Warmup changes the in-memory state; this is not a frozen forward-only timing"]}
    write_metrics(output / "metrics.json", report)
    failure, config, input_records = None, None, []

    def deadline():
        if time.perf_counter()-started > max_seconds:
            raise TimeoutError("Rate probe exceeded its cooperative limit; use wrapper for a hard deadline")

    try:
        if source_hashes() != IMPORTED_SOURCE_HASHES:
            raise RuntimeError("Rate probe source changed after import")
        device = require_gpu()
        report["host_gpu"] = True
        torch.backends.cuda.matmul.allow_tf32 = True
        deadline()
        config_path = checkpoint.parent / "study_config.json"
        config_record = study.file_record(config_path)
        input_records = [config_record]
        report["inputs_before"] = input_records
        config = study.read_config(config_path)
        study.check_record(config_record)
        deadline()
        if checkpoint.parent != study.folder_for(config):
            raise ValueError("Checkpoint is outside its bound study run")
        report["spectral_weights"] = validate_probe_config(config)
        report["training_config"] = config
        report["training_config_sha256"] = config_hash(config)
        manifest_path = Path(config["study"]["token_manifest"]["path"])
        input_paths = [checkpoint, manifest_path]
        input_paths += [manifest_path.parent / name for name in sorted(config["study"]["data_files"])]
        input_records = [config_record, *[study.file_record(path) for path in input_paths]]
        report["inputs_before"] = input_records
        report["environment"] = environment(device)
        report["environment"]["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        deadline()
        model = training.build_model(config, config["study"]["vocab_size"], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                      weight_decay=config["weight_decay"])
        generator, metadata = torch.Generator(device="cpu"), {}
        deadline()
        source_step, pulses, stripped, fired_sums, fired_tokens = training.load_checkpoint(
            checkpoint, model, optimizer, generator, device, config=config, resume_metadata=metadata)
        report["restore_metadata"] = metadata
        if metadata.get("resume_status") != "state_complete" or metadata.get("resume_limitations") or pulses or stripped:
            raise ValueError("Rate probe requires complete checkpoint state in the same environment")
        reject_intervention_boundary(config, source_step, warmup+steps)
        report["source_step"] = source_step
        report["probe_virtual_step_end"] = source_step+warmup+steps
        report["initialization_seconds"] = time.perf_counter()-started
        deadline()
        memory_guard(device)
        prepared = time.perf_counter()
        batches, data_record = prepare_batches(config, generator, source_step, warmup+steps, output, device, deadline)
        report["data"] = data_record
        torch.cuda.synchronize(device)
        report["batch_preparation_seconds"] = time.perf_counter()-prepared
        report["device_used_mib_before"] = memory_guard(device)
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(warmup+steps):
            deadline()
            torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            values, fired_sums, fired_tokens = ordinary_update(
                model, optimizer, batches[index*config["accum"]:(index+1)*config["accum"]],
                config, source_step+index, device, fired_sums, fired_tokens)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-update_started
            report["updates"].append({"index": index, "warmup": index < warmup, "seconds": seconds, **values})
            memory_guard(device)
        ensure_finite_state({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        measured = [row["seconds"] for row in report["updates"] if not row["warmup"]]
        report["summary"] = {"mean_seconds_per_update": statistics.mean(measured),
                             "median_seconds_per_update": statistics.median(measured),
                             "minimum_seconds_per_update": min(measured), "maximum_seconds_per_update": max(measured),
                             "measured_seconds": sum(measured),
                             "warmup_seconds": sum(row["seconds"] for row in report["updates"] if row["warmup"]),
                             "tokens_per_update": config["microbatch"]*config["accum"]*config["context"],
                             "tokens_per_second_from_mean": config["microbatch"]*config["accum"]*config["context"]/statistics.mean(measured),
                             "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2,
                             "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/1024**2}
        deadline()
        report["complete"], report["status"] = True, "passed"
    except BaseException as exc:
        failure = exc
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        postcheck_started = time.perf_counter()
        try:
            report["source_sha256_after"] = source_hashes()
            if report["source_sha256_after"] != IMPORTED_SOURCE_HASHES:
                raise RuntimeError("Source changed during rate probe")
            report["inputs_after"] = [study.file_record(record["path"]) for record in input_records]
            if report["inputs_after"] != input_records:
                raise RuntimeError("Bound input changed during rate probe")
            if config is not None:
                study.validate_config(config)  # Includes full production bundle/input closure.
            if "data" in report:
                study.check_record(report["data"]["sample_log"])
            deadline()
            report["freshness_checks_passed"] = True
        except BaseException as exc:
            report["complete"], report["status"] = False, "failed"
            report["postcheck_error"] = f"{type(exc).__name__}: {exc}"
            if failure is None:
                failure = exc
        report["postcheck_seconds"] = time.perf_counter()-postcheck_started
        report["full_elapsed_seconds_before_final_receipt"] = time.perf_counter()-started
        write_metrics(output / "metrics.json", report)
    if failure is not None:
        raise failure
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=120.)
    args = parser.parse_args(argv)
    report = run_probe(args.checkpoint, args.output, warmup=args.warmup, steps=args.steps, max_seconds=args.max_seconds)
    print(json.dumps({"complete": report["complete"], **report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
