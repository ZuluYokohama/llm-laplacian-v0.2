"""One exploratory matched timing pair; no confirmatory collector or promotion.

prepare is CPU-only. run-pair explicitly dispatches two fresh budgeted workers.
Only trusted local completed-study checkpoints may be deserialized. There is no
retry/resume, and no held-out evaluation or mutation of a training run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import study_budget as budget


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def sources():
    return [file_record(path) for path in
            [SCRIPT, ROOT / "scripts/study_budget.py", ROOT / "pyproject.toml",
             *sorted((ROOT / "src/laplacian").glob("*.py"))]]


IMPORTED_SOURCES = sources()
from laplacian import study_train as study
from laplacian import train as training
from laplacian.device import GUARD_MIB
from laplacian.operators_torch import fractional_policy
from laplacian.reproducibility import config_hash, ensure_finite_state, environment, restore_rng_state, rng_state

KIND = "exploratory_timing_pair_v1"
BOUNDARY = "synchronized-zero-grad-forward-loss-finite-backward-accumulation-clipping-adamw-v1"
LEDGER = ROOT / "runs/publication_study/budget.jsonl"
DECISIONS = ROOT / "paper/planning/decisions.json"
ORDER = {"AB": ("baseline", "optimized"), "BA": ("optimized", "baseline")}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def immutable_json(path, value):
    """Exclusive LF UTF-8 files; sidecars use the existing logical-text convention."""
    path = Path(path)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    with path.with_name(path.name + ".sha256").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(hashlib.sha256(raw).hexdigest() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_record(path)


def check(record):
    if file_record(record["path"]) != record:
        raise ValueError(f"Bound input changed: {record['path']}")


def load_json(path):
    raw = Path(path).read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def verified_json(path):
    record = file_record(path)
    value = study.read_verified_metrics(path)
    check(record)
    return value


def observed(records):
    result = []
    for record in records:
        try:
            result.append(file_record(record["path"]))
        except OSError as exc:
            result.append({"path": record["path"], "read_error": type(exc).__name__})
    return result


def receipt_base():
    return {"schema_version": 1, "kind": KIND, "complete": False,
            "confirmatory_claim_permitted": False, "numerical_acceptance": "not_evaluated",
            "eligibility": "not_evaluated", "created_unix": time.time(),
            "scope": "Exploratory one-pair cost diagnostic; no quality or confirmatory speed claim"}


def counts(warmup, steps):
    if type(warmup) is not int or type(steps) is not int or not 1 <= warmup <= 3 or not 1 <= steps <= 12:
        raise ValueError("Prototype requires 1..3 warmup and 1..12 measured updates")


def validate_work(config):
    expected = {"m": 4, "k": 0, "microbatch": 8, "accum": 8, "context": 256}
    if any(type(config.get(key)) is not int or config[key] != value for key, value in expected.items()):
        raise ValueError("Prototype requires production M4/k0/microbatch8/accum8/context256")
    if config["study"].get("vocab_size") != 50257 or config["study"].get("stage") != "ce_pretrain":
        raise ValueError("Prototype requires a bound CE parent with the pinned production vocabulary")


def bound_inputs(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint.parent / "study_config.json"
    records = [file_record(config_path), file_record(checkpoint)]
    config = study.read_config(config_path)
    check(records[0])
    validate_work(config)
    if checkpoint.parent != study.folder_for(config):
        raise ValueError("Checkpoint is outside its bound study run")
    metrics_path = checkpoint.parent / "metrics.json"
    records.extend(file_record(path) for path in (metrics_path, metrics_path.with_name("metrics.json.sha256")))
    metrics = verified_json(metrics_path)
    if (metrics.get("complete") is not True or metrics.get("steps") != config["steps"]
            or metrics.get("config") != config):
        raise ValueError("Prototype requires a completed CE run and matching metrics/configuration")
    allowed = [metrics.get("study", {}).get("final_checkpoint"), *metrics.get("study", {}).get("timing_states", [])]
    if records[1] not in allowed:
        raise ValueError("Checkpoint bytes are not bound in the completed study receipt")
    manifest = config["study"]["token_manifest"]
    records.append(manifest)
    records.extend(file_record(Path(manifest["path"]).parent / name) for name in sorted(config["study"]["data_files"]))
    for record in records:
        check(record)
    return config, records


def state_hash(value):
    """Typed, framed semantic state identity, independent of torch.save layout/device."""
    digest = hashlib.sha256()
    def frame(tag, raw):
        digest.update(tag.encode("ascii") + b":" + str(len(raw)).encode("ascii") + b":" + raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            frame("tensor", canonical([str(tensor.dtype), list(tensor.shape)]))
            frame("data", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            frame("numpy", canonical([item.dtype.str, list(item.shape)]))
            frame("data", item.tobytes(order="C"))
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            frame("dict", str(len(item)).encode("ascii"))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            frame(type(item).__name__, str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        elif item is None or type(item) in (bool, int, float, str):
            frame(type(item).__name__, canonical(item))
        else:
            raise TypeError(f"Unsupported state identity type: {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def batch_manifest(config, seed, warmup, steps):
    folder = Path(config["study"]["token_manifest"]["path"]).parent
    tokens = np.memmap(folder / "train.u16", dtype="<u2", mode="r")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    before = tensor_hash(generator.get_state())
    rows = []
    for index in range(warmup + steps):
        for micro in range(config["accum"]):
            starts = torch.randint(0, len(tokens) - config["context"] - 1,
                                   (config["microbatch"],), generator=generator).tolist()
            packed = np.stack([np.array(tokens[start:start + config["context"] + 1], dtype=np.int64) for start in starts])
            inputs, targets = torch.from_numpy(packed[:, :-1]), torch.from_numpy(packed[:, 1:])
            rows.append({"update": index, "micro": micro, "starts": starts, "shape": list(inputs.shape),
                         "inputs_sha256": tensor_hash(inputs), "targets_sha256": tensor_hash(targets)})
    return {"schema_version": 1, "split": "train", "batch_seed": seed,
            "token_manifest_sha256": config["study"]["token_manifest"]["sha256"],
            "train_tokens": config["study"]["data_files"]["train.u16"],
            "story_index": config["study"]["data_files"]["train.stories.jsonl"],
            "sampler_policy": "legacy_exclusive_high_len_minus_context_minus_one",
            "sampler_sha256_before": before, "sampler_sha256_after": tensor_hash(generator.get_state()),
            "batches": rows}


def prepare(checkpoint, output, *, pair_id, penalty_step, batch_seed, order, baseline_source=None, warmup=3, steps=12):
    counts(warmup, steps)
    if not isinstance(pair_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,49}", pair_id):
        raise ValueError("Pair ID must be a unique filename-safe identifier of at most 50 characters")
    if type(penalty_step) is not int or penalty_step not in (0, 135000):
        raise ValueError("Penalty step must be exactly 0 or 135000")
    if type(batch_seed) is not int or not 0 <= batch_seed < 2**63 or order not in ORDER:
        raise ValueError("Explicit nonnegative int63 sampler seed and AB/BA order are required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    attempt = receipt_base()
    attempted = [Path(checkpoint).resolve(), Path(checkpoint).resolve().parent / "study_config.json",
                 Path(baseline_source or study.DEFAULT_BASELINE).resolve()]
    attempt.update(checkpoint_requested=str(Path(checkpoint).resolve()), sources_before=IMPORTED_SOURCES,
                   attempted_inputs_before=observed([{"path": str(path)} for path in attempted]))
    immutable_json(output / "prepare_attempt.json", attempt)
    records = []
    try:
        if sources() != IMPORTED_SOURCES:
            raise ValueError("Prototype source changed after import")
        config, records = bound_inputs(checkpoint)
        for record in attempt["attempted_inputs_before"]:
            if "read_error" in record:
                raise ValueError("A requested input was unreadable at preparation start")
            check(record)
        baseline = file_record(baseline_source or study.DEFAULT_BASELINE)
        if baseline["sha256"] != study.BASELINE_SHA256:
            raise ValueError("Baseline must match immutable prior loss source")
        records.append(baseline)
        batches = batch_manifest(config, batch_seed, warmup, steps)
        batch_record = immutable_json(output / "batches.json", batches)
        for record in records:
            check(record)
        study.validate_config(config)
        if observed(records) != records:
            raise ValueError("Bound input changed during final preparation validation")
        if sources() != IMPORTED_SOURCES:
            raise ValueError("Prototype source changed during preparation")
        plan = receipt_base()
        plan.update(complete=True, stage="prepared", pair_id=pair_id, order=order, penalty_step=penalty_step,
                    batch_seed=batch_seed, warmup=warmup, steps=steps, inputs=records,
                    config=records[0], checkpoint=records[1], baseline_source=baseline,
                    batch_manifest=batch_record, sources=IMPORTED_SOURCES,
                    training_config_sha256=config_hash(config), fractional_policy=fractional_policy(),
                    optimizer_policy="preserve_checkpoint_parameter_groups_and_moments_hold_loaded_lr_constant",
                    sampler_override="local_data_generator_only_explicit_pair_seed",
                    timer_boundary=BOUNDARY, cpu_preparation_seconds=time.perf_counter() - started)
        immutable_json(output / "plan.json", plan)
        return plan
    except BaseException as exc:
        error = {**attempt, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                 "inputs_before": records, "inputs_after": observed(records),
                 "attempted_inputs_after": observed([{"path": str(path)} for path in attempted]),
                 "sources_after": observed(IMPORTED_SOURCES)}
        immutable_json(output / "prepare_error.json", error)
        raise


def read_plan(path):
    path = Path(path).resolve()
    raw_record = file_record(path)
    plan = verified_json(path)
    if (plan.get("schema_version") != 1 or plan.get("kind") != KIND or plan.get("complete") is not True
            or plan.get("confirmatory_claim_permitted") is not False or plan.get("stage") != "prepared"):
        raise ValueError("Only a prepared exploratory timing plan is accepted")
    counts(plan["warmup"], plan["steps"])
    if (type(plan["penalty_step"]) is not int or plan["penalty_step"] not in (0, 135000)
            or type(plan["batch_seed"]) is not int or not 0 <= plan["batch_seed"] < 2**63 or plan["order"] not in ORDER
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,49}", plan["pair_id"])):
        raise ValueError("Invalid exploratory pair identity or override")
    if plan["sources"] != IMPORTED_SOURCES or sources() != IMPORTED_SOURCES:
        raise ValueError("Plan source identity is stale")
    if (plan["timer_boundary"] != BOUNDARY or plan["fractional_policy"] != fractional_policy()
            or plan["optimizer_policy"] != "preserve_checkpoint_parameter_groups_and_moments_hold_loaded_lr_constant"
            or plan["sampler_override"] != "local_data_generator_only_explicit_pair_seed"):
        raise ValueError("Unsupported benchmark override contract")
    for record in [raw_record, *plan["inputs"], plan["batch_manifest"]]:
        check(record)
    if plan["config"] != plan["inputs"][0] or plan["checkpoint"] != plan["inputs"][1] or plan["baseline_source"] not in plan["inputs"]:
        raise ValueError("Plan input roles differ from bound records")
    return plan, raw_record


def require_gpu(run_id):
    if (not run_id.strip() or os.environ.get("STUDY_BUDGET_RUN_ID") != run_id
            or Path(os.environ.get("STUDY_BUDGET_LEDGER", "")).resolve() != LEDGER.resolve()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"):
        raise RuntimeError("Pair worker requires its canonical budget run and CUDA device zero")
    # The accounting parent holds the ledger lock until this child exits. Read
    # the durable launch prefix without acquiring that lock (which would deadlock).
    run = budget.accounting(budget.read_ledger(LEDGER)).get(run_id)
    if run is None or run["settled"] or not run["launch_authorized"]:
        raise RuntimeError("Pair worker requires an actually launched open canonical reservation")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Pair worker requires the approved RTX 5070")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pair worker requires BF16 support")
    return device


def memory_guard(device):
    if device.type != "cuda":
        return 0.0  # Only reachable through explicit CPU test substitutions.
    free, total = torch.cuda.mem_get_info(device)
    used = (total - free) / 1024**2
    if used > GUARD_MIB:
        raise RuntimeError(f"Device usage {used:.0f} MiB exceeds {GUARD_MIB:.0f} MiB")
    return used


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def override_config(config, plan, implementation):
    if implementation not in ORDER[plan["order"]]:
        raise ValueError("Unknown implementation")
    derived = copy.deepcopy(config)
    derived.update(spectral=True, dead_units=False)
    derived["study"].update(loss_implementation=implementation, penalty_step=plan["penalty_step"],
                            baseline_loss_source=plan["baseline_source"] if implementation == "baseline" else None)
    return derived


STATE_KEYS = ("step", "model", "optimizer", "generator", "rng_state", "requires_grad", "module_training",
              "pulses_done", "stripped", "fired_tokens", "fired_sums")


def capture_state(model, optimizer, generator, device, payload, fired_sums):
    return {"step": payload["step"], "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "generator": generator.get_state(), "rng_state": rng_state(device),
            "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
            "module_training": {name: module.training for name, module in model.named_modules()},
            "pulses_done": payload["pulses_done"], "stripped": payload["stripped"],
            "fired_tokens": payload["fired_tokens"], "fired_sums": fired_sums}


def restore(checkpoint, config, device):
    model = training.build_model(config, config["study"]["vocab_size"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  weight_decay=config["weight_decay"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    study.validate_checkpoint(payload, config, model=model)
    filename = Path(checkpoint).name
    snapshot = re.fullmatch(r"state_step(\d{7})\.pt", filename)
    expected_step = config["steps"] if filename == "last.pt" else int(snapshot[1]) if snapshot else None
    if (expected_step is None or payload["step"] != expected_step
            or (snapshot and expected_step not in config["study"]["snapshot_steps"])):
        raise ValueError("Checkpoint step differs from its completed final/snapshot role")
    if payload["pulses_done"] or payload["stripped"] or payload["environment"] != environment(device):
        raise ValueError("Checkpoint requires unchanged environment and no previous interventions")
    expected = state_hash({key: payload.get(key) for key in STATE_KEYS})
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    generator = torch.Generator(device="cpu").set_state(payload["generator"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(payload["requires_grad"][name])
    for name, module in model.named_modules():
        module.training = payload["module_training"][name]
    fired_sums = None if payload.get("fired_sums") is None else [value.to(device) for value in payload["fired_sums"]]
    if not restore_rng_state(payload["rng_state"], device):
        raise ValueError("Checkpoint RNG device domain changed")
    actual = state_hash(capture_state(model, optimizer, generator, device, payload, fired_sums))
    if expected != actual:
        raise ValueError("Loaded full training state differs from checkpoint")
    return model, optimizer, generator, payload, fired_sums, actual


def reconstruct_batches(config, plan, generator, device):
    expected = verified_json(plan["batch_manifest"]["path"])
    actual = batch_manifest(config, plan["batch_seed"], plan["warmup"], plan["steps"])
    if actual != expected:
        raise ValueError("Prepared batch sequence differs from immutable manifest")
    generator.manual_seed(plan["batch_seed"])
    tokens = np.memmap(Path(config["study"]["token_manifest"]["path"]).parent / "train.u16", dtype="<u2", mode="r")
    result = []
    for row in expected["batches"]:
        starts = torch.randint(0, len(tokens) - config["context"] - 1,
                               (config["microbatch"],), generator=generator).tolist()
        if starts != row["starts"]:
            raise ValueError("Sampler replay disagrees with batch starts")
        packed = np.stack([np.array(tokens[start:start + config["context"] + 1], dtype=np.int64) for start in starts])
        result.append((torch.from_numpy(packed[:, :-1]).to(device), torch.from_numpy(packed[:, 1:]).to(device)))
    if tensor_hash(generator.get_state()) != expected["sampler_sha256_after"]:
        raise ValueError("Sampler replay final state differs")
    return result


def update(model, optimizer, batches, config, device):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = ce_sum = 0.0
    for inputs, targets in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, ce, _outputs, _penalties = training.forward_loss(model, inputs, targets, config, 0)
        loss_value, ce_value = float(loss.detach().item()), float(ce.detach().item())
        if not math.isfinite(loss_value) or not math.isfinite(ce_value):
            raise FloatingPointError("Nonfinite loss or CE before optimizer mutation")
        (loss / config["accum"]).backward()
        loss_sum += loss_value
        ce_sum += ce_value
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"], error_if_nonfinite=True)
    except RuntimeError as exc:
        raise FloatingPointError("Nonfinite gradients before optimizer mutation") from exc
    optimizer.step()
    return {"loss": loss_sum / config["accum"], "ce": ce_sum / config["accum"], "gradient_norm": float(norm)}


def worker(plan_path, implementation, output, run_id):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = receipt_base()
    receipt.update(implementation=implementation, run_id=run_id, execution_id=uuid.uuid4().hex, pid=os.getpid(),
                   plan_requested=str(Path(plan_path).resolve()), sources_before=IMPORTED_SOURCES, updates=[],
                   attempted_plan_before=observed([{"path": str(Path(plan_path).resolve())}]))
    immutable_json(output / "attempt.json", receipt)
    records, config = [], None
    try:
        device = require_gpu(run_id)  # Before CUDA queries, allocation, baseline loading or checkpoint pickle.
        torch.backends.cuda.matmul.allow_tf32 = True
        plan, plan_record = read_plan(plan_path)
        if receipt["attempted_plan_before"] != [plan_record]:
            raise ValueError("Requested plan bytes changed during worker admission")
        expected_id = f"pair-{plan['pair_id']}-{implementation}"
        if run_id != expected_id or implementation not in ORDER[plan["order"]]:
            raise ValueError("Worker identity differs from explicit pair plan")
        records = [plan_record, *plan["inputs"], plan["batch_manifest"]]
        config, inputs = bound_inputs(plan["checkpoint"]["path"])
        if inputs != plan["inputs"][:-1] or config_hash(config) != plan["training_config_sha256"]:
            raise ValueError("Original training configuration or inputs differ from plan")
        # Load the same baseline module in both processes before restoring RNG.
        study.baseline_module(plan["baseline_source"]["path"], plan["baseline_source"]["sha256"])
        memory_guard(device)
        model, optimizer, generator, payload, fired_sums, restored_hash = restore(plan["checkpoint"]["path"], config, device)
        receipt.update(source_step=payload["step"], restored_state_sha256=restored_hash,
                       source_checkpoint=plan["checkpoint"], original_training_config_sha256=config_hash(config),
                       local_sampler_sha256_before_override=tensor_hash(generator.get_state()),
                       optimizer_parameter_groups=[{key: value for key, value in group.items() if key != "params"}
                                                   for group in optimizer.state_dict()["param_groups"]],
                       environment=environment(device), cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                       model_parameters=sum(parameter.numel() for parameter in model.parameters()),
                       model_layers=len(model.blocks), fractional_policy=fractional_policy())
        derived = override_config(config, plan, implementation)
        receipt.update(benchmark_override={"spectral": True, "dead_units": False, "penalty_step": plan["penalty_step"],
                                          "loss_implementation": implementation, "baseline_loss_source": derived["study"]["baseline_loss_source"]},
                       benchmark_config=derived, benchmark_config_sha256=config_hash(derived),
                       spectral_weights=training.penalty_weights(derived, 0), optimizer_policy=plan["optimizer_policy"],
                       sampler_override=plan["sampler_override"], timer_boundary=BOUNDARY,
                       setup_seconds_before_batches=time.perf_counter() - started)
        prepare_started = time.perf_counter()
        batches = reconstruct_batches(config, plan, generator, device)
        # Construction/import/preparation must not perturb the checkpoint's global streams.
        restore_rng_state(payload["rng_state"], device)
        receipt["prepared_state_sha256"] = state_hash(capture_state(model, optimizer, generator, device, payload, fired_sums))
        receipt["batch_manifest"] = plan["batch_manifest"]
        receipt["local_sampler_sha256_after_preparation"] = tensor_hash(generator.get_state())
        sync(device)
        receipt["batch_preparation_and_state_hash_seconds"] = time.perf_counter() - prepare_started
        memory_guard(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timing_started = time.perf_counter()
        for index in range(plan["warmup"] + plan["steps"]):
            sync(device)
            update_started = time.perf_counter()
            values = update(model, optimizer, batches[index * config["accum"]:(index + 1) * config["accum"]], derived, device)
            sync(device)
            seconds = time.perf_counter() - update_started
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Invalid measured duration")
            receipt["updates"].append({"index": index, "warmup": index < plan["warmup"], "seconds": seconds, **values})
            memory_guard(device)
        receipt["loop_wall_seconds_including_memory_guards"] = time.perf_counter() - timing_started
        ensure_finite_state({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        receipt["finite_checks_passed"] = True
        receipt["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        receipt["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None
        post_started = time.perf_counter()
        study.validate_config(config)
        final_inputs = observed(records)
        if final_inputs != records:
            raise ValueError("Bound input changed during worker or final validation")
        final_sources = sources()
        if final_sources != IMPORTED_SOURCES:
            raise ValueError("Prototype source changed during worker")
        receipt["postcheck_seconds"] = time.perf_counter() - post_started
        receipt["mean_seconds_per_update"] = statistics.mean(row["seconds"] for row in receipt["updates"] if not row["warmup"])
        receipt.update(complete=True, status="completed", freshness_checks_passed=True, inputs_before=records,
                       inputs_after=final_inputs, sources_after=final_sources, finished_unix=time.time(),
                       elapsed_seconds_before_receipt=time.perf_counter() - started,
                       plan=plan_record, budget={"run_id": run_id, "ledger": str(LEDGER.resolve())})
        immutable_json(output / "metrics.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(complete=False, status="failed", error=f"{type(exc).__name__}: {exc}",
                       inputs_before=records, inputs_after=observed(records), sources_after=observed(IMPORTED_SOURCES),
                       elapsed_seconds_before_receipt=time.perf_counter() - started)
        immutable_json(output / "error.json", receipt)
        raise


def worker_command(plan_path, implementation, output, run_id):
    return [sys.executable, str(SCRIPT), "_worker", "--plan", str(Path(plan_path).resolve()),
            "--implementation", implementation, "--output", str(output), "--run-id", run_id]


def audit_budget(run_id, command, output):
    with budget.ledger_lock(LEDGER):
        records = budget.read_ledger(LEDGER)
        run = budget.accounting(records).get(run_id)
        if (run is None or not run["settled"] or not run["launch_authorized"]
                or run["settlement"].get("status") != "completed" or run["settlement"].get("returncode") != 0):
            raise ValueError("Worker lacks a completed canonical budget settlement")
        directory = Path(run["reservation"]["directory"])
        result_record, provenance_record = file_record(directory / "result.json"), file_record(directory / "provenance.json")
        result, provenance = load_json(result_record["path"]), load_json(provenance_record["path"])
        if result.get("status") != "completed" or result.get("returncode") != 0 or result.get("logs_complete") is not True:
            raise ValueError("Canonical budget result is not a complete successful invocation")
        if (provenance.get("command") != budget.public_command(command) or provenance.get("cuda_visible_devices") != "0"
                or provenance.get("wrapper_sha256") != file_record(ROOT / "scripts/study_budget.py")["sha256"]):
            raise ValueError("Budget command or wrapper provenance differs")
        check(result_record)
        check(provenance_record)
        prefix = records[:run["settlement"]["sequence"] + 1]
        path = output / f"{run_id}-budget-prefix.jsonl"
        with path.open("xb") as handle:
            for record in prefix:
                handle.write(budget.canonical(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {"run_id": run_id, "reservation": run["reservation"], "settlement": run["settlement"],
            "result": result_record, "provenance": provenance_record, "ledger_prefix": file_record(path)}


def validate_pair(receipts, plan):
    if len(receipts) != 2 or [item["implementation"] for item in receipts] != list(ORDER[plan["order"]]):
        raise ValueError("Pair implementation order is incomplete or differs")
    for item in receipts:
        if (item.get("kind") != KIND or item.get("complete") is not True or item.get("finite_checks_passed") is not True
                or item.get("confirmatory_claim_permitted") is not False or item.get("freshness_checks_passed") is not True
                or item.get("numerical_acceptance") != "not_evaluated" or item.get("eligibility") != "not_evaluated"):
            raise ValueError("Worker did not finish its exploratory checks")
        rows = item["updates"]
        if len(rows) != plan["warmup"] + plan["steps"]:
            raise ValueError("Worker update population differs")
        for index, row in enumerate(rows):
            if (type(row["index"]) is not int or row["index"] != index
                    or row["warmup"] is not (index < plan["warmup"])
                    or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) for key in ("seconds", "loss", "ce"))
                    or row["seconds"] <= 0):
                raise ValueError("Worker update record is invalid")
        if item["batch_manifest"] != plan["batch_manifest"] or item["source_checkpoint"] != plan["checkpoint"]:
            raise ValueError("Worker batch/checkpoint differs from plan")
    if receipts[0]["execution_id"] == receipts[1]["execution_id"]:
        raise ValueError("Worker executions were not distinct")
    for key in ("restored_state_sha256", "prepared_state_sha256", "environment", "cudnn_allow_tf32",
                "optimizer_parameter_groups", "source_step", "spectral_weights", "timer_boundary",
                "local_sampler_sha256_before_override", "local_sampler_sha256_after_preparation"):
        if receipts[0][key] != receipts[1][key]:
            raise ValueError(f"Pair state or method differs: {key}")
    return {item["implementation"]: statistics.mean(row["seconds"] for row in item["updates"] if not row["warmup"])
            for item in receipts}


def run_pair(plan_path, reserve_seconds):
    if type(reserve_seconds) not in (int, float) or not math.isfinite(reserve_seconds) or not .5 <= reserve_seconds <= 600:
        raise ValueError("Explicit per-worker reservation must be in [0.5, 600] seconds")
    plan_path = Path(plan_path).resolve()
    output = plan_path.parent / "dispatch"
    output.mkdir(exist_ok=False)  # No implicit restart/retry, even after a failed dispatch.
    started = time.perf_counter()
    receipt = receipt_base()
    receipt.update(plan_requested=str(plan_path), reserve_seconds_per_worker=reserve_seconds, workers=[])
    immutable_json(output / "attempt.json", receipt)
    try:
        plan, plan_record = read_plan(plan_path)
        with budget.ledger_lock(LEDGER):
            cap, _approval = budget.approval(DECISIONS, LEDGER)
            runs = budget.accounting(budget.read_ledger(LEDGER))
            if any(not run["settled"] for run in runs.values()):
                raise ValueError("Unfinished budget reservation blocks pair dispatch")
            if sum(run["charge_seconds"] for run in runs.values()) + 2 * reserve_seconds > cap:
                raise ValueError("Remaining budget cannot cover both worker reservations")
            if any(f"pair-{plan['pair_id']}-{name}" in runs for name in ORDER[plan["order"]]):
                raise ValueError("Pair run IDs already exist; retries are not authorized")
        receipts, evidence = [], []
        for implementation in ORDER[plan["order"]]:
            check(plan_record)
            read_plan(plan_path)
            run_id = f"pair-{plan['pair_id']}-{implementation}"
            worker_output = output / implementation
            command = worker_command(plan_path, implementation, worker_output, run_id)
            immutable_json(output / f"launch-{implementation}.json", {
                **receipt_base(), "run_id": run_id, "plan": plan_record, "command": budget.public_command(command),
                "reserve_seconds": reserve_seconds, "status": "launch_intent_not_execution_evidence"})
            code = budget.run_study(argparse.Namespace(command=command, run_id=run_id, reserve_seconds=reserve_seconds,
                                                      cuda_device="0", ledger=LEDGER, decisions=DECISIONS))
            if code != 0:
                raise RuntimeError(f"Budgeted worker failed: {run_id}; pair cannot be promoted or retried")
            proof = audit_budget(run_id, command, output)
            metrics_path = worker_output / "metrics.json"
            metrics_record = file_record(metrics_path)
            sidecar_record = file_record(metrics_path.with_name("metrics.json.sha256"))
            worker_receipt = verified_json(metrics_path)
            check(metrics_record)
            check(sidecar_record)
            if worker_receipt.get("run_id") != run_id or worker_receipt.get("plan") != plan_record:
                raise ValueError("Worker receipt differs from dispatched run/plan")
            receipts.append(worker_receipt)
            evidence.append(proof)
            receipt["workers"].append({"metrics": metrics_record, "sidecar": sidecar_record,
                                       "budget": proof})
        if evidence[1]["reservation"]["sequence"] != evidence[0]["settlement"]["sequence"] + 1:
            raise ValueError("Another budget event intervened between pair workers")
        means = validate_pair(receipts, plan)
        check(plan_record)
        read_plan(plan_path)
        for item in receipt["workers"]:
            for record in (item["metrics"], item["sidecar"], item["budget"]["result"], item["budget"]["provenance"], item["budget"]["ledger_prefix"]):
                check(record)
        receipt.update(complete=True, status="completed", plan=plan_record, means_seconds_per_update=means,
                       descriptive_baseline_over_optimized=means["baseline"] / means["optimized"],
                       charged_seconds=sum(item["settlement"]["charge_seconds"] for item in evidence),
                       controller_elapsed_seconds=time.perf_counter() - started,
                       limitation="One exploratory pair; no CI, numerical equivalence, eligibility or confirmatory promotion")
        immutable_json(output / "metrics.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(complete=False, status="failed", error=f"{type(exc).__name__}: {exc}",
                       controller_elapsed_seconds=time.perf_counter() - started)
        immutable_json(output / "error.json", receipt)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--pair-id", required=True)
    prepare_parser.add_argument("--penalty-step", type=int, choices=(0, 135000), required=True)
    prepare_parser.add_argument("--batch-seed", type=int, required=True)
    prepare_parser.add_argument("--order", choices=tuple(ORDER), required=True)
    prepare_parser.add_argument("--baseline-source", type=Path)
    prepare_parser.add_argument("--warmup", type=int, default=3)
    prepare_parser.add_argument("--steps", type=int, default=12)
    run = sub.add_parser("run-pair")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--reserve-seconds-per-worker", type=float, required=True)
    child = sub.add_parser("_worker", help=argparse.SUPPRESS)
    child.add_argument("--plan", type=Path, required=True)
    child.add_argument("--implementation", choices=("baseline", "optimized"), required=True)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.checkpoint, args.output, pair_id=args.pair_id, penalty_step=args.penalty_step,
                         batch_seed=args.batch_seed, order=args.order, baseline_source=args.baseline_source,
                         warmup=args.warmup, steps=args.steps)
    elif args.command == "run-pair":
        result = run_pair(args.plan, args.reserve_seconds_per_worker)
    else:
        result = worker(args.plan, args.implementation, args.output, args.run_id)
    print(json.dumps({"complete": result["complete"], "kind": KIND, "confirmatory_claim_permitted": False}))


if __name__ == "__main__":
    main()
