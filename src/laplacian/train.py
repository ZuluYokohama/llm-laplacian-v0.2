"""Spectral-LLM training, the Arm 0 smoke, Arm 3 masks, and distillation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import warnings

import torch
import torch.nn.functional as F

from laplacian.arms import PULSE_STEPS, get_config, layers_0index
from laplacian.culling import cull_pulse, node_only_pulse, scrub_optimizer
from laplacian.data import DATA, TokenStore
from laplacian.device import enforce_guard, memory_mib, pick_5070
from laplacian.losses import spectral_penalties, total_loss
from laplacian.model import SpectralTransformer
from laplacian.runtime import RUNS, apply_env, read_json, write_metrics
from laplacian.reproducibility import (
    CHECKPOINT_VERSION, config_hash, ensure_finite_state, environment,
    fresh_metadata, restore_rng_state, rng_state, seed_everything,
)
from laplacian.schedule import gammas


def log_line(line):
    print(line, flush=True)
    path = RUNS / "heartbeat.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")


def learning_rate(step, total, warmup, peak, floor):
    if step < warmup:
        return peak * float(step + 1) / float(max(warmup, 1))
    progress = (step - warmup) / float(max(total - warmup, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return floor + (peak - floor) * cosine


def read_meta():
    path = DATA / "meta.json"
    if not path.exists():
        raise SystemExit("data/meta.json is missing. Run python -m laplacian.data first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(config, vocab, device):
    model = SpectralTransformer(
        vocab=vocab,
        m=int(config["m"]),
        k=int(config["k"]),
        context=int(config["context"]),
    )
    return model.to(device)


def run_dir(config):
    path = RUNS / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def already_done(config):
    path = run_dir(config) / "metrics.json"
    if not path.exists():
        return False
    return bool(read_json(path).get("complete"))


def penalty_weights(config, step):
    if config.get("manifold_constant") is not None:
        return {
            "manifold": float(config["manifold_constant"]),
            "fractional": 0.0,
            "hyper": 0.0,
            "dirichlet": 0.0,
        }
    if not config["spectral"]:
        return {"manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 0.0}
    fixed_step = config.get("study", {}).get("penalty_step")
    table = gammas(step if fixed_step is None else int(fixed_step))
    return {key: float(table[key]) for key in ("manifold", "fractional", "hyper", "dirichlet")}


def forward_loss(model, inputs, targets, config, step):
    outputs = model(inputs)
    logits = outputs["logits"].float()
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    weights = penalty_weights(config, step)
    if any(value != 0.0 for value in weights.values()):
        if config.get("study", {}).get("loss_implementation") == "baseline":
            from laplacian.study_train import baseline_penalties
            penalties = baseline_penalties(config, outputs, layers_0index(config), float(config["exponent"]))
        else:
            penalties = spectral_penalties(
                outputs, layers_0index(config), float(config["exponent"]), weights=weights
            )
        loss = total_loss(ce, penalties, weights)
    else:
        penalties = None
        loss = ce
    return loss, ce, outputs, penalties


def save_checkpoint(path, model, optimizer, generator, step, pulses_done, stripped, fired_sums=None, fired_tokens=0,
                    *, config=None, resume_metadata=None, elapsed_seconds_total=0.0,
                    last_loss=None, last_ce=None):
    metadata = dict(fresh_metadata() if resume_metadata is None else resume_metadata)
    metadata["elapsed_seconds_total"] = float(elapsed_seconds_total)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "pulses_done": list(pulses_done),
        "stripped": list(stripped),
        "fired_tokens": int(fired_tokens),
        "rng_state": rng_state(next(model.parameters()).device),
        "environment": environment(next(model.parameters()).device),
        "reproducibility": metadata,
        "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
        "module_training": {name: module.training for name, module in model.named_modules()},
        "last_loss": last_loss,
        "last_ce": last_ce,
    }
    if config is not None:
        payload["training_config"] = dict(config)
        payload["training_config_sha256"] = config_hash(config)
    if fired_sums is not None:
        payload["fired_sums"] = [item.detach().cpu() for item in fired_sums]
    ensure_finite_state(payload)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_weights(path, model, step):
    payload = {"step": step, "model": model.state_dict()}
    ensure_finite_state(payload)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_generator(generator, state):
    """The RNG state has to be a contiguous CPU byte tensor.

    torch.load(map_location=cuda) moves every saved tensor, including the
    generator state, onto the GPU. set_state then rejects it.
    """
    generator.set_state(state.detach().to(device="cpu").contiguous())


def load_checkpoint(path, model, optimizer, generator, device, *, config=None, resume_metadata=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if config is not None and "study" in config:
        from laplacian.study_train import validate_checkpoint
        validate_checkpoint(checkpoint, config, model=model)
    saved_hash = checkpoint.get("training_config_sha256")
    if saved_hash is not None and config is not None and saved_hash != config_hash(config):
        raise ValueError("Checkpoint training configuration differs; refusing a mismatched resume")
    ensure_finite_state(checkpoint)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_generator(generator, checkpoint["generator"])
    for name, parameter in model.named_parameters():
        if name in checkpoint.get("requires_grad", {}):
            parameter.requires_grad_(checkpoint["requires_grad"][name])
    for name, module in model.named_modules():
        if name in checkpoint.get("module_training", {}):
            module.training = checkpoint["module_training"][name]
        elif hasattr(module, "stripped") and int(module.stripped.item()) == 1:
            module.weight.requires_grad_(False)
    metadata = dict(checkpoint.get("reproducibility") or fresh_metadata())
    limitations = list(metadata.get("resume_limitations", []))
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION or "rng_state" not in checkpoint:
        limitations.append("legacy checkpoint lacks complete global RNG state; continuation is non-exact")
        metadata["elapsed_seconds_total"] = 0.0
        metadata["elapsed_seconds_complete"] = False
    elif not restore_rng_state(checkpoint["rng_state"], device):
        limitations.append("CPU/CUDA RNG domain changed; continuation is non-exact")
    if saved_hash is None or config is None:
        limitations.append("training configuration was not verified; continuation is non-exact")
    if checkpoint.get("environment") != environment(device):
        limitations.append("execution environment differs or was not recorded; bitwise replay is not established")
    metadata["resume_limitations"] = list(dict.fromkeys(limitations))
    metadata["resume_status"] = "non_exact" if limitations else "state_complete"
    metadata["last_loss"] = checkpoint.get("last_loss")
    metadata["last_ce"] = checkpoint.get("last_ce")
    if resume_metadata is not None:
        resume_metadata.update(metadata)
    if limitations:
        warnings.warn("Non-exact checkpoint continuation: " + "; ".join(metadata["resume_limitations"]), RuntimeWarning)
    raw_sums = checkpoint.get("fired_sums")
    fired_sums = None if raw_sums is None else [item.to(device) for item in raw_sums]
    fired_tokens = int(checkpoint.get("fired_tokens") or 0)
    return (
        int(checkpoint["step"]),
        list(checkpoint.get("pulses_done", [])),
        list(checkpoint.get("stripped", [])),
        fired_sums,
        fired_tokens,
    )


def finite_loss_values(loss, ce, step):
    """Check before backward/optimizer mutation, retaining the last good checkpoint."""
    loss_value, ce_value = float(loss.detach().item()), float(ce.detach().item())
    if not math.isfinite(loss_value) or not math.isfinite(ce_value):
        log_line(f"FAILED step={step} nonfinite_loss nan=1 checkpoint_preserved=1")
        raise FloatingPointError(f"Non-finite loss or CE before optimizer update at step {step}")
    return loss_value, ce_value


def checked_optimizer_step(model, optimizer, clip, step):
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip), error_if_nonfinite=True)
    except RuntimeError as exc:
        log_line(f"FAILED step={step} gradient_check nan=1 checkpoint_preserved=1")
        raise FloatingPointError(f"Gradient validation failed before optimizer update at step {step}") from exc
    optimizer.step()
    return norm


def _add_fired(fired_sums, fired_tokens, outputs):
    """Add one microbatch into the dead-unit window. fired is [batch, time, neurons]."""
    totals = [item.detach().float().sum(dim=(0, 1)) for item in outputs["fired"]]
    tokens = int(outputs["fired"][0].shape[0] * outputs["fired"][0].shape[1])
    if fired_sums is None:
        return totals, tokens
    combined = [previous + extra for previous, extra in zip(fired_sums, totals)]
    return combined, fired_tokens + tokens


def resample_dead(model, rates, threshold=5e-6):
    changed = 0
    for block, rate in zip(model.blocks, rates):
        dead = (block.mlp.neuron_mask > 0) & (rate < threshold)
        index = torch.where(dead)[0]
        if index.numel() == 0:
            continue
        changed += int(index.numel())
        # Advanced indexing returns a copy, so in-place init on the slice
        # would not reach the parameter.
        with torch.no_grad():
            up = torch.empty(
                index.numel(),
                block.mlp.up.weight.shape[1],
                device=block.mlp.up.weight.device,
                dtype=block.mlp.up.weight.dtype,
            )
            torch.nn.init.normal_(up, mean=0.0, std=0.02)
            block.mlp.up.weight.index_copy_(0, index, up)
            std = 0.02 / math.sqrt(2 * len(model.blocks))
            down = torch.empty(
                block.mlp.down.weight.shape[0],
                index.numel(),
                device=block.mlp.down.weight.device,
                dtype=block.mlp.down.weight.dtype,
            )
            torch.nn.init.normal_(down, mean=0.0, std=std)
            block.mlp.down.weight.index_copy_(1, index, down)
            block.mlp.up_mask.index_fill_(0, index, 1)
            block.mlp.down_mask.index_copy_(
                1,
                index,
                torch.ones(
                    block.mlp.down_mask.shape[0],
                    index.numel(),
                    device=block.mlp.down_mask.device,
                    dtype=block.mlp.down_mask.dtype,
                ),
            )
    return changed


def apply_strips(model, config, step, done):
    if not config["norm_removal"]:
        return done
    start = int(config["steps"] * 240_000 / 300_000)
    span = int(config["steps"] - start)
    points = [start + index * span // 5 for index in range(5)]
    updated = list(done)
    for index, point in enumerate(points):
        if step < point or index in updated:
            continue
        model.strip_layer(index)
        if index == 4:
            model.norm.strip()
        updated.append(index)
        log_line(f"PROGRESS norm_strip layer={index} step={step}")
    return updated


def calibration_batches(store, config, generator):
    produced = 0
    target = int(config["calibration_sequences"])
    while produced < target:
        inputs, _targets = store.batch(int(config["microbatch"]), generator)
        produced += int(inputs.shape[0])
        yield inputs


def run_pulse(model, optimizer, store, config, generator):
    started = time.perf_counter()
    reports = cull_pulse(
        model,
        calibration_batches(store, config, generator),
        int(config["calibration_sequences"]),
        float(config["pulse_fraction"]),
        float(config["eps"]),
        float(config["retain_floor"]),
        config["pulse_mode"],
        int(config["microbatch"]),
    )
    scrub_optimizer(model, optimizer)
    return reports, time.perf_counter() - started


def train(config):
    study_manifest = None
    if "study" in config:
        from laplacian.study_train import StudyTokenStore, completion_record, file_record, folder_for, validate_config
        study_manifest = validate_config(config)
        if folder_for(config) != (RUNS / config["run_id"]).resolve():
            raise ValueError("Study and trainer RUNS directories differ")
    if already_done(config):
        if study_manifest is not None:
            raise FileExistsError("Completed study runs are immutable; choose a new run_id")
        log_line(f"SKIP {config['run_id']} complete")
        return
    apply_env()
    seed_everything(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    device, name = pick_5070()
    if study_manifest is not None:
        if device.type == "cuda" and not all(os.environ.get(key) for key in ("STUDY_BUDGET_RUN_ID", "STUDY_BUDGET_LEDGER")):
            raise RuntimeError("CUDA study training must run through scripts/study_budget.py")
        meta = {"vocab": study_manifest["tokenizer"]["vocab_size"]}
        store = StudyTokenStore(config)
    else:
        meta = read_meta()
        store = TokenStore(DATA / "tinystories_train.u16", context=int(config["context"]))
    model = build_model(config, int(meta["vocab"]), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        betas=tuple(config["betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    folder = run_dir(config)
    step = 0
    pulses_done = []
    stripped = []
    fired_sums = None
    fired_tokens = 0
    resume_metadata = fresh_metadata()
    last = folder / "last.pt"
    if last.exists():
        step, pulses_done, stripped, fired_sums, fired_tokens = load_checkpoint(
            last, model, optimizer, generator, device, config=config, resume_metadata=resume_metadata
        )
        log_line(f"PROGRESS resume {config['run_id']} step={step}")
        if study_manifest is not None and resume_metadata["resume_limitations"]:
            raise ValueError("Study resume environment differs; exact state continuation is required")
    if study_manifest is not None:
        store.start_segment(folder, step)
    if config["pulse_only"]:
        _pulse_only(config, model, optimizer, store, generator, device, name, meta)
        return
    probe_seconds = None
    if int(config["spectral_probe_steps"]) > 0 and step == 0:
        probe_seconds = _spectral_probe(model, store, config, device, generator)
    peak = memory_mib()
    started = time.perf_counter()
    started_step = step
    elapsed_before = float(resume_metadata["elapsed_seconds_total"])
    last_ce = resume_metadata.get("last_ce")
    last_loss = resume_metadata.get("last_loss")
    total_steps = int(config["steps"])
    stride = max(total_steps // 20, 1)
    if study_manifest is not None and step in config["study"]["snapshot_steps"]:
        snapshot = folder / f"state_step{step:07d}.pt"
        if not snapshot.exists():
            save_checkpoint(snapshot, model, optimizer, generator, step, pulses_done, stripped, fired_sums, fired_tokens,
                            config=config, resume_metadata=resume_metadata, elapsed_seconds_total=elapsed_before,
                            last_loss=last_loss, last_ce=last_ce)
        if not last.exists():
            save_checkpoint(last, model, optimizer, generator, step, pulses_done, stripped, fired_sums, fired_tokens,
                            config=config, resume_metadata=resume_metadata, elapsed_seconds_total=elapsed_before,
                            last_loss=last_loss, last_ce=last_ce)
    if study_manifest is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        study_loop_started = time.perf_counter()
    while step < total_steps:
        stripped = apply_strips(model, config, step, stripped)
        rate = learning_rate(step, total_steps, int(config["warmup"]), float(config["lr"]), float(config["min_lr"]))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        loss_sum = 0.0
        outputs = None
        for _micro in range(int(config["accum"])):
            if study_manifest is not None:
                store.set_position(step, _micro)
            inputs, targets = store.batch(int(config["microbatch"]), generator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, ce, outputs, _penalties = forward_loss(model, inputs, targets, config, step)
            loss_value, ce_value = finite_loss_values(loss, ce, step)
            (loss / float(config["accum"])).backward()
            ce_sum += ce_value
            loss_sum += loss_value
            if config["dead_units"] and step < 30_000:
                fired_sums, fired_tokens = _add_fired(fired_sums, fired_tokens, outputs)
        checked_optimizer_step(model, optimizer, config["clip"], step)
        step += 1
        last_ce = ce_sum / float(config["accum"])
        last_loss = loss_sum / float(config["accum"])
        if config["dead_units"] and step < 30_000 and step % 1000 == 0 and fired_tokens > 0:
            window_tokens = fired_tokens
            rates = [total / float(window_tokens) for total in fired_sums]
            changed = resample_dead(model, rates)
            fired_sums = None
            fired_tokens = 0
            log_line(f"PROGRESS dead_resample step={step} neurons={changed} tokens={window_tokens}")
        if config["pulses"] and step in PULSE_STEPS and step not in pulses_done:
            reports, pulse_seconds = run_pulse(model, optimizer, store, config, generator)
            pulses_done.append(step)
            log_line(f"PROGRESS pulse step={step} seconds={pulse_seconds:.3f} layers={len(reports)}")
        if step % 10 == 0:
            used = memory_mib()
            peak = max(peak, used)
            log_line(
                f"STEP {step}/{total_steps} loss={last_loss:.6f} ce={last_ce:.6f} "
                f"vram_mb={used:.0f} nan=0"
            )
        study_snapshot = study_manifest is not None and step in config["study"]["snapshot_steps"]
        if step % config.get("study", {}).get("checkpoint_every", 1000) == 0 or study_snapshot:
            if study_manifest is not None:
                store.flush()
            save_checkpoint(
                last, model, optimizer, generator, step, pulses_done, stripped, fired_sums, fired_tokens,
                config=config, resume_metadata=resume_metadata,
                elapsed_seconds_total=elapsed_before + time.perf_counter() - started,
                last_loss=last_loss, last_ce=last_ce,
            )
        if study_snapshot:
            snapshot = folder / f"state_step{step:07d}.pt"
            if snapshot.exists():
                raise FileExistsError(f"Study timing snapshot already exists: {snapshot}")
            save_checkpoint(snapshot, model, optimizer, generator, step, pulses_done, stripped, fired_sums, fired_tokens,
                            config=config, resume_metadata=resume_metadata,
                            elapsed_seconds_total=elapsed_before + time.perf_counter() - started,
                            last_loss=last_loss, last_ce=last_ce)
        if step % stride == 0:
            save_weights(folder / f"weights_step{step:07d}.pt", model, step)
        if step % 10 == 0:
            enforce_guard()
    if study_manifest is not None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        study_loop_seconds = time.perf_counter() - study_loop_started
        study_completion = completion_record(config, folder, store)
    save_checkpoint(last, model, optimizer, generator, step, pulses_done, stripped, fired_sums, fired_tokens,
                    config=config, resume_metadata=resume_metadata,
                    elapsed_seconds_total=elapsed_before + time.perf_counter() - started,
                    last_loss=last_loss, last_ce=last_ce)
    save_weights(folder / "weights_final.pt", model, step)
    elapsed = time.perf_counter() - started
    payload = {
        "run_id": config["run_id"],
        "complete": True,
        "host_gpu": device.type == "cuda",
        "device_name": name,
        "steps": step,
        "seconds": elapsed,
        "seconds_scope": "current_process_training_segment",
        "segment_seconds": elapsed,
        "segment_steps": step - started_step,
        "resumed_from_step": started_step,
        "elapsed_seconds_total": elapsed_before + elapsed,
        "elapsed_seconds_complete": resume_metadata["elapsed_seconds_complete"],
        "reproducibility": {**resume_metadata, "elapsed_seconds_total": elapsed_before + elapsed,
                            "last_loss": last_loss, "last_ce": last_ce},
        "seconds_per_step": elapsed / max(step - started_step, 1),
        "tokens_per_step": int(config["microbatch"]) * int(config["accum"]) * int(config["context"]),
        "final_loss": last_loss,
        "final_ce": last_ce,
        "peak_vram_mib": peak,
        "pulses_done": pulses_done,
        "stripped_layers": stripped,
        "spectral_probe_seconds_per_step": probe_seconds,
        "config": config,
        "vocab": int(meta["vocab"]),
    }
    if study_manifest is not None:
        # Revalidate after final artifact writes, immediately before declaring success.
        validate_config(config)
        payload["study"] = study_completion
        payload["study"]["final_checkpoint"] = file_record(last)
        payload["study"]["final_weights"] = file_record(folder / "weights_final.pt")
        payload["study"]["timing"] = {
            "update_loop_wall_seconds": study_loop_seconds,
            "update_loop_steps": step - started_step,
            "update_loop_seconds_per_step": study_loop_seconds / (step - started_step) if step > started_step else None,
            "update_loop_scope": "synchronized boundaries; includes sampling/logging, update, in-loop checkpoints, exports and memory guards",
            "initial_snapshot_seconds": study_loop_started - started,
            "post_loop_validation_export_seconds": time.perf_counter() - study_loop_started - study_loop_seconds,
            "end_to_end_accounting": "budget wrapper wall time includes pre-loop manifest validation, model setup and teardown",
        }
        payload["study_budget"] = {key: os.environ.get(key) for key in ("STUDY_BUDGET_RUN_ID", "STUDY_BUDGET_LEDGER")}
    digest = write_metrics(folder / "metrics.json", payload)
    log_line(f"DONE {config['run_id']} sha256={digest} steps={step}")


def _spectral_probe(model, store, config, device, generator):
    probe_config = dict(config)
    probe_config["spectral"] = True
    clone = copy.deepcopy(model)
    opt = torch.optim.AdamW(clone.parameters(), lr=float(config["lr"]))
    steps = int(config["spectral_probe_steps"])
    started = time.perf_counter()
    for index in range(steps):
        opt.zero_grad(set_to_none=True)
        inputs, targets = store.batch(int(config["microbatch"]), generator)
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, ce, _outputs, _penalties = forward_loss(clone, inputs, targets, probe_config, 135000)
        finite_loss_values(loss, ce, index)
        loss.backward()
        checked_optimizer_step(clone, opt, float("inf"), index)
        if index == steps - 1:
            log_line(f"PROGRESS spectral_probe steps={steps}")
    del clone, opt
    torch.cuda.empty_cache()
    return (time.perf_counter() - started) / float(steps)


def _pulse_only(config, model, optimizer, store, generator, device, name, meta):
    started = time.perf_counter()
    reports, pulse_seconds = run_pulse(model, optimizer, store, config, generator)
    elapsed = time.perf_counter() - started
    payload = {
        "run_id": config["run_id"],
        "complete": True,
        "host_gpu": True,
        "device_name": name,
        "m": int(config["m"]),
        "k": int(config["k"]),
        "calibration_sequences": int(config["calibration_sequences"]),
        "seconds": elapsed,
        "pulse_seconds": pulse_seconds,
        "reports": reports,
        "peak_vram_mib": memory_mib(),
        "vocab": int(meta["vocab"]),
        "tokens_note": "one culling pulse on a freshly initialized M=8 model",
    }
    save_weights(run_dir(config) / "weights_final.pt", model, 0)
    digest = write_metrics(run_dir(config) / "metrics.json", payload)
    log_line(f"DONE {config['run_id']} sha256={digest} pulse_seconds={pulse_seconds:.3f}")


def _activation_dirichlet(hidden, affinity):
    # hidden [T, N], affinity [T, T]. Score is h^T L h per neuron.
    adjacency = 0.5 * (affinity + affinity.transpose(0, 1))
    adjacency = adjacency - torch.diag(torch.diag(adjacency))
    degree = adjacency.sum(dim=1)
    lap_h = degree[:, None] * hidden - adjacency @ hidden
    return (hidden * lap_h).sum(dim=0)


def run_masks(config):
    if already_done(config):
        log_line("SKIP arm3_masks complete")
        return
    apply_env()
    seed_everything(config["seed"])
    device, name = pick_5070()
    meta = read_meta()
    source = RUNS / "arm3" / "weights_final.pt"
    if not source.exists():
        raise SystemExit("arm3 weights_final.pt is missing")
    store = TokenStore(DATA / "tinystories_val.u16", context=int(config["context"]))
    train_store = TokenStore(DATA / "tinystories_train.u16", context=int(config["context"]))
    base = torch.load(source, map_location=device, weights_only=False)["model"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    variants = []

    def fresh():
        model = build_model(config, int(meta["vocab"]), device)
        model.load_state_dict(base)
        model.eval()
        return model

    def score(model, label):
        scorer = torch.Generator(device="cpu")
        scorer.manual_seed(1)
        ce_total = 0.0
        batches = 32
        with torch.no_grad():
            for _batch in range(batches):
                inputs, targets = store.batch(int(config["microbatch"]), scorer)
                inputs = inputs.to(device)
                targets = targets.to(device)
                _loss, ce, _outputs, _penalties = forward_loss(model, inputs, targets, config, 0)
                ce_total += float(ce.item())
        value = ce_total / batches
        log_line(f"PROGRESS mask {label} val_ce={value:.6f}")
        return {"name": label, "val_ce": value, "val_batches": batches, "val_microbatch": int(config["microbatch"])}

    variants.append(score(fresh(), "uncut"))
    for fraction in (0.20, 0.30, 0.40):
        model = fresh()
        node_only_pulse(
            model,
            calibration_batches(train_store, config, generator),
            int(config["calibration_sequences"]),
            fraction,
        )
        variants.append(score(model, f"aei_nodes_{int(fraction * 100)}"))
    combined = fresh()
    combined_config = dict(config)
    combined_config["pulse_fraction"] = 0.40
    combined_config["pulse_mode"] = "topk"
    combined_config["retain_floor"] = 0.25
    reports, _seconds = run_pulse(combined, None, train_store, combined_config, generator)
    row = score(combined, "aei40_jl75_suzuki")
    row["pulse_reports"] = reports
    variants.append(row)
    magnitude = fresh()
    with torch.no_grad():
        for block in magnitude.blocks:
            score_n = block.mlp.down.weight.detach().norm(dim=0)
            quota = max(1, int(math.floor(block.mlp.hidden * 0.40)))
            remove = torch.topk(score_n, quota, largest=False).indices
            block.mlp.neuron_mask[remove] = 0
    variants.append(score(magnitude, "magnitude_lowest_40"))
    quadratic = fresh()
    inputs, _targets = train_store.batch(1, generator)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = quadratic(inputs)
        for index, block in enumerate(quadratic.blocks):
            affinity = outputs["attn"][index][0].float().mean(dim=0)
            hidden = outputs["hidden"][index][0].float()
            neuron_score = _activation_dirichlet(hidden, affinity)
            quota = max(1, int(math.floor(block.mlp.hidden * 0.40)))
            remove = torch.topk(neuron_score, quota, largest=False).indices
            block.mlp.neuron_mask[remove] = 0
    variants.append(score(quadratic, "quadratic_dirichlet_lowest_40"))
    payload = {
        "run_id": "arm3_masks",
        "complete": True,
        "host_gpu": True,
        "device_name": name,
        "source": "runs/arm3/weights_final.pt",
        "no_finetune": True,
        "variants": variants,
        "quadratic_definition": "per neuron h^T L h on one validation-length attention affinity; drop the lowest 40%",
        "magnitude_definition": "L2 norm of the down-projection column; drop the lowest 40%",
    }
    digest = write_metrics(run_dir(config) / "metrics.json", payload)
    log_line(f"DONE arm3_masks sha256={digest}")


def run_distill(config):
    if already_done(config):
        log_line(f"SKIP {config['run_id']} complete")
        return
    apply_env()
    seed_everything(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    device, name = pick_5070()
    meta = read_meta()
    teacher_path = RUNS / config["teacher"] / "weights_final.pt"
    if not teacher_path.exists():
        raise SystemExit(f"missing teacher checkpoint {teacher_path}")
    teacher_cfg = get_config(config["teacher"])
    teacher = build_model(teacher_cfg, int(meta["vocab"]), device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=False)["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student_cfg = dict(config)
    student_cfg["spectral"] = config.get("manifold_constant") is not None
    model = build_model(config, int(meta["vocab"]), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        betas=tuple(config["betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    store = TokenStore(DATA / "tinystories_train.u16", context=int(config["context"]))
    folder = run_dir(config)
    step = 0
    resume_metadata = fresh_metadata()
    last = folder / "last.pt"
    if last.exists():
        step, _pulses, _stripped, _fired_sums, _fired_tokens = load_checkpoint(
            last, model, optimizer, generator, device, config=config, resume_metadata=resume_metadata
        )
    total_steps = int(config["steps"])
    started = time.perf_counter()
    started_step = step
    elapsed_before = float(resume_metadata["elapsed_seconds_total"])
    peak = memory_mib()
    last_loss = resume_metadata.get("last_loss")
    last_ce = resume_metadata.get("last_ce")
    while step < total_steps:
        rate = learning_rate(step, total_steps, int(config["warmup"]), float(config["lr"]), float(config["min_lr"]))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        ce_sum = 0.0
        for _micro in range(int(config["accum"])):
            inputs, targets = store.batch(int(config["microbatch"]), generator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                student_loss, ce, outputs, _penalties = forward_loss(model, inputs, targets, student_cfg, step)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)["logits"].float()
                student_log = F.log_softmax(outputs["logits"].float(), dim=-1)
                teacher_log = F.log_softmax(teacher_logits, dim=-1)
                kl = torch.sum(torch.exp(teacher_log) * (teacher_log - student_log), dim=-1).mean()
                loss = student_loss + kl
            loss_value, ce_value = finite_loss_values(loss, ce, step)
            (loss / float(config["accum"])).backward()
            loss_sum += loss_value
            ce_sum += ce_value
        checked_optimizer_step(model, optimizer, config["clip"], step)
        step += 1
        last_loss = loss_sum / float(config["accum"])
        last_ce = ce_sum / float(config["accum"])
        if step % 10 == 0:
            used = memory_mib()
            peak = max(peak, used)
            log_line(
                f"STEP {step}/{total_steps} loss={last_loss:.6f} ce={last_ce:.6f} "
                f"vram_mb={used:.0f} nan=0"
            )
        if step % 1000 == 0:
            save_checkpoint(last, model, optimizer, generator, step, [], [],
                            config=config, resume_metadata=resume_metadata,
                            elapsed_seconds_total=elapsed_before + time.perf_counter() - started,
                            last_loss=last_loss, last_ce=last_ce)
        if step % 10 == 0:
            enforce_guard()
    save_checkpoint(last, model, optimizer, generator, step, [], [],
                    config=config, resume_metadata=resume_metadata,
                    elapsed_seconds_total=elapsed_before + time.perf_counter() - started,
                    last_loss=last_loss, last_ce=last_ce)
    save_weights(folder / "weights_final.pt", model, step)
    elapsed = time.perf_counter() - started
    payload = {
        "run_id": config["run_id"],
        "complete": True,
        "host_gpu": True,
        "device_name": name,
        "teacher": config["teacher"],
        "kl_direction": "KL(teacher || student) added to the student loss",
        "manifold_constant": config.get("manifold_constant"),
        "steps": step,
        "seconds": elapsed,
        "seconds_scope": "current_process_training_segment",
        "segment_seconds": elapsed,
        "segment_steps": step - started_step,
        "resumed_from_step": started_step,
        "elapsed_seconds_total": elapsed_before + elapsed,
        "elapsed_seconds_complete": resume_metadata["elapsed_seconds_complete"],
        "reproducibility": {**resume_metadata, "elapsed_seconds_total": elapsed_before + elapsed,
                            "last_loss": last_loss, "last_ce": last_ce},
        "seconds_per_step": elapsed / max(step - started_step, 1),
        "tokens_per_step": int(config["microbatch"]) * int(config["accum"]) * int(config["context"]),
        "final_loss": last_loss,
        "final_ce": last_ce,
        "peak_vram_mib": peak,
        "config": config,
    }
    digest = write_metrics(folder / "metrics.json", payload)
    log_line(f"DONE {config['run_id']} sha256={digest}")


def write_eta():
    arm = read_json(RUNS / "smoke_arm0" / "metrics.json")
    pulse = read_json(RUNS / "smoke_pulse" / "metrics.json")
    plain = float(arm["seconds_per_step"])
    spectral = float(arm["spectral_probe_seconds_per_step"])
    pulse_seconds = float(pulse["pulse_seconds"])
    plain_runs = ["arm0", "arm1", "arm3"]
    spectral_runs = ["arm2", "arm4", "scale_m2", "scale_m4", "scale_m16"]
    spectral_runs.extend(
        [
            f"grid_s{str(exponent).replace('.', 'p')}_l{''.join(str(layer) for layer in layers)}"
            for exponent in (0.25, 0.5, 0.75)
            for layers in ([1, 5], [2, 5], [1, 3, 5])
            if not (exponent == 0.5 and layers == [2, 5])
        ]
    )
    days = (len(plain_runs) * 300_000 * plain + len(spectral_runs) * 300_000 * spectral) / 86400.0
    days += (2 * 30_000 * spectral) / 86400.0
    days += (7 * len(["arm4", "scale_m2", "scale_m4", "scale_m16"]) * pulse_seconds) / 86400.0
    text = "\n".join(
        [
            "# ETA from the Arm 0 smoke and one M=8 pulse",
            "",
            f"- Plain M=4 seconds/step: {plain:.6f}",
            f"- Spectral M=4 seconds/step at the step-135000 weights, {arm['spectral_probe_seconds_per_step']} probe: {spectral:.6f}",
            f"- One M=8 culling pulse, calibration {pulse['calibration_sequences']} sequences: {pulse_seconds:.3f} seconds",
            f"- Plain 300k runs counted at the plain rate: {', '.join(plain_runs)}",
            f"- Spectral 300k runs counted at the spectral rate: {', '.join(spectral_runs)}",
            "- Arm 1 is M=32 and Arm 5 is SmolLM2-135M. Neither rate was measured here, so both are outside this projection.",
            f"- Projection of the counted TinyStories runs plus seven pulses on the four full-protocol widths that are not aliases: {days:.2f} days",
            "- Aliases grid s=0.5 layers {2,5} and scale M=8 are not counted twice.",
            "",
        ]
    )
    path = RUNS / "ETA.md"
    path.write_text(text, encoding="utf-8")
    payload = {
        "run_id": "eta",
        "complete": True,
        "host_gpu": False,
        "plain_seconds_per_step": plain,
        "spectral_seconds_per_step": spectral,
        "pulse_seconds": pulse_seconds,
        "projected_days_counted_runs": days,
        "caveat": "M=32 and SmolLM2-135M are not in the smoke rates",
    }
    digest = write_metrics(RUNS / "eta" / "metrics.json", payload)
    log_line(f"DONE eta sha256={digest} projected_days={days:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    if args.arm == "write_eta":
        write_eta()
        return
    config = get_config(args.arm)
    if config["masks"]:
        run_masks(config)
    elif config["teacher"]:
        run_distill(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
