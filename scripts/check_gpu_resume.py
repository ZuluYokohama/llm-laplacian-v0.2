"""Short RTX 5070/BF16 checkpoint replay gate; synthetic data, no quality claim.

Run only through the approved budget wrapper. The 60-second cooperative deadline
does not interrupt an in-flight CUDA call; the wrapper must impose a hard limit.
This reconstructs model/optimizer objects in one process, not a process restart.
No tolerance relaxation or automatic deterministic-mode retry is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "pyproject.toml"]
    paths.extend(sorted((ROOT / "src/laplacian").glob("*.py")))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


# Capture before importing the local training implementation in the CLI process.
IMPORTED_SOURCE_HASHES = source_hashes()

from laplacian import train as training
from laplacian.arms import get_config
from laplacian.model import SpectralTransformer
from laplacian.operators_torch import fractional_policy
from laplacian.reproducibility import (
    config_hash, ensure_finite_state, environment, fresh_metadata, rng_state, seed_everything,
)
from laplacian.runtime import write_metrics

SPECTRAL_STEP = 135_000
TOTAL_UPDATES = 4
INTERRUPT_AFTER = 2
RESAMPLE_AFTER = 3


def cpu_snapshot(value):
    """Own all snapshot storage: later optimizer/model mutation cannot change it."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def compare_state(expected, actual, *, label="state", max_examples=20):
    """Exact nested comparison, with finite maximum differences for diagnosis."""
    result = {"label": label, "exact": True, "mismatch_count": 0, "examples": [],
              "tensor_count": 0, "array_count": 0, "max_abs_difference": 0.0}

    def mismatch(path, reason, difference=None):
        result["exact"] = False
        result["mismatch_count"] += 1
        if difference is not None:
            result["max_abs_difference"] = max(result["max_abs_difference"], difference)
        if len(result["examples"]) < max_examples:
            row = {"path": path, "reason": reason}
            if difference is not None:
                row["max_abs_difference"] = difference
            result["examples"].append(row)

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            result["tensor_count"] += 1
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                mismatch(path, "tensor type, shape, or dtype differs")
                return
            left, right = left.detach().cpu(), right.detach().cpu()
            if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
                mismatch(path, "nonfinite tensor")
                return
            if not torch.equal(left, right):
                # Inputs here are real model/optimizer/RNG tensors, including bytes.
                difference = float((left.to(torch.float64)-right.to(torch.float64)).abs().max()) if left.numel() else 0.0
                mismatch(path, "tensor values differ", difference)
        elif isinstance(left, np.ndarray):
            result["array_count"] += 1
            if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
                mismatch(path, "array type, shape, or dtype differs")
            elif not np.isfinite(left).all() or not np.isfinite(right).all():
                mismatch(path, "nonfinite array")
            elif not np.array_equal(left, right):
                difference = float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)))) if left.size else 0.0
                mismatch(path, "array values differ", difference)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                mismatch(path, "dictionary keys differ")
                return
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            if type(right) is not type(left) or len(left) != len(right):
                mismatch(path, "sequence type or length differs")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{index}]")
        elif type(left) is not type(right):
            mismatch(path, "scalar type differs")
        elif isinstance(left, float) and (not math.isfinite(left) or not math.isfinite(right)):
            mismatch(path, "nonfinite scalar")
        elif left != right:
            difference = abs(left-right) if isinstance(left, (int, float)) and not isinstance(left, bool) else None
            mismatch(path, "scalar values differ", float(difference) if difference is not None else None)

    visit(expected, actual, label)
    return result


def gate_config(seed):
    config = get_config("arm2")
    config.update(run_id="gpu-resume-gate", seed=seed, steps=TOTAL_UPDATES,
                  m=4, k=0, context=16, microbatch=8, accum=1, pulses=False,
                  spectral_probe_steps=0, dead_units=True)
    # No shortened schedule: every forward explicitly uses SPECTRAL_STEP.
    config["gate_fixed_forward_step"] = SPECTRAL_STEP
    config["gate_forced_resample_after"] = RESAMPLE_AFTER
    config["gate_vocab"] = 128
    return config


def require_gpu():
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Gate requires the approved RTX 5070 as visible CUDA device zero")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The approved device must support BF16 autocast")
    return device


def new_training_state(config, device):
    seed_everything(config["seed"])
    model = SpectralTransformer(vocab=128, m=4, k=0, d_model=256, n_layers=5,
                                context=config["context"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                 betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    return model, optimizer, generator


def synthetic_batch(config, generator, device):
    tokens = torch.randint(0, 128, (config["microbatch"], config["context"]+1), generator=generator)
    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device), digest


def random_probe(device):
    # Exercise every checkpoint RNG domain, even where this tiny model normally
    # consumes only CPU initialization, a data generator, and CUDA resampling.
    return {"python": random.random(), "numpy": float(np.random.random()),
            "torch_cpu": float(torch.rand(())), "torch_cuda": float(torch.rand((), device=device))}


def final_snapshot(model, optimizer, generator, fired_sums, fired_tokens, device):
    result = {
        "model_and_buffers": model.state_dict(), "optimizer": optimizer.state_dict(),
        "data_generator": generator.get_state(), "global_rng": rng_state(device),
        "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
        "module_training": {name: module.training for name, module in model.named_modules()},
        "fired_sums": fired_sums, "fired_tokens": fired_tokens,
    }
    ensure_finite_state(result)
    return cpu_snapshot(result)


def run_updates(model, optimizer, generator, config, device, start, end, checkpoint,
                traces, deadline, started, *, fired_sums=None, fired_tokens=0, metadata=None):
    metadata = fresh_metadata() if metadata is None else metadata
    for update in range(start+1, end+1):
        deadline()
        optimizer.zero_grad(set_to_none=True)
        inputs, targets, batch_hash = synthetic_batch(config, generator, device)
        probes = random_probe(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, ce, outputs, penalties = training.forward_loss(model, inputs, targets, config, SPECTRAL_STEP)
        loss_value, ce_value = training.finite_loss_values(loss, ce, update)
        ensure_finite_state(penalties, "component_penalties")
        loss.backward()
        gradient_norm = training.checked_optimizer_step(model, optimizer, config["clip"], update)
        fired_sums, fired_tokens = training._add_fired(fired_sums, fired_tokens, outputs)
        resampled = 0
        if update == RESAMPLE_AFTER:
            rates = [torch.zeros_like(block.mlp.neuron_mask) for block in model.blocks]
            resampled = training.resample_dead(model, rates)
            expected = sum(int(block.mlp.neuron_mask.sum()) for block in model.blocks)
            if resampled != expected or resampled == 0:
                raise RuntimeError("Forced real dead-unit resampling did not update every live neuron")
            fired_sums, fired_tokens = None, 0
        # Production save_checkpoint checks all model/optimizer state before
        # atomic replacement; a nonfinite update cannot replace the last good file.
        training.save_checkpoint(checkpoint, model, optimizer, generator, update, [], [],
                                 fired_sums, fired_tokens, config=config, resume_metadata=metadata,
                                 elapsed_seconds_total=time.perf_counter()-started,
                                 last_loss=loss_value, last_ce=ce_value)
        torch.cuda.synchronize(device)
        traces.append({"update": update, "batch_sha256": batch_hash, "rng_probe": probes,
                       "loss": loss_value, "cross_entropy": ce_value,
                       "gradient_norm": float(gradient_norm), "resampled_neurons": resampled,
                       "penalties": {key: cpu_snapshot(value).tolist() if isinstance(value, torch.Tensor) else value
                                     for key, value in penalties.items()}})
        del loss, ce, outputs, penalties, inputs, targets
        deadline()
    return fired_sums, fired_tokens


def run_gate(output, *, seed=1729, max_seconds=60.0):
    if not math.isfinite(max_seconds) or max_seconds <= 0 or max_seconds > 60:
        raise ValueError("max_seconds must be finite and in (0, 60]; use the wrapper for a hard timeout")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    payload = {
        "complete": False, "status": "running", "run_id": output.name, "host_gpu": False,
        "scope": "Synthetic same-process CUDA/BF16 object-reconstruction replay; no quality inference",
        "precision_policy": "Production TF32 matmul enabled, BF16 autocast; no automatic determinism change",
        "comparison_policy": "Exact values, shapes and dtypes; atol=0, rtol=0; no automatic retry",
        "source_sha256_before": IMPORTED_SOURCE_HASHES, "fractional_policy": fractional_policy(),
        "max_seconds_cooperative": max_seconds, "seed": seed,
        "updates_per_path": TOTAL_UPDATES, "interrupt_after": INTERRUPT_AFTER,
        "forced_resample_after": RESAMPLE_AFTER,
        "synthetic_data": {"vocab": 128, "microbatch": 8, "context": 16,
                           "sampling": "CPU generator randint, next-token targets; no external data"},
        "model": {"layers": 5, "width": 256, "mlp_multiplier": 4},
        "comparisons": [], "traces": {"uninterrupted": [], "resumed": []},
        "limitations": ["Single seed, four updates, one process/device; no cross-process or cross-hardware claim",
                        "No culling boundary; real dead-unit resampling is forced after update three",
                        "RNG probes intentionally exercise all saved domains in addition to model/data operations"],
    }
    failure = None

    def deadline():
        if time.perf_counter()-started > max_seconds:
            raise TimeoutError("GPU resume gate exceeded its cooperative time limit; no success promoted")

    try:
        if source_hashes() != IMPORTED_SOURCE_HASHES:
            raise RuntimeError("Source changed after import; restart the gate from stable sources")
        device = require_gpu()
        payload["host_gpu"] = True
        torch.backends.cuda.matmul.allow_tf32 = True
        payload["environment"] = environment(device)
        payload["environment"]["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        config = gate_config(seed)
        payload["training_config"] = config
        payload["training_config_sha256"] = config_hash(config)
        payload["fixed_spectral_step"] = SPECTRAL_STEP
        payload["spectral_weights"] = training.penalty_weights(config, SPECTRAL_STEP)
        if not all(weight > 0 for weight in payload["spectral_weights"].values()):
            raise RuntimeError("The replay gate requires all four spectral penalties to be active")
        payload["stage"] = "uninterrupted"
        model, optimizer, generator = new_training_state(config, device)
        sums, tokens = run_updates(model, optimizer, generator, config, device, 0, TOTAL_UPDATES,
                                  output / "uninterrupted-last.pt", payload["traces"]["uninterrupted"],
                                  deadline, started)
        expected = final_snapshot(model, optimizer, generator, sums, tokens, device)
        del model, optimizer, generator, sums
        gc.collect()

        payload["stage"] = "before_restart"
        model, optimizer, generator = new_training_state(config, device)
        sums, tokens = run_updates(model, optimizer, generator, config, device, 0, INTERRUPT_AFTER,
                                  output / "resumed-last.pt", payload["traces"]["resumed"], deadline, started)
        boundary_expected = final_snapshot(model, optimizer, generator, sums, tokens, device)
        boundary_checkpoint = output / "resume-at-step2.pt"
        shutil.copyfile(output / "resumed-last.pt", boundary_checkpoint)
        del model, optimizer, generator, sums
        gc.collect()

        payload["stage"] = "reconstruction_and_load"
        model, optimizer, generator = new_training_state({**config, "seed": seed+1}, device)
        random_probe(device)  # Deliberately disturb all global streams before restoration.
        metadata = {}
        restored_step, pulses, stripped, sums, tokens = training.load_checkpoint(
            boundary_checkpoint, model, optimizer, generator, device,
            config=config, resume_metadata=metadata)
        payload["resume_metadata"] = metadata
        if restored_step != INTERRUPT_AFTER or pulses or stripped or metadata["resume_status"] != "state_complete":
            raise RuntimeError("Checkpoint did not restore the expected complete replay state")
        restored = final_snapshot(model, optimizer, generator, sums, tokens, device)
        payload["comparisons"].append(compare_state(boundary_expected, restored, label="checkpoint_roundtrip"))
        del boundary_expected, restored
        if not payload["comparisons"][-1]["exact"]:
            raise RuntimeError("Checkpoint restoration differs before the resumed updates")

        payload["stage"] = "after_restart"
        sums, tokens = run_updates(model, optimizer, generator, config, device, restored_step, TOTAL_UPDATES,
                                  output / "resumed-last.pt", payload["traces"]["resumed"], deadline, started,
                                  fired_sums=sums, fired_tokens=tokens, metadata=metadata)
        actual = final_snapshot(model, optimizer, generator, sums, tokens, device)
        for key in expected:
            payload["comparisons"].append(compare_state(expected[key], actual[key], label=key))
        payload["comparisons"].append(compare_state(payload["traces"]["uninterrupted"],
                                                    payload["traces"]["resumed"], label="update_traces"))
        payload["max_abs_difference"] = max(row["max_abs_difference"] for row in payload["comparisons"])
        payload["checkpoint_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.pt")}
        if not all(row["exact"] for row in payload["comparisons"]):
            raise RuntimeError("Exact CUDA replay failed; inspect differences before choosing a deterministic policy")
        deadline()
        payload["complete"] = True
        payload["status"] = "passed"
        payload["stage"] = "finished"
    except BaseException as exc:
        failure = exc
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            payload["source_sha256_after"] = source_hashes()
            payload["sources_unchanged"] = payload["source_sha256_after"] == IMPORTED_SOURCE_HASHES
            if not payload["sources_unchanged"]:
                payload["complete"] = False
                payload["status"] = "failed"
                source_error = "Source changed during GPU replay gate"
                payload["error"] = payload.get("error", "") + ("; " if "error" in payload else "") + source_error
                if failure is None:
                    failure = RuntimeError(source_error)
        except Exception as exc:
            payload["complete"] = False
            payload["status"] = "failed"
            payload["source_check_error"] = f"{type(exc).__name__}: {exc}"
            if failure is None:
                failure = exc
        payload["seconds"] = time.perf_counter()-started
        payload["max_abs_difference"] = max((row["max_abs_difference"] for row in payload["comparisons"]), default=0.0)
        write_metrics(output / "metrics.json", payload)
    if failure is not None:
        raise failure
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    report = run_gate(args.output, seed=args.seed, max_seconds=args.max_seconds)
    print(json.dumps({"complete": report["complete"], "status": report["status"],
                      "max_abs_difference": report["max_abs_difference"], "seconds": report["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
