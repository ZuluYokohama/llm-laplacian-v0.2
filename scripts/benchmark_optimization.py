"""Bounded, source-pinned comparison of old and current spectral training losses.

This is a host benchmark, not a Jovin isolated receipt or a quality experiment.
It reads local token windows and optional trusted local weights, and never writes
checkpoints or resumes/overwrites an existing experiment's metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from laplacian import losses as candidate
from laplacian.arms import get_config, layers_0index
from laplacian.data import DATA, TokenStore
from laplacian.device import GUARD_MIB
from laplacian.model import SpectralTransformer
from laplacian.operators_torch import fractional_policy
from laplacian.runtime import write_metrics
from laplacian.train import penalty_weights


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources():
    paths = sorted((ROOT / "src" / "laplacian").glob("*.py")) + [Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("optimization_baseline_losses", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import baseline source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def forward_loss(model, batch, module, config, phase, optimized, device):
    inputs, targets = batch
    with autocast(device):
        output = model(inputs)
        logits = output["logits"].float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        weights = penalty_weights(config, phase)
        options = {"weights": weights} if optimized else {}
        penalties = module.spectral_penalties(output, layers_0index(config), config["exponent"], **options)
        loss = module.total_loss(ce, penalties, weights)
    return loss, ce


def capture_gradients(model, batches, module, config, phase, optimized, device):
    model.zero_grad(set_to_none=True)
    losses, ces = [], []
    for batch in batches:
        loss, ce = forward_loss(model, batch, module, config, phase, optimized, device)
        (loss / len(batches)).backward()
        losses.append(float(loss.detach()))
        ces.append(float(ce.detach()))
    gradients = {name: None if param.grad is None else param.grad.detach().cpu().clone()
                 for name, param in model.named_parameters()}
    return {"loss": statistics.mean(losses), "ce": statistics.mean(ces)}, gradients


def compare_gradients(reference, actual, loss_reference, loss_actual, *, loss_rtol, grad_rtol, atol):
    """Check every gradient tensor; report global and worst absolute error.

    Relative L2 uses all parameter elements, avoiding unstable elementwise
    relative errors at zero. A global max-absolute gate catches localized drift.
    """
    finite = all(math.isfinite(v) for v in (*loss_reference.values(), *loss_actual.values()))
    matching = set(reference) == set(actual)
    squared_error = squared_reference = 0.0
    max_error = max_reference = 0.0
    worst_parameter = None
    missing = []
    for name in sorted(set(reference) | set(actual)):
        left, right = reference.get(name), actual.get(name)
        if left is None or right is None:
            if left is not None or right is not None:
                matching = False
                missing.append(name)
            continue
        if left.shape != right.shape:
            matching = False
            missing.append(name)
            continue
        valid = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        finite = finite and valid
        if not valid:
            continue
        difference = (left - right).double()
        local_max = float(difference.abs().max())
        if local_max > max_error:
            max_error, worst_parameter = local_max, name
        max_reference = max(max_reference, float(left.abs().max()))
        squared_error += float(difference.square().sum())
        squared_reference += float(left.double().square().sum())
    relative_l2 = math.sqrt(squared_error) / max(math.sqrt(squared_reference), 1e-12)
    loss_error = abs(loss_actual["loss"] - loss_reference["loss"])
    loss_ok = loss_error <= atol + loss_rtol * abs(loss_reference["loss"])
    grad_ok = relative_l2 <= grad_rtol and max_error <= atol + grad_rtol * max_reference
    return {
        "passed": bool(finite and matching and loss_ok and grad_ok),
        "finite": finite, "matching_gradients": matching,
        "missing_or_shape_mismatch": missing,
        "baseline": loss_reference, "candidate": loss_actual,
        "loss_absolute_error": loss_error, "gradient_relative_l2": relative_l2,
        "gradient_max_absolute_error": max_error,
        "baseline_gradient_max_absolute": max_reference,
        "worst_parameter": worst_parameter,
        "tolerances": {"loss_relative": loss_rtol, "gradient_relative": grad_rtol, "absolute": atol},
    }


def memory_guard(device, ceiling_mib):
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    used = (total - free) / 1024**2
    if used > ceiling_mib:
        raise RuntimeError(f"CUDA device usage {used:.0f} MiB exceeds guard {ceiling_mib:.0f} MiB")


def measure(model, batches, module, config, phase, optimized, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                  weight_decay=config["weight_decay"])

    def step(index):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for micro in range(args.accum):
            batch = batches[(index * args.accum + micro) % len(batches)]
            loss, ce = forward_loss(model, batch, module, config, phase, optimized, device)
            (loss / args.accum).backward()
            values.append(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"], error_if_nonfinite=True)
        optimizer.step()
        value = float(torch.stack(values).mean())
        if not math.isfinite(value) or not bool(torch.isfinite(norm)):
            raise RuntimeError("Non-finite loss or gradients in benchmark")
        return value

    for index in range(args.warmup):
        step(index)
    sync(device)
    memory_guard(device, args.memory_limit_mib)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples, final_loss = [], None
    for index in range(args.steps):
        sync(device)
        started = time.perf_counter()
        final_loss = step(index + args.warmup)
        sync(device)
        samples.append(time.perf_counter() - started)
        memory_guard(device, args.memory_limit_mib)
    median = statistics.median(samples)
    return {
        "seconds_per_step_samples": samples, "median_seconds_per_step": median,
        "mean_seconds_per_step": statistics.mean(samples),
        "tokens_per_second": args.microbatch * args.context * args.accum / median,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None,
        "final_loss": final_loss, "finite": True,
    }


def git_info():
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "status_porcelain": run("status", "--porcelain")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--phases", type=int, nargs="+", default=[0, 135000])
    parser.add_argument("--memory-limit-mib", type=float, default=GUARD_MIB)
    args = parser.parse_args(argv)
    if min(args.steps, args.accum, args.microbatch, args.context) < 1 or args.warmup < 1:
        parser.error("steps, warmup, accum, microbatch and context must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve() / "metrics.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing benchmark: {output}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise SystemExit("Benchmark supports cpu or cuda devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    baseline = load_baseline(args.baseline_source.resolve())
    initial_sources = sources()
    baseline_hash = sha256(args.baseline_source)
    config = get_config("arm2")
    config.update(microbatch=args.microbatch, context=args.context, accum=args.accum, dead_units=False)
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    store = TokenStore(DATA / "tinystories_val.u16", context=args.context)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
    payload = {
        "run_id": args.output_dir.name, "complete": False, "host_gpu": device.type == "cuda",
        "jovin_receipt": False, "kind": "paired_optimization_benchmark",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed validation batches; forward/backward/clipping/AdamW timing. No quality or long-run convergence claim.",
        "timing": "Synchronized host wall time; data transfers excluded; warmup updates precede measured updates.",
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "transformers", "datasets", "pytest")},
        "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": "bfloat16_autocast" if device.type == "cuda" else "float32",
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "git": git_info(), "source_sha256": initial_sources,
        "baseline_source": str(args.baseline_source.resolve()), "baseline_sha256": baseline_hash,
        "checkpoint": None if not args.checkpoint else {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint), "step": checkpoint.get("step")},
        "dataset": {"path": str(DATA / "tinystories_val.u16"), "sha256": sha256(DATA / "tinystories_val.u16"), "meta_sha256": sha256(DATA / "meta.json")},
        "config": config, "steps": args.steps, "warmup": args.warmup, "seeds": args.seeds,
        "fractional_policy": fractional_policy(),
        "phases": args.phases, "memory_limit_mib": args.memory_limit_mib, "cases": [],
    }
    try:
        for seed_index, seed in enumerate(args.seeds):
            torch.manual_seed(seed)
            model = SpectralTransformer(vocab=int(meta["vocab"]), m=config["m"], k=config["k"], context=args.context)
            if checkpoint:
                model.load_state_dict(checkpoint["model"])
            initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.to(device).train()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            host_batches = [store.batch(args.microbatch, generator) for _ in range((args.steps + args.warmup) * args.accum)]
            batch_hash = hashlib.sha256(b"".join(tensor.numpy().tobytes() for batch in host_batches for tensor in batch)).hexdigest()
            batches = [(inputs.to(device), targets.to(device)) for inputs, targets in host_batches]
            for phase in args.phases:
                case = {"seed": seed, "schedule_step": phase, "weights": penalty_weights(config, phase), "batch_sha256": batch_hash}
                print(f"VERIFY seed={seed} schedule_step={phase}", flush=True)
                model.load_state_dict(initial_state)
                old_loss, old_grads = capture_gradients(model, batches[:args.accum], baseline, config, phase, False, device)
                model.load_state_dict(initial_state)
                new_loss, new_grads = capture_gradients(model, batches[:args.accum], candidate, config, phase, True, device)
                case["equivalence"] = compare_gradients(old_grads, new_grads, old_loss, new_loss,
                    loss_rtol=5e-3 if device.type == "cuda" else 2e-5,
                    grad_rtol=2.5e-2 if device.type == "cuda" else 1e-4,
                    atol=1e-5 if device.type == "cuda" else 1e-7)
                del old_grads, new_grads
                payload["cases"].append(case)
                if not case["equivalence"]["passed"]:
                    raise RuntimeError(f"Loss/gradient equivalence failed at seed={seed}, schedule_step={phase}")
                order = [("baseline", baseline, False), ("candidate", candidate, True)]
                if seed_index % 2:
                    order.reverse()
                case["timing_order"] = [name for name, _, _ in order]
                for name, module, optimized in order:
                    model.load_state_dict(initial_state)
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    torch.manual_seed(seed)
                    print(f"BENCH seed={seed} schedule_step={phase} variant={name}", flush=True)
                    case[name] = measure(model, batches, module, config, phase, optimized, device, args)
                case["speedup"] = case["baseline"]["median_seconds_per_step"] / case["candidate"]["median_seconds_per_step"]
                write_metrics(output, payload)
                print(f"RESULT seed={seed} schedule_step={phase} speedup={case['speedup']:.3f}", flush=True)
            del model, initial_state, batches
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if initial_sources != sources() or baseline_hash != sha256(args.baseline_source):
            raise RuntimeError("Source files changed during benchmark; rerun against stable source")
        payload["summary"] = {str(phase): {"median_speedup": statistics.median(case["speedup"] for case in payload["cases"] if case["schedule_step"] == phase)} for phase in args.phases}
        payload["complete"] = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_metrics(output, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
