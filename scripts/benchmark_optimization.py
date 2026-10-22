"""Bounded, source-pinned comparison of old and current spectral training losses.

This is a host benchmark, not a Jovin isolated receipt or a quality experiment.
It reads local token windows and optional trusted local weights, and never writes
checkpoints or resumes/overwrites an existing experiment's metrics. Schema v2
adds component and penalty-only gates; historical v1 receipts are not upgraded
by changing this script.
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
import os
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

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources():
    paths = sorted((ROOT / "src" / "laplacian").glob("*.py")) + [Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


IMPORTED_SOURCE_HASHES = sources()
PREEXISTING_LAPLACIAN_MODULES = sorted(name for name in sys.modules if name == "laplacian" or name.startswith("laplacian."))

from laplacian import losses as candidate
from laplacian.arms import get_config, layers_0index
from laplacian.data import DATA, TokenStore
from laplacian.device import GUARD_MIB
from laplacian.model import SpectralTransformer
from laplacian.operators_torch import fractional_policy
from laplacian.runtime import write_metrics
from laplacian.train import penalty_weights

POST_IMPORT_SOURCE_HASHES = sources()


def load_baseline(path, expected_sha256):
    """Execute exactly the bytes captured before import, without a cached pyc."""
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("Baseline source changed before execution")
    spec = importlib.util.spec_from_file_location("optimization_baseline_losses", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import baseline source: {path}")
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if sha256(path) != expected_sha256:
        raise RuntimeError("Baseline source changed during import")
    return module


def check_sources(initial_sources, baseline_path, baseline_hash):
    if sources() != initial_sources or sha256(baseline_path) != baseline_hash:
        raise RuntimeError("Source files changed during benchmark; rerun against stable source")


def admit_device(name):
    """Require study authorization before querying or initializing CUDA."""
    device = torch.device(name)
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError("Benchmark supports cpu or cuda devices")
    if (not os.environ.get("STUDY_BUDGET_RUN_ID", "").strip()
            or Path(os.environ.get("STUDY_BUDGET_LEDGER", "")).resolve() != (ROOT / "runs/publication_study/budget.jsonl").resolve()
            or device.index not in (None, 0)):
        raise RuntimeError("CUDA benchmark requires the canonical study-budget wrapper and visible device0")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("CUDA benchmark requires the approved RTX 5070")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA benchmark requires BF16 support")
    return device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


def forward_loss(model, batch, module, config, phase, optimized, device, *, penalty_only=False):
    inputs, targets = batch
    with autocast(device):
        output = model(inputs)
        logits = output["logits"].float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        weights = penalty_weights(config, phase)
        options = {"weights": weights} if optimized else {}
        penalties = module.spectral_penalties(output, layers_0index(config), config["exponent"], **options)
        active = {name: penalties[name] for name, weight in weights.items() if weight != 0.0}
        if penalty_only:
            loss = sum((weights[name] * value for name, value in active.items()), ce.new_zeros(()))
        else:
            loss = module.total_loss(ce, penalties, weights)
    return loss, ce, active


def capture_gradients(model, batches, module, config, phase, optimized, device, *, penalty_only=False):
    model.zero_grad(set_to_none=True)
    values = {}
    for batch in batches:
        loss, ce, penalties = forward_loss(model, batch, module, config, phase, optimized, device, penalty_only=penalty_only)
        if loss.requires_grad:
            (loss / len(batches)).backward()
        scalars = {"loss": loss, "ce": ce, **{f"penalty_{name}": value for name, value in penalties.items()}}
        for name, value in scalars.items():
            values.setdefault(name, []).append(float(value.detach()))
    gradients = {name: None if param.grad is None else param.grad.detach().cpu().clone()
                 for name, param in model.named_parameters()}
    return {name: statistics.mean(samples) for name, samples in values.items()}, gradients


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
        if name not in reference or name not in actual:
            missing.append(name)
            continue
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
        local_max = float(difference.abs().max()) if difference.numel() else 0.0
        if local_max > max_error:
            max_error, worst_parameter = local_max, name
        max_reference = max(max_reference, float(left.abs().max()) if left.numel() else 0.0)
        squared_error += float(difference.square().sum())
        squared_reference += float(left.double().square().sum())
    relative_l2 = math.sqrt(squared_error) / max(math.sqrt(squared_reference), 1e-12)
    loss_error = abs(loss_actual["loss"] - loss_reference["loss"])
    value_errors = {name: abs(loss_actual[name] - value) for name, value in loss_reference.items() if name in loss_actual}
    loss_ok = set(loss_reference) == set(loss_actual) and all(
        error <= atol + loss_rtol * abs(loss_reference[name]) for name, error in value_errors.items())
    grad_ok = relative_l2 <= grad_rtol and max_error <= atol + grad_rtol * max_reference
    return {
        "passed": bool(finite and matching and loss_ok and grad_ok),
        "finite": finite, "matching_gradients": matching,
        "missing_or_shape_mismatch": missing,
        "baseline": loss_reference, "candidate": loss_actual,
        "loss_absolute_error": loss_error, "gradient_relative_l2": relative_l2,
        "value_absolute_errors": value_errors,
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
            loss, ce, _penalties = forward_loss(model, batch, module, config, phase, optimized, device)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Non-finite loss before benchmark optimizer update")
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


def summarize(cases, phases):
    """Descriptive medians by phase and observed order, without a CI claim."""
    summary = {}
    for phase in phases:
        selected = [case for case in cases if case["schedule_step"] == phase]
        strata = {}
        for case in selected:
            strata.setdefault("_then_".join(case["timing_order"]), []).append(case["speedup"])
        summary[str(phase)] = {
            "median_speedup": statistics.median(case["speedup"] for case in selected),
            "case_count": len(selected),
            "by_timing_order": {order: {"case_count": len(values), "median_speedup": statistics.median(values)}
                                for order, values in strata.items()},
        }
    return summary


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
    payload = {
        "schema_version": 2, "run_id": args.output_dir.name, "complete": False, "host_gpu": False,
        "jovin_receipt": False, "kind": "paired_optimization_benchmark",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed validation batches; forward/backward/clipping/AdamW timing. No quality or long-run convergence claim.",
        "source_capture_scope": "Candidate source captured immediately before and after this script's laplacian imports, then before/after baseline execution and at completion. Requires fresh laplacian imports; disk hashes are not a Jovin receipt or an independent in-memory bytecode attestation.",
        "preexisting_laplacian_modules": PREEXISTING_LAPLACIAN_MODULES,
        "import_source_sha256": IMPORTED_SOURCE_HASHES,
        "post_import_source_sha256": POST_IMPORT_SOURCE_HASHES,
        "cases": [],
    }
    initial_sources = baseline_hash = None
    input_hashes = {}
    try:
        initial_sources = sources()
        payload["source_sha256"] = initial_sources
        if initial_sources != IMPORTED_SOURCE_HASHES or POST_IMPORT_SOURCE_HASHES != IMPORTED_SOURCE_HASHES:
            raise RuntimeError("Candidate sources changed at or since import; start a fresh benchmark process")
        if PREEXISTING_LAPLACIAN_MODULES:
            raise RuntimeError("Source-pinned benchmark requires a fresh process without cached laplacian modules")
        device = admit_device(args.device)
        baseline_hash = sha256(args.baseline_source)
        payload.update(baseline_source=str(args.baseline_source.resolve()), baseline_sha256=baseline_hash)
        baseline = load_baseline(args.baseline_source.resolve(), baseline_hash)
        check_sources(initial_sources, args.baseline_source, baseline_hash)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
        config = get_config("arm2")
        config.update(microbatch=args.microbatch, context=args.context, accum=args.accum, dead_units=False)
        input_paths = [DATA / "meta.json", DATA / "tinystories_val.u16"] + ([args.checkpoint] if args.checkpoint else [])
        input_hashes = {str(path.resolve()): sha256(path) for path in input_paths}
        meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
        store = TokenStore(DATA / "tinystories_val.u16", context=args.context)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
        if any(sha256(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("Benchmark input changed during loading")
        payload.update({
            "host_gpu": device.type == "cuda",
            "timing": "Synchronized host wall time; data transfers excluded; warmup updates precede measured updates.",
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "transformers", "datasets", "pytest")},
            "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(),
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "precision": "bfloat16_autocast" if device.type == "cuda" else "float32",
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "git": git_info(),
            "checkpoint": None if not args.checkpoint else {"path": str(args.checkpoint.resolve()), "sha256": input_hashes[str(args.checkpoint.resolve())], "step": checkpoint.get("step")},
            "dataset": {"path": str(DATA / "tinystories_val.u16"), "sha256": input_hashes[str((DATA / "tinystories_val.u16").resolve())], "meta_sha256": input_hashes[str((DATA / "meta.json").resolve())]},
            "config": config, "steps": args.steps, "warmup": args.warmup, "seeds": args.seeds,
            "fractional_policy": fractional_policy(),
            "phases": args.phases, "memory_limit_mib": args.memory_limit_mib,
            "equivalence_scope": "Total loss/gradients, active unweighted component values and separate weighted penalty-only gradients on these batches and initial state. Random-initialization passes do not establish equivalence on trained checkpoints.",
            "timing_order_policy": "Reverse baseline/candidate order when (seed_index + phase_index) is odd; descriptive order strata, no inferential timing claim.",
        })
        for seed_index, seed in enumerate(args.seeds):
            torch.manual_seed(seed)
            model = SpectralTransformer(vocab=int(meta["vocab"]), m=config["m"], k=config["k"], context=args.context)
            if checkpoint is not None:
                model.load_state_dict(checkpoint["model"])
            initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.to(device).train()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            host_batches = [store.batch(args.microbatch, generator) for _ in range((args.steps + args.warmup) * args.accum)]
            batch_hash = hashlib.sha256(b"".join(tensor.numpy().tobytes() for batch in host_batches for tensor in batch)).hexdigest()
            batches = [(inputs.to(device), targets.to(device)) for inputs, targets in host_batches]
            for phase_index, phase in enumerate(args.phases):
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
                case["equivalence"]["checkpoint_supplied"] = args.checkpoint is not None
                case["equivalence"]["initial_state_scope"] = "supplied_checkpoint" if args.checkpoint else "random_initialization_only"
                del old_grads, new_grads
                payload["cases"].append(case)
                if not case["equivalence"]["passed"]:
                    raise RuntimeError(f"Loss/gradient equivalence failed at seed={seed}, schedule_step={phase}")
                model.load_state_dict(initial_state)
                old_loss, old_grads = capture_gradients(model, batches[:args.accum], baseline, config, phase, False, device, penalty_only=True)
                model.load_state_dict(initial_state)
                new_loss, new_grads = capture_gradients(model, batches[:args.accum], candidate, config, phase, True, device, penalty_only=True)
                case["penalty_equivalence"] = compare_gradients(old_grads, new_grads, old_loss, new_loss,
                    loss_rtol=5e-3 if device.type == "cuda" else 2e-5,
                    grad_rtol=2.5e-2 if device.type == "cuda" else 1e-4,
                    atol=1e-5 if device.type == "cuda" else 1e-7)
                case["penalty_equivalence"]["active_terms"] = [name for name, weight in case["weights"].items() if weight != 0]
                del old_grads, new_grads
                if not case["penalty_equivalence"]["passed"]:
                    raise RuntimeError(f"Penalty-only equivalence failed at seed={seed}, schedule_step={phase}")
                order = [("baseline", baseline, False), ("candidate", candidate, True)]
                if (seed_index + phase_index) % 2:
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
        check_sources(initial_sources, args.baseline_source, baseline_hash)
        if any(sha256(path) != digest for path, digest in input_hashes.items()):
            raise RuntimeError("Benchmark input changed during execution")
        payload["summary"] = summarize(payload["cases"], args.phases)
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
