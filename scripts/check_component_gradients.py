"""Exploratory component-by-component production-model gradient diagnostic.

No optimizer updates or CE backward. CUDA execution requires the canonical
budget wrapper. Inputs remain trusted local checkpoints; hashes do not make
arbitrary pickle content safe. This next-slice diagnostic is not confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
COMPONENTS = ("manifold", "fractional", "hyper", "dirichlet")
POLICY = "component-unweighted-bf16-v1-draft"
TOLERANCES = {"value_atol": 1e-5, "value_rtol": .005,
              "gradient_relative_l2": .025, "tensor_atol": 1e-5, "tensor_rtol": .025}


def source_hashes():
    paths = sorted((ROOT / "src/laplacian").glob("*.py")) + [
        Path(__file__).resolve(), ROOT / "scripts/benchmark_optimization.py", ROOT / "pyproject.toml",
        ROOT / "paper/planning/FRACTIONAL_POLICY.md",
        ROOT / "paper/planning/COMPONENT_GRADIENT_GATE_DRAFT.md",
        ROOT / "tests/test_component_gradients.py",
        ROOT / "runs/optimization_20261021/baseline_losses.py",
    ]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


IMPORTED_SOURCES = source_hashes()
PRELOADED_LAPLACIAN = sorted(name for name in sys.modules if name == "laplacian" or name.startswith("laplacian."))
_benchmark_path = ROOT / "scripts/benchmark_optimization.py"
_benchmark_raw = _benchmark_path.read_bytes()
if hashlib.sha256(_benchmark_raw).hexdigest() != IMPORTED_SOURCES["scripts/benchmark_optimization.py"]:
    raise RuntimeError("Benchmark helper changed before import")
_spec = importlib.util.spec_from_file_location("component_gradient_benchmark_helper", _benchmark_path)
benchmark = importlib.util.module_from_spec(_spec)
exec(compile(_benchmark_raw, str(_benchmark_path), "exec"), benchmark.__dict__)

from laplacian import losses as candidate
from laplacian import study_train as study
from laplacian import train as training
from laplacian.arms import layers_0index
from laplacian.operators_torch import fractional_matrix, fractional_policy
from laplacian.reproducibility import config_hash, ensure_finite_state, environment, rng_state, restore_rng_state
from laplacian.runtime import write_metrics
from laplacian.schedule import gammas

POST_IMPORT_SOURCES = source_hashes()


def require_gpu():
    return benchmark.admit_device("cuda:0")


def validate_production(config):
    expected = {"m": 4, "k": 0, "microbatch": 8, "accum": 8, "context": 256,
                "layers_1index": [2, 5], "exponent": .5}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Component gate requires production M4/k0/8x8/context256/layers2,5/exponent0.5")
    if config["study"].get("vocab_size") != 50257 or config["study"].get("stage") != "ce_pretrain":
        raise ValueError("Component gate requires a completed production CE parent")


def bind_checkpoint(checkpoint, records):
    """Admit a completed run and exact declared checkpoint bytes before unpickling."""
    folder = checkpoint.parent
    config_path, metrics_path = folder / "study_config.json", folder / "metrics.json"
    for path in (config_path, config_path.with_name("study_config.json.sha256"), metrics_path,
                 metrics_path.with_name("metrics.json.sha256"), checkpoint):
        records.append(study.file_record(path))
    config_document = study.read_verified_metrics(config_path)
    config = study.read_config(config_path)
    if config_document.get("config") != config:
        raise ValueError("Configuration changed between reads")
    metrics = study.read_verified_metrics(metrics_path)
    validate_production(config)
    if (metrics.get("complete") is not True or metrics.get("config") != config
            or metrics.get("steps") != config["steps"] or metrics.get("run_id") != config["run_id"]
            or folder != study.folder_for(config)):
        raise ValueError("Checkpoint requires matching completed CE metrics/configuration")
    receipt = metrics.get("study", {})
    if (receipt.get("config_sha256") != config_hash(config)
            or receipt.get("fractional_policy") != fractional_policy()):
        raise ValueError("Completion receipt policy/configuration differs")
    record = records[-1]
    if checkpoint.name == "last.pt" and receipt.get("final_checkpoint") == record:
        expected_step = config["steps"]
    else:
        expected_step = next((step for step in config["study"]["snapshot_steps"]
                              if checkpoint.name == f"state_step{step:07d}.pt"), None)
        if expected_step is None or record not in receipt.get("timing_states", []):
            raise ValueError("Checkpoint is not a declared completed-run timing state")
    manifest_path = Path(config["study"]["token_manifest"]["path"])
    records.append(study.file_record(manifest_path))
    records.extend(study.file_record(manifest_path.parent / name) for name in sorted(config["study"]["data_files"]))
    for bound in records:
        study.check_record(bound)
    return config, expected_step


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_batches(config, generator, source_step, output, device, deadline):
    store = study.StudyTokenStore(config)
    store.start_segment(output, source_step)
    batches, rows = [], []
    before = tensor_hash(generator.get_state())
    for micro in range(config["accum"]):
        deadline()
        store.set_position(source_step, micro)
        inputs, targets = store.batch(config["microbatch"], generator)
        rows.append({"micro": micro, "shape": list(inputs.shape), "inputs_sha256": tensor_hash(inputs),
                     "targets_sha256": tensor_hash(targets)})
        batches.append(inputs.to(device))
    store.flush()
    return batches, {"split": "train", "batches": rows, "sample_log": study.file_record(store.sample_log),
                     "sampler_before": before, "sampler_after": tensor_hash(generator.get_state())}


def component_gradient(model, batches, module, config, component, *, optimized, device, deadline=lambda: None):
    """Unweighted microbatch mean; no CE or other component enters backward."""
    if component not in COMPONENTS or not batches:
        raise ValueError("A known component and nonempty batch list are required")
    model.zero_grad(set_to_none=True)
    values = []
    selection = {name: float(name == component) for name in COMPONENTS}
    for inputs in batches:
        deadline()
        with benchmark.autocast(device):
            output = model(inputs)
            options = {"weights": selection} if optimized else {}
            penalties = module.spectral_penalties(output, layers_0index(config), config["exponent"], **options)
            loss = penalties[component]
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite {component} value before backward")
        if loss.requires_grad:
            (loss / len(batches)).backward()
        values.append(value)
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                 for name, parameter in model.named_parameters()}
    ensure_finite_state(gradients, f"{component} gradient")
    return statistics.mean(values), gradients


def compare_component(reference_value, candidate_value, reference, actual):
    """Relative cap has NO epsilon floor; zero references require exact zeros."""
    finite = math.isfinite(reference_value) and math.isfinite(candidate_value)
    matching = set(reference) == set(actual)
    rows, mismatches = [], []
    square_error = square_reference = square_candidate = 0.
    for name in sorted(set(reference) | set(actual)):
        if name not in reference or name not in actual:
            matching = False
            mismatches.append(name)
            continue
        left, right = reference[name], actual[name]
        if left is None or right is None:
            if left is not None or right is not None:
                matching = False
                mismatches.append(name)
            rows.append({"name": name, "reference_none": left is None, "candidate_none": right is None})
            continue
        if left.shape != right.shape:
            matching = False
            mismatches.append(name)
            continue
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            finite = False
            rows.append({"name": name, "finite": False, "passed": False})
            continue
        left, right = left.double(), right.double()
        difference = right-left
        local_error = float(difference.abs().max()) if difference.numel() else 0.
        local_reference = float(left.abs().max()) if left.numel() else 0.
        local_reference_norm = float(left.norm())
        local_candidate_norm = float(right.norm())
        local_limit = TOLERANCES["tensor_atol"] + TOLERANCES["tensor_rtol"]*local_reference
        passed = local_error <= local_limit and (local_reference_norm != 0 or local_candidate_norm == 0)
        rows.append({"name": name, "shape": list(left.shape), "max_abs_error": local_error,
                     "reference_max_abs": local_reference, "reference_l2": local_reference_norm,
                     "candidate_l2": local_candidate_norm, "absolute_plus_relative_limit": local_limit,
                     "reference_exact_zero": local_reference_norm == 0, "passed": passed})
        square_error += float(difference.square().sum())
        square_reference += float(left.square().sum())
        square_candidate += float(right.square().sum())
    ref_norm, candidate_norm, error_norm = map(math.sqrt, (square_reference, square_candidate, square_error))
    relative = error_norm / ref_norm if ref_norm > 0 else None
    norm_passed = relative <= TOLERANCES["gradient_relative_l2"] if relative is not None else candidate_norm == 0
    value_error = abs(candidate_value-reference_value) if finite else None
    value_passed = finite and value_error <= TOLERANCES["value_atol"] + TOLERANCES["value_rtol"]*abs(reference_value)
    tensor_passed = all(row.get("passed", True) for row in rows)
    return {"passed": bool(finite and matching and value_passed and norm_passed and tensor_passed),
            "finite": finite, "matching_connectivity_and_shapes": matching, "mismatches": mismatches,
            "reference_value": reference_value if math.isfinite(reference_value) else None,
            "candidate_value": candidate_value if math.isfinite(candidate_value) else None,
            "value_abs_error": value_error, "value_passed": value_passed,
            "reference_gradient_l2": ref_norm, "candidate_gradient_l2": candidate_norm,
            "gradient_absolute_l2_error": error_norm, "gradient_relative_l2_error": relative,
            "gradient_norm_passed": norm_passed, "tensor_guards_passed": tensor_passed,
            "nonzero_gradient_observed": ref_norm > 0, "tiny_reference_norm": 0 < ref_norm < 1e-5,
            "parameters": rows}


def archive_sources(output):
    records = []
    for relative, digest in IMPORTED_SOURCES.items():
        raw = (ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError("Source changed before archive")
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(raw)
        records.append(study.file_record(destination))
    return records


def run_gate(checkpoint, output, *, phase=135000, max_seconds=120.):
    if phase not in (0, 135000) or type(phase) is not int:
        raise ValueError("Phase must be 0 or 135000")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
        raise ValueError("Cooperative limit must be in (0, 120] seconds")
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, snapshots, config, failure = [], [], None, None
    report = {"schema_version": 1, "complete": False, "status": "running", "host_gpu": False,
              "jovin_receipt": False, "policy": POLICY, "tolerances": dict(TOLERANCES),
              "scope": "Exploratory per-component unweighted gradient fidelity; no optimizer updates, quality, speedup or confirmatory promotion",
              "phase": phase, "schedule_weights_for_selection_only": {name: gammas(phase)[name] for name in COMPONENTS},
              "fractional_policy": fractional_policy(), "source_sha256_before": IMPORTED_SOURCES,
              "source_capture_scope": "Disk bytes before/after fresh package imports and at completion; not in-memory bytecode attestation",
              "checkpoint_requested": str(checkpoint), "inputs_before": records, "components": [],
              "max_seconds_cooperative": max_seconds,
              "budget": {key: os.environ.get(key) for key in ("STUDY_BUDGET_RUN_ID", "STUDY_BUDGET_LEDGER")}}
    write_metrics(output / "metrics.json", report)

    def deadline():
        if time.perf_counter()-started > max_seconds:
            raise TimeoutError("Component gate exceeded cooperative limit; wrapper supplies hard deadline")

    try:
        if PRELOADED_LAPLACIAN or POST_IMPORT_SOURCES != IMPORTED_SOURCES or source_hashes() != IMPORTED_SOURCES:
            raise RuntimeError("Component gate requires fresh unchanged package imports")
        device = require_gpu()  # Canonical budget admission before any CUDA query.
        report["host_gpu"] = device.type == "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        snapshots = archive_sources(output)
        report["source_snapshot"] = snapshots
        config, expected_step = bind_checkpoint(checkpoint, records)
        report["training_config"] = config
        report["training_config_sha256"] = config_hash(config)
        deadline()
        baseline = benchmark.load_baseline(study.DEFAULT_BASELINE, study.BASELINE_SHA256)
        if baseline.fractional_matrix is not fractional_matrix or candidate.fractional_matrix is not fractional_matrix:
            raise RuntimeError("Both implementations must share the repaired fractional operator")
        if source_hashes() != IMPORTED_SOURCES:
            raise RuntimeError("Source changed during baseline load")
        model = training.build_model(config, config["study"]["vocab_size"], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]),
                                      weight_decay=config["weight_decay"])
        generator, metadata = torch.Generator(device="cpu"), {}
        for record in records:
            study.check_record(record)
        step, pulses, stripped, _sums, _tokens = training.load_checkpoint(
            checkpoint, model, optimizer, generator, device, config=config, resume_metadata=metadata)
        if step != expected_step or pulses or stripped or metadata.get("resume_status") != "state_complete" or metadata.get("resume_limitations"):
            raise ValueError("Component gate requires exact bound step and complete same-environment state")
        report.update(source_step=step, restore_metadata=metadata, environment=environment(device))
        report["environment"]["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        saved = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        saved_flags = {name: module.training for name, module in model.named_modules()}
        saved_requires = {name: param.requires_grad for name, param in model.named_parameters()}
        saved_rng = rng_state(device)
        report["restored_model_state_sha256"] = state_hash(saved)
        parameter_hash = state_hash(dict(model.named_parameters()))
        report["restored_parameter_sha256"] = parameter_hash
        report["restored_rng_domains"] = sorted(saved_rng)
        batches, report["data"] = prepare_batches(config, generator, step, output, device, deadline)
        report["model_shape"] = {"layers": len(model.blocks), "width": model.d_model,
                                 "vocabulary": config["study"]["vocab_size"], "context": config["context"],
                                 "microbatch": config["microbatch"], "accum": len(batches), "m": config["m"]}
        for component, weight in report["schedule_weights_for_selection_only"].items():
            if weight == 0:
                continue
            row = {"component": component, "unweighted": True, "passed": False}
            report["components"].append(row)
            results = []
            post_states = []
            for module, optimized in ((baseline, False), (candidate, True)):
                deadline()
                model.load_state_dict(saved)
                for name, layer in model.named_modules():
                    layer.training = saved_flags[name]
                for name, param in model.named_parameters():
                    param.requires_grad_(saved_requires[name])
                if not restore_rng_state(saved_rng, device):
                    raise ValueError("Cannot restore every checkpoint RNG domain")
                if device.type == "cuda":
                    benchmark.memory_guard(device, benchmark.GUARD_MIB)
                results.append(component_gradient(model, batches, module, config, component,
                                                  optimized=optimized, device=device, deadline=deadline))
                benchmark.sync(device)
                if state_hash(dict(model.named_parameters())) != parameter_hash:
                    raise RuntimeError("Component backward changed model parameters")
                # Training RMSNorm legitimately updates running buffers in forward.
                # Both passes start from the same buffers and must end identically.
                post_states.append(state_hash(model.state_dict()))
            if post_states[0] != post_states[1]:
                raise RuntimeError("Baseline/candidate forward buffers diverged")
            row["matched_post_forward_state_sha256"] = post_states[0]
            row.update(compare_component(results[0][0], results[1][0], results[0][1], results[1][1]))
            del results
        report["all_active_components_nonzero"] = all(row["nonzero_gradient_observed"] for row in report["components"])
        report["coverage_note"] = "Zero-reference cases do not establish nonzero-gradient fidelity coverage"
        ensure_finite_state({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        if not report["components"] or not all(row["passed"] for row in report["components"]):
            raise ValueError("At least one individual component failed its fixed numerical gate")
        report["complete"], report["status"] = True, "passed"
    except BaseException as exc:
        failure = exc
        report.update(complete=False, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            report["source_sha256_after"] = source_hashes()
            if report["source_sha256_after"] != IMPORTED_SOURCES:
                raise RuntimeError("Source changed during component gate")
            report["inputs_after"] = [study.file_record(record["path"]) for record in records]
            if report["inputs_after"] != records:
                raise RuntimeError("Bound input changed during component gate")
            if config is not None:
                study.validate_config(config)
            for record in snapshots:
                study.check_record(record)
            if "data" in report:
                study.check_record(report["data"]["sample_log"])
            deadline()
            report["freshness_checks_passed"] = True
        except BaseException as exc:
            report.update(complete=False, status="failed", postcheck_error=f"{type(exc).__name__}: {exc}")
            failure = failure or exc
        report["elapsed_seconds_before_final_receipt"] = time.perf_counter()-started
        write_metrics(output / "metrics.json", report)
    if failure is not None:
        raise failure
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", type=int, choices=(0, 135000), default=135000)
    parser.add_argument("--max-seconds", type=float, default=120.)
    args = parser.parse_args(argv)
    report = run_gate(args.checkpoint, args.output, phase=args.phase, max_seconds=args.max_seconds)
    print(json.dumps({"complete": report["complete"], "components": len(report["components"]),
                      "all_active_components_nonzero": report["all_active_components_nonzero"]}))


if __name__ == "__main__":
    main()
