"""Manifest-bound CE pretraining and explicit paired continuation forks.

The defaults are a proposed bounded protocol, not the original 300k schedule.
Only ``train`` executes updates. ``fork`` creates a new, trusted local checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from laplacian.arms import get_config
from laplacian.operators_torch import fractional_matrix, fractional_policy
from laplacian.reproducibility import CHECKPOINT_VERSION, config_hash, ensure_finite_state, fresh_metadata
from laplacian.runtime import ROOT, RUNS, read_json, write_metrics
from laplacian.study_tokens import validate_bundle

PROTOCOL = "bounded-ce50k-paired10k-v1-proposed"
BASELINE_SHA256 = "604a993fc6db5cb0b86df375f6415313bab6b88600642acc3d40a65bd151e803"
DEFAULT_BASELINE = ROOT / "runs/optimization_20261021/baseline_losses.py"


def snapshot_steps(steps):
    return sorted({0, max(1, math.ceil(steps * 0.1)), max(1, steps // 2), steps})


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def check_record(record):
    if file_record(record["path"]) != record:
        raise ValueError(f"Study bound file changed: {record['path']}")


def source_records():
    # Include all local trainer/operator dependencies, including data validation.
    return [file_record(path) for path in sorted(Path(__file__).parent.glob("*.py"))]


def folder_for(config):
    run_id = config["run_id"]
    if not isinstance(run_id, str) or not run_id or Path(run_id).is_absolute():
        raise ValueError("Study run_id must be a relative path inside RUNS")
    folder = (RUNS / run_id).resolve()
    if not folder.is_relative_to(RUNS.resolve()) or folder == RUNS.resolve():
        raise ValueError("Study run_id escapes RUNS")
    return folder


def validate_config(config):
    study = config["study"]
    if study.get("schema_version") != 1 or study.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported study protocol")
    if study.get("fractional_policy") != fractional_policy():
        raise ValueError("Study fractional policy changed")
    if study.get("sources") != source_records():
        raise ValueError("Study source files changed; make a new protocol/run instead of resuming")
    if type(study.get("checkpoint_every")) is not int or study["checkpoint_every"] <= 0:
        raise ValueError("Study checkpoint cadence must be a positive integer")
    check_record(study["token_manifest"])
    manifest = validate_bundle(study["token_manifest"]["path"])
    if study.get("data_files") != manifest["files"] or study.get("vocab_size") != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("Study token manifest binding differs")
    if config.get("pulses") or config.get("norm_removal") or config.get("pulse_only") or config.get("teacher") or config.get("spectral_probe_steps"):
        raise ValueError("The bounded study protocol excludes pruning, stripping, distillation and probes")
    if config.get("manifold_constant") is not None:
        raise ValueError("The study protocol uses CE or the explicit fixed full penalty schedule")
    if any(type(config[key]) is not int or config[key] <= 0 for key in ("steps", "microbatch", "accum", "context")):
        raise ValueError("Positive integer steps, microbatch, accum and context are required")
    if study.get("snapshot_steps") != snapshot_steps(config["steps"]):
        raise ValueError("Study must prospectively retain initial/early/middle/final states")
    stage, implementation = study.get("stage"), study.get("loss_implementation")
    if stage == "ce_pretrain":
        if config["spectral"] or implementation != "optimized" or study.get("penalty_step") is not None:
            raise ValueError("CE pretraining cannot enable spectral penalties")
    elif stage == "spectral_continuation":
        if (not config["spectral"] or config["dead_units"] or study.get("penalty_step") != 135000
                or config["lr"] != 3e-5 or config["min_lr"] != 3e-5 or config["warmup"] != 0):
            raise ValueError("Continuations require fixed step135000, constant LR3e-5 and no dead-unit resampling")
        if not study.get("lineage"):
            raise ValueError("Continuation requires explicit parent checkpoint lineage")
        check_record(study["lineage"]["checkpoint"])
        check_record(study["lineage"]["parent_metrics"])
    else:
        raise ValueError("Unknown study stage")
    if implementation == "baseline":
        record = study.get("baseline_loss_source")
        if not record or record["sha256"] != BASELINE_SHA256:
            raise ValueError("Baseline must be the immutable prior loss source")
        check_record(record)
    elif implementation != "optimized" or study.get("baseline_loss_source") is not None:
        raise ValueError("Invalid study loss implementation")
    folder_for(config)
    return manifest


def make_pretrain_config(token_manifest, run_id, seed, *, steps=50000):
    path = Path(token_manifest).resolve()
    manifest = validate_bundle(path)
    config = get_config("arm0")
    config.update(run_id=run_id, seed=int(seed), steps=int(steps), lr=3e-4, min_lr=3e-5, warmup=2000)
    config["study"] = {
        "schema_version": 1, "protocol": PROTOCOL, "protocol_status": "proposed_not_confirmatory_frozen",
        "stage": "ce_pretrain", "loss_implementation": "optimized", "baseline_loss_source": None,
        "penalty_step": None, "checkpoint_every": 1000, "snapshot_steps": snapshot_steps(config["steps"]),
        "token_manifest": file_record(path), "data_files": manifest["files"],
        "vocab_size": manifest["tokenizer"]["vocab_size"], "sources": source_records(),
        "fractional_policy": fractional_policy(), "lineage": None,
    }
    validate_config(config)
    return config


def write_config(folder, config):
    write_metrics(folder / "study_config.json", {
        "schema_version": 1, "config": config, "config_sha256": config_hash(config),
    })


def read_config(path):
    path = Path(path).resolve()
    document = read_json(path)
    config = document["config"]
    if document.get("schema_version") != 1 or document.get("config_sha256") != config_hash(config):
        raise ValueError("Invalid study configuration digest")
    if path != folder_for(config) / "study_config.json":
        raise ValueError("Study configuration must remain in its bound run directory")
    validate_config(config)
    return config


def read_verified_metrics(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")  # Universal newlines match runtime.write_metrics.
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if path.with_name(path.name + ".sha256").read_text(encoding="utf-8").strip() != actual:
        raise ValueError("Study metrics sidecar digest differs")
    return json.loads(text)


def prepare_pretrain(config):
    validate_config(config)
    folder = folder_for(config)
    folder.mkdir(parents=True, exist_ok=False)
    write_config(folder, config)
    return folder / "study_config.json"


def validate_checkpoint(payload, config, *, model=None):
    """Study resume requires a modern, fully bound checkpoint before mutation."""
    required = {"model", "optimizer", "generator", "rng_state", "requires_grad", "module_training", "environment",
                "reproducibility", "training_config", "training_config_sha256", "step", "pulses_done", "stripped", "fired_tokens"}
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION or not required.issubset(payload):
        raise ValueError("Study requires a full current trusted checkpoint, not weights or a legacy checkpoint")
    if (payload["training_config_sha256"] != config_hash(payload["training_config"])
            or payload["training_config_sha256"] != config_hash(config)
            or payload["training_config"] != config):
        raise ValueError("Study checkpoint training configuration differs")
    metadata = payload["reproducibility"]
    if metadata.get("resume_limitations") or metadata.get("elapsed_seconds_complete") is not True:
        raise ValueError("Study requires complete RNG/config/time checkpoint provenance")
    if not isinstance(metadata.get("elapsed_seconds_total"), (int, float)) or metadata["elapsed_seconds_total"] < 0:
        raise ValueError("Study checkpoint requires nonnegative elapsed time")
    if not {"python", "numpy", "torch_cpu"}.issubset(payload["rng_state"]):
        raise ValueError("Study checkpoint lacks global RNG states")
    if payload["environment"].get("device_type") == "cuda" and "torch_cuda" not in payload["rng_state"]:
        raise ValueError("Study checkpoint lacks CUDA RNG state")
    if type(payload["fired_tokens"]) is not int or payload["fired_tokens"] < 0:
        raise ValueError("Invalid study firing-token counter")
    if payload["fired_tokens"] > 0 and not payload.get("fired_sums"):
        raise ValueError("Study checkpoint lacks firing-window sums")
    if (not payload["requires_grad"] or not payload["module_training"]
            or not all(type(flag) is bool for flag in [*payload["requires_grad"].values(), *payload["module_training"].values()])):
        raise ValueError("Study checkpoint lacks complete training flags")
    if model is not None and (set(payload["requires_grad"]) != {name for name, _ in model.named_parameters()}
                              or set(payload["module_training"]) != {name for name, _ in model.named_modules()}):
        raise ValueError("Study checkpoint training flag keysets differ from model")
    if type(payload["step"]) is not int or not 0 <= payload["step"] <= config["steps"]:
        raise ValueError("Invalid study checkpoint step")
    ensure_finite_state(payload)


def fork_checkpoint(parent_checkpoint, run_id, *, loss_implementation, steps=10000, baseline_loss_source=None):
    """Deserialize only a trusted local full checkpoint. This function runs no GPU work."""
    parent_checkpoint = Path(parent_checkpoint).resolve()
    parent_record = file_record(parent_checkpoint)
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    parent = payload.get("training_config", {})
    validate_config(parent)
    validate_checkpoint(payload, parent)
    if parent["study"]["stage"] != "ce_pretrain" or payload["step"] != parent["steps"]:
        raise ValueError("Fork requires the final checkpoint of a completed CE pretrain")
    metrics_path = parent_checkpoint.parent / "metrics.json"
    metrics = read_verified_metrics(metrics_path)
    if (metrics.get("complete") is not True or metrics.get("steps") != payload["step"]
            or config_hash(metrics.get("config", {})) != config_hash(parent)):
        raise ValueError("Parent completion metrics do not match the checkpoint/configuration")
    if metrics.get("study", {}).get("final_checkpoint") != parent_record:
        raise ValueError("Parent final checkpoint bytes do not match completion metrics")
    if parent_checkpoint != folder_for(parent) / "last.pt":
        raise ValueError("Parent must be its study run's final last.pt checkpoint")
    config = json.loads(json.dumps(parent))
    config.update(run_id=run_id, steps=int(steps), spectral=True, dead_units=False, lr=3e-5, min_lr=3e-5, warmup=0)
    config["study"].update(
        stage="spectral_continuation", loss_implementation=loss_implementation, penalty_step=135000,
        snapshot_steps=snapshot_steps(config["steps"]),
        baseline_loss_source=file_record(baseline_loss_source or DEFAULT_BASELINE) if loss_implementation == "baseline" else None,
        lineage={"checkpoint": parent_record, "parent_step": payload["step"], "parent_run_id": parent["run_id"],
                 "parent_config_sha256": config_hash(parent), "parent_config": parent,
                 "parent_elapsed_seconds_total": payload["reproducibility"]["elapsed_seconds_total"],
                 "parent_metrics": file_record(metrics_path), "fork_created_utc": datetime.now(timezone.utc).isoformat(),
                 "changed_fields": ["run_id", "steps", "spectral", "dead_units", "lr", "min_lr", "warmup", "study"],
                 "state_transfer": "model_optimizer_buffers_all_rng_flags_preserved; branch_step_reset_to_zero"},
    )
    validate_config(config)
    check_record(parent_record)
    folder = folder_for(config)
    folder.mkdir(parents=True, exist_ok=False)
    # Preserve tensor states, Python/NumPy/CPU/CUDA RNGs and training flags verbatim.
    payload.update(step=0, training_config=config, training_config_sha256=config_hash(config),
                   reproducibility=fresh_metadata(), last_loss=None, last_ce=None)
    ensure_finite_state(payload)
    temporary = folder / "last.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(folder / "last.pt")
    write_config(folder, config)
    return folder / "study_config.json"


@functools.lru_cache(maxsize=4)
def baseline_module(path, digest):
    record = file_record(path)
    if digest != BASELINE_SHA256 or record["sha256"] != digest:
        raise ValueError("Baseline source changed before module load")
    spec = importlib.util.spec_from_file_location("_study_baseline_losses_" + digest, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.fractional_matrix is not fractional_matrix:
        raise ValueError("Baseline must share the repaired fractional operator")
    return module


def baseline_penalties(config, outputs, layers, exponent):
    record = config["study"]["baseline_loss_source"]
    return baseline_module(record["path"], record["sha256"]).spectral_penalties(outputs, layers, exponent)


class StudyTokenStore:
    """Training split only. Offset logs bind each drawn context to story indices.

    Windows may cross EOS/story boundaries, matching the legacy stream sampler.
    A record describes a draw, not proof that its optimizer update committed.
    """

    def __init__(self, config):
        self.config = config
        self.context = int(config["context"])
        folder = Path(config["study"]["token_manifest"]["path"]).parent
        self.tokens = np.memmap(folder / "train.u16", dtype="<u2", mode="r")
        if len(self.tokens) <= self.context + 2:
            raise ValueError("Study train token stream is too short")
        self.sample_log = None
        self.position = None

    def start_segment(self, folder, start_step):
        self.sample_log = folder / ("samples-" + uuid.uuid4().hex + ".jsonl")
        record = {"kind": "segment", "schema_version": 1, "start_step": start_step,
                  "config_sha256": config_hash(self.config), "split": "train", "offset_unit": "tokens",
                  "train_tokens": self.config["study"]["data_files"]["train.u16"],
                  "story_index": self.config["study"]["data_files"]["train.stories.jsonl"],
                  "window_tokens": self.context + 1, "records_describe": "draws_not_committed_updates"}
        with self.sample_log.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def set_position(self, step, micro):
        self.position = (int(step), int(micro))

    def batch(self, count, generator):
        if self.sample_log is None or self.position is None:
            raise RuntimeError("Study sampling requires an explicit segment and update/microbatch position")
        starts = torch.randint(0, len(self.tokens) - self.context - 1, (count,), generator=generator).tolist()
        packed = np.stack([np.array(self.tokens[start:start + self.context + 1], dtype=np.int64) for start in starts])
        with self.sample_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"kind": "draw", "step": self.position[0], "micro": self.position[1], "starts": starts}, separators=(",", ":")) + "\n")
        batch = torch.from_numpy(packed)
        return batch[:, :-1], batch[:, 1:]

    def flush(self):
        if self.sample_log is not None:
            with self.sample_log.open("ab") as handle:
                handle.flush()
                os.fsync(handle.fileno())


def completion_record(config, folder, store):
    store.flush()
    validate_config(config)
    return {"protocol": PROTOCOL, "fractional_policy": fractional_policy(),
            "config_sha256": config_hash(config), "sample_logs": [file_record(path) for path in sorted(folder.glob("samples-*.jsonl"))],
            "timing_states": [file_record(folder / f"state_step{step:07d}.pt") for step in config["study"]["snapshot_steps"]],
            "lineage": config["study"]["lineage"], "freshness_checked_at_completion": True,
            "limitation": "Training completion is not a quality or generalization result; evaluation remains separate."}


@contextmanager
def run_lock(folder):
    """OS lock is released after process death; an existing file is not a stale lock."""
    with (folder / ".study.lock").open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute(config_path):
    from laplacian.train import train
    config = read_config(config_path)
    with run_lock(folder_for(config)):
        train(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train", help="Create a fresh CE run or strictly resume an existing study config")
    run.add_argument("--config", type=Path)
    run.add_argument("--token-manifest", type=Path)
    run.add_argument("--run-id")
    run.add_argument("--seed", type=int)
    run.add_argument("--steps", type=int)
    fork = commands.add_parser("fork", help="Prepare (do not execute) a continuation from a trusted final CE checkpoint")
    fork.add_argument("--parent-checkpoint", type=Path, required=True)
    fork.add_argument("--run-id", required=True)
    fork.add_argument("--loss-implementation", choices=("baseline", "optimized"), required=True)
    fork.add_argument("--baseline-loss-source", type=Path)
    fork.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.command == "fork":
        if args.loss_implementation != "baseline" and args.baseline_loss_source is not None:
            parser.error("--baseline-loss-source applies only to baseline continuations")
        path = fork_checkpoint(args.parent_checkpoint, args.run_id, loss_implementation=args.loss_implementation,
                               steps=args.steps, baseline_loss_source=args.baseline_loss_source)
        print(json.dumps({"prepared": str(path), "training_executed": False}))
    else:
        if args.config:
            if any(value is not None for value in (args.token_manifest, args.run_id, args.seed, args.steps)):
                parser.error("--config resumes unchanged; no training overrides are accepted")
            path = args.config
        else:
            if args.token_manifest is None or args.run_id is None or args.seed is None:
                parser.error("fresh training requires --token-manifest, --run-id and --seed")
            path = prepare_pretrain(make_pretrain_config(args.token_manifest, args.run_id, args.seed,
                                                        steps=args.steps if args.steps is not None else 50000))
        execute(path)


if __name__ == "__main__":
    main()
