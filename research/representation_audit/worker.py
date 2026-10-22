"""Contained CPU worker; launched only by the audited controller after CE completion."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import representation_audit as audit
import limits


def recommendations(rows):
    saturated = sum(layer["affinity"]["near_zero_fraction"] == 1.0 or layer["affinity"]["near_one_fraction"] == 1.0
                    for row in rows for layer in row["metrics"]["layers"])
    return {"state": "proposal_only", "diagnostic_scope": "seed0_training_windows_only",
            "observed_fully_saturated_layer_batches": saturated,
            "next_experiment": "training_only_bandwidth_sweep" if saturated else "representation_intervention_controls",
            "controls": ["same frozen checkpoints and windows", "existing tau=1 graph", "example/feature permutation controls"],
            "endpoints": ["affinity saturation frequency", "normalized gap and effective rank", "numerical invariant residuals"],
            "parameters": {"tau_sweep_if_saturated": [0.25, 0.5, 1.0, 2.0, 4.0]},
            "proposed_budget": {"cpu_threads": 1, "maximum_cpu_elapsed_seconds": 7200, "gpu_hours": 0},
            "stop_conditions": ["source/input drift", "nonfinite diagnostics", "resource limit"],
            "interpretation": "Screen hypotheses only; no causal or held-out quality conclusion. No experiment is automatically launched.",
            "confirmatory_claim_permitted": False}


def execute(manifest_path, directory):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Worker requires explicit CUDA disablement")
    containment = limits.establish(audit.LIMITS["memory_bytes"])
    # The OS containment handle is deliberately retained by this live process.
    audit.immutable(directory / "containment.json", {k: v for k, v in containment.items() if k != "handle"})
    m = audit.checked_manifest(manifest_path)
    if not audit.readiness(m)["ready"]:
        raise ValueError("CE dependency is incomplete")
    result = measure(m, directory)
    audit.checked_manifest(manifest_path)
    audit.immutable(directory / "worker_result.json", result)


def measure(m, directory, model_factory=None):
    """Model-path integration seam; production admission remains in execute()."""
    import numpy as np
    import torch
    from laplacian.model import SpectralTransformer
    from metrics import analyze

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    config = json.loads(Path(m["config"]["path"]).read_bytes())["config"]
    tokens = np.memmap(m["train_inputs"][0]["path"], dtype="<u2", mode="r")
    rows = []
    with (directory / "measurements.jsonl").open("x", encoding="utf-8", newline="\n") as measurements:
        for step, checkpoint in zip(m["steps"], m["snapshots"]):
            audit.check(checkpoint)
            # Trusted local checkpoints only, admitted by original controller hash.
            payload = torch.load(checkpoint["path"], map_location="cpu", mmap=True, weights_only=False)
            if (type(payload.get("step")) is not int or payload["step"] != step
                    or payload.get("checkpoint_version") != 2 or payload.get("training_config") != config
                    or payload.get("training_config_sha256") != audit.config_hash(config)):
                raise ValueError("Checkpoint update/config identity differs")
            if payload.get("reproducibility", {}).get("resume_limitations"):
                raise ValueError("Checkpoint has resume limitations")
            with torch.device("meta"):
                model = (model_factory or SpectralTransformer)(vocab=config["study"]["vocab_size"], m=config["m"], k=config["k"], context=256)
            model.load_state_dict(payload["model"], strict=True, assign=True)
            model.eval().requires_grad_(False)
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Nonfinite/non-CPU model state")
            with torch.inference_mode():
                for batch_number, starts in enumerate(m["sample_offsets"]):
                    packed = np.stack([np.array(tokens[start:start + 257], dtype=np.int64) for start in starts])
                    outputs = model(torch.from_numpy(packed[:, :-1].copy()))
                    # The audit does not evaluate task CE. Drop vocabulary logits
                    # before diagnostics to avoid retaining the largest tensor.
                    del outputs["logits"]
                    metrics = analyze(outputs, selected_layers=(1, 4), exponent=float(config["exponent"]))
                    row = {"step": step, "batch": batch_number, "starts": starts,
                           "window_sha256": hashlib.sha256(packed.astype("<u2").tobytes()).hexdigest(),
                           "metrics": metrics}
                    measurements.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    measurements.flush()
                    os.fsync(measurements.fileno())
                    rows.append(row)
                    print(json.dumps({"step": step, "batch": batch_number, "state": "measured"}), flush=True)
                    del outputs, packed, metrics
            del model, payload
            gc.collect()
            audit.check(checkpoint)
    audit.immutable(directory / "next_experiment.json", recommendations(rows))
    return {"state": "completed", "complete": True,
                    "checkpoints": 4, "microbatches": 32, "scope": audit.SCOPE, "device": "cpu",
                    "torch_threads": torch.get_num_threads(), "checkpoint_and_training_input_hashes": "passed",
                    "gpu_invocations": 0, "held_out_examples": 0, "confirmatory_claim_permitted": False}


def main():
    manifest, directory = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    try:
        audit.require(directory == manifest.parent / "attempt", "Worker output escapes prepared attempt")
        audit.require((directory / "started.json").is_file(), "Worker lacks controller admission")
        audit.require(sys.stdin.buffer.readline() == b"AUDIT_CONTAINED_V1\n", "Worker lacks parent-owned resource containment")
        execute(manifest, directory)
        return 0
    except Exception as exc:
        resource = isinstance(exc, MemoryError) or any(term in str(exc).lower() for term in
                    ("can't allocate memory", "cannot allocate memory", "defaultcpuallocator", "not enough memory"))
        traceback.print_exc()
        path = directory / "worker_result.json"
        if directory.is_dir() and not path.exists():
            audit.immutable(path, {"state": "resource_exhausted" if resource else "failed", "complete": False,
                            "error": f"{type(exc).__name__}: {exc}", "confirmatory_claim_permitted": False})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
