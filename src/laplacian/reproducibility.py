"""Random-state and environment bookkeeping for the custom-model trainer.

Restoring RNG state is necessary, but not a guarantee of bitwise equivalence
across PyTorch versions, hardware, or nondeterministic kernels.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random

import numpy as np
import torch


CHECKPOINT_VERSION = 2


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # manual_seed seeds CPU and all CUDA devices without opening a CUDA context.
    torch.manual_seed(seed)


def rng_state(device):
    """Capture global streams and the single CUDA device used by this trainer."""
    device = torch.device(device)
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        state["torch_cuda"] = {"device_index": index, "state": torch.cuda.get_rng_state(index)}
    return state


def restore_rng_state(state, device):
    """Restore after model/optimizer construction has consumed random numbers."""
    device = torch.device(device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu().contiguous())
    saved_cuda = state.get("torch_cuda")
    if device.type == "cuda":
        if saved_cuda is None:
            return False
        torch.cuda.set_rng_state(saved_cuda["state"].detach().cpu().contiguous(), device=device)
    elif saved_cuda is not None:
        return False
    return True


def environment(device):
    device = torch.device(device)
    return {
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "num_threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
    }


def config_hash(config):
    # Relocating a run does not change its updates; all other config is checked.
    relevant = {key: value for key, value in config.items() if key != "run_id"}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def fresh_metadata():
    return {"resume_status": "fresh_seeded", "resume_limitations": [],
            "elapsed_seconds_total": 0.0, "elapsed_seconds_complete": True}


def ensure_finite_state(value, label="state"):
    """Reject an invalid artifact before touching its previous valid file."""
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite {label}; checkpoint was not written")
    elif isinstance(value, dict):
        for key, item in value.items():
            ensure_finite_state(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            ensure_finite_state(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {label}; checkpoint was not written")
