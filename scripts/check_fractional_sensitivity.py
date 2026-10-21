"""Exploratory cutoff sensitivity on a historical checkpoint, before protocol freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from laplacian.data import DATA, TokenStore
from laplacian.losses import _affinity, _quadratic
from laplacian.model import SpectralTransformer
from laplacian.operators_torch import fractional_matrix, fractional_policy
from laplacian.runtime import write_metrics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Requires approved RTX 5070 visible as CUDA device zero")
    paths = [Path(__file__).resolve(), *[ROOT / f"src/laplacian/{name}.py" for name in
        ("operators_torch", "losses", "model", "data", "culling", "runtime")]]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    data_path = DATA / "tinystories_val.u16"
    metadata = json.loads((DATA / "meta.json").read_text())
    payload = {"run_id": args.output.name, "complete": False, "host_gpu": True,
        "scope": "Exploratory fractional-component sensitivity only; historical checkpoint/validation batches; no quality or convergence inference.",
        "policy": fractional_policy(), "cutoffs": [1e-5, 1e-4, 1e-3], "exponent": 0.5,
        "seeds": [0, 1, 2], "source_sha256": hashes, "precision": "cuda_bf16_autocast_tf32_enabled",
        "checkpoint_sha256": sha(args.checkpoint), "dataset_sha256": sha(data_path),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "cases": []}
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        initial = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
        model = SpectralTransformer(vocab=metadata["vocab"], m=4, k=0, context=256).to(device).train()
        store = TokenStore(data_path, context=256)
        parameters = list(model.parameters())
        for seed in payload["seeds"]:
            inputs, _ = store.batch(8, torch.Generator().manual_seed(seed))
            batch_hash = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
            inputs = inputs.to(device)
            reference_gradients, reference_value = None, None
            pending = []
            for cutoff in (1e-4, 1e-5, 1e-3):
                model.load_state_dict(initial)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(inputs)
                    value = inputs.new_zeros((), dtype=torch.float32)
                    spectra = []
                    for states in output["pooled"]:
                        states = states.float()
                        lap = _affinity(states, 1.).double()
                        powered = fractional_matrix(lap, 0.5, cutoff=cutoff).float()
                        value = value + .5 * _quadratic(powered, states)
                        eigenvalues = torch.linalg.eigvalsh((lap.detach() + lap.detach().T) / 2)
                        spectra.append({"min": float(eigenvalues.min()), "max": float(eigenvalues.max()),
                                        "negative_count": int((eigenvalues < 0).sum()),
                                        "transition_count": int(((eigenvalues >= 0) & (eigenvalues < cutoff)).sum()),
                                        "count": eigenvalues.numel()})
                gradients = torch.autograd.grad(value, parameters, allow_unused=True)
                finite = bool(torch.isfinite(value)) and all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
                if not finite:
                    raise RuntimeError(f"Nonfinite cutoff sensitivity seed={seed} cutoff={cutoff}")
                measured = float(value.detach())
                norm = sum(float(g.double().square().sum()) for g in gradients if g is not None) ** .5
                if cutoff == 1e-4:
                    reference_gradients = [None if g is None else g.detach().clone() for g in gradients]
                    reference_value = measured
                    reference_norm = norm
                error = sum(float((g.double()-r.double()).square().sum()) for g, r in zip(gradients, reference_gradients) if g is not None) ** .5
                pending.append({"seed": seed, "batch_sha256": batch_hash, "cutoff": cutoff,
                    "fractional_value": measured, "gradient_l2": norm, "finite": finite,
                    "value_delta_vs_default": measured-reference_value,
                    "gradient_relative_l2_vs_default": error / max(reference_norm, 1e-30), "layer_spectra": spectra})
                del output, gradients, value
            payload["cases"].extend(pending)
        if hashes != {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}:
            raise RuntimeError("Source changed during sensitivity diagnostic")
        if sha(args.checkpoint) != payload["checkpoint_sha256"] or sha(data_path) != payload["dataset_sha256"]:
            raise RuntimeError("Input changed during diagnostic")
        payload["complete"] = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_metrics(args.output / "metrics.json", payload)
    print(json.dumps({"complete": True, "cases": len(payload["cases"]),
                      "max_relative_gradient_change": max(c["gradient_relative_l2_vs_default"] for c in payload["cases"])}))


if __name__ == "__main__":
    main()
