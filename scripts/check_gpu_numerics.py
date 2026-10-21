"""Small hardware gate for the repaired fractional objective; no quality inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from laplacian.losses import spectral_penalties
from laplacian.operators_torch import fractional_matrix, fractional_policy
from laplacian.runtime import write_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or "5070" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Gate requires the approved RTX 5070 as visible CUDA device zero")
    sources = [Path(__file__).resolve(), ROOT / "src/laplacian/operators_torch.py", ROOT / "src/laplacian/losses.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    payload = {"complete": False, "run_id": args.output.name, "host_gpu": True,
               "scope": "Small fractional numerical hardware checks; no training or quality conclusion",
               "device": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "fractional_policy": fractional_policy(),
               "source_sha256": hashes, "cases": []}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for exponent in (0.25, 0.5, 0.75, 1.0):
            for name, data in (("separated", torch.eye(4) * 100), ("zero", torch.zeros(4, 4)),
                               ("constant", torch.ones(4, 4)), ("duplicate", torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]))):
                for autocast in (False, True):
                    states = data.to(device).requires_grad_()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        loss = spectral_penalties({"pooled": [states]}, [0], exponent,
                            weights={"fractional": 1., "manifold": 0., "hyper": 0., "dirichlet": 0.})["fractional"]
                    gradient = torch.autograd.grad(loss, states)[0]
                    finite = bool(torch.isfinite(loss) & torch.isfinite(gradient).all())
                    payload["cases"].append({"fixture": name, "exponent": exponent,
                        "bf16_autocast": autocast, "loss": float(loss),
                        "gradient_norm": float(gradient.norm()), "finite": finite})
                    if not finite:
                        raise RuntimeError(f"nonfinite {name}/{exponent}/{autocast}")
            # Analytic repeated-eigenvalue derivative, including the zero matrix.
            for eigenvalue in (0., 0.3):
                matrix = (torch.eye(4, dtype=torch.float64, device=device) * eigenvalue).requires_grad_()
                weight = torch.arange(16, dtype=torch.float64, device=device).reshape(4, 4) / 16
                result = fractional_matrix(matrix, exponent)
                gradient = torch.autograd.grad((result * weight).sum(), matrix)[0]
                slope = (2-exponent) * 1e-4 ** (exponent-1) if eigenvalue == 0 else exponent * eigenvalue ** (exponent-1)
                expected = slope * (weight + weight.T) / 2
                torch.testing.assert_close(gradient, expected, atol=1e-9, rtol=1e-9)
                payload["cases"].append({"fixture": "analytic_repeated", "eigenvalue": eigenvalue,
                    "exponent": exponent, "max_abs_error": float((gradient-expected).abs().max()), "finite": True})
        if hashes != {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
            raise RuntimeError("source changed during hardware gate")
        payload["complete"] = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_metrics(args.output / "metrics.json", payload)
    print(json.dumps({"complete": True, "cases": len(payload["cases"])}))


if __name__ == "__main__":
    main()
