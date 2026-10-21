"""Compare the host training λ2 with the process-isolated reference.

This process must not open the training GPU. The numeric result is a host
check. It is not a jovin-harness receipt. Jovin cannot import PyTorch: the
sandbox image has no torch, no mounts, and no GPU.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "reference" / "operators" / "fiedler.py"


def _reference_fiedler():
    spec = importlib.util.spec_from_file_location("reference_fiedler_isolated", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(nodes=12):
    """Return the agreement report. Raises if the host drifted from the reference."""
    import torch
    from torch import nn

    from laplacian.losses import _lambda2
    from laplacian.train import resample_dead

    reference = _reference_fiedler()
    report = reference.fiedler_aei(reference.path_adjacency(nodes), steps=80)
    adjacency = torch.zeros(nodes, nodes)
    for index in range(nodes - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    host = float(_lambda2(adjacency, steps=80))
    reference_lambda2 = float(report["lambda2"])
    gap = abs(host - reference_lambda2)
    if report["degenerate"] or gap > 1e-3:
        raise RuntimeError(
            f"host lambda2 {host} disagrees with reference {reference_lambda2} (gap {gap})"
        )
    if host > 1.0:
        raise RuntimeError(f"host lambda2 {host} is in the spectral-radius range, not the Fiedler range")

    class _Mlp:
        def __init__(self):
            self.neuron_mask = torch.ones(4)
            self.up = nn.Linear(2, 4, bias=False)
            self.down = nn.Linear(4, 2, bias=False)
            self.up_mask = torch.ones(4)
            self.down_mask = torch.ones(2, 4)

    class _Model:
        def __init__(self):
            self.blocks = [type("Block", (), {"mlp": _Mlp()})()]

    model = _Model()
    before = model.blocks[0].mlp.up.weight.detach().clone()
    changed = resample_dead(model, [torch.zeros(4)])
    wrote = changed == 4 and not torch.equal(model.blocks[0].mlp.up.weight, before)
    if not wrote:
        raise RuntimeError("resample_dead did not write through to the parameter")
    return {
        "run_id": "host_invariants",
        "complete": True,
        "host_gpu": False,
        "jovin_receipt": False,
        "kind": "host_check",
        "note": "Agrees with reference/operators/fiedler.py. Not a jovin receipt.",
        "path_nodes": nodes,
        "reference_lambda2": reference_lambda2,
        "host_lambda2": host,
        "abs_gap": gap,
        "reference_degenerate": bool(report["degenerate"]),
        "resample_write_through": True,
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if "torch" in sys.modules:
        raise SystemExit("torch was imported before the GPU was hidden")
    sys.path.insert(0, str(ROOT / "src"))
    from laplacian.runtime import write_metrics

    payload = check()
    digest = write_metrics(ROOT / "runs" / "host_invariants" / "metrics.json", payload)
    print(
        f"DONE host_invariants sha256={digest} "
        f"reference_lambda2={payload['reference_lambda2']:.6f} "
        f"host_lambda2={payload['host_lambda2']:.6f} gap={payload['abs_gap']:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"FAILED host_invariants {exc}", flush=True)
        raise SystemExit(1)
