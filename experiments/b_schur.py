"""Experiment B: boundary Schur resistance versus the full grid Laplacian.

A 17-by-17 identity check against the dense solver is required before the
513 and 2561 lattices. Those two never allocate a dense vertex matrix.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from laplacian.device import GUARD_MIB, VramGuard, enforce_guard, memory_mib, pick_5070
from laplacian.grid import _dense_matrix, iterative_resistances
from laplacian.mcp_tools import gpu_telemetry_and_guard, verification_gate
from laplacian.operators_torch import effective_resistance
from laplacian.runtime import apply_env, write_metrics


def _pairs(side):
    return (
        (((0, 0), (0, side - 1)), "top_corners"),
        (((0, 0), (side - 1, side - 1)), "diagonal_corners"),
        (((0, side // 2), (side - 1, side // 2)), "opposite_midpoints"),
    )


def _dense_gap(side, pair, measured):
    lap = _dense_matrix("neumann", side, torch.float64, "cpu")
    start, end = pair
    dense = effective_resistance(lap, start[0] * side + start[1], end[0] * side + end[1])
    return {
        "dense": dense,
        "full_gap": abs(measured["full"] - dense),
        "schur_gap": abs(measured["schur"] - dense),
    }


def _run_side(side, device, tol, maxiter):
    started = time.perf_counter()
    rows = []
    peak = memory_mib()
    for pair, name in _pairs(side):
        enforce_guard()
        measured = iterative_resistances(
            side, pair, torch.float64, device, tol, maxiter, f"side={side} pair={name}"
        )
        row = {"pair": name, "endpoints": [list(pair[0]), list(pair[1])], **measured}
        if side <= 17:
            row.update(_dense_gap(side, pair, measured))
        rows.append(row)
        peak = max(peak, memory_mib())
        print(
            f"PROGRESS side={side} pair={name} full={measured['full']:.6e} "
            f"schur={measured['schur']:.6e} gap={measured['gap']:.3e}",
            flush=True,
        )
    return {
        "side": side,
        "pairs": rows,
        "max_solver_gap": max(row["gap"] for row in rows),
        "max_dense_gap": max((row.get("full_gap", 0.0) for row in rows), default=None)
        if side <= 17
        else None,
        "seconds": time.perf_counter() - started,
        "peak_vram_mib": peak,
        "solver": "pcg-vcycle-jacobi",
    }


class BulkMPN(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False) for _ in range(layers)]
        )

    def forward(self, field):
        hidden = field
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        return hidden


def _bulk(device):
    channels = 32
    depth = 15
    completed = None
    tripped = False
    reason = ""
    peak = memory_mib()
    seconds = []
    for side in (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560):
        estimate = side * side * channels * 4 * (depth + 4) / (1024 * 1024)
        if memory_mib() + estimate > GUARD_MIB:
            tripped = True
            reason = "estimate_before_alloc"
            break
        module = None
        field = None
        try:
            started = time.perf_counter()
            module = BulkMPN(channels, depth).to(device)
            field = torch.randn(1, channels, side, side, device=device)
            output = module(field)
            output.square().mean().backward()
            used = enforce_guard()
            peak = max(peak, used)
            completed = side
            seconds.append({"side": side, "seconds": time.perf_counter() - started, "vram_mib": used})
            print(f"PROGRESS bulk side={side} vram_mib={used:.0f}", flush=True)
        except (VramGuard, torch.cuda.OutOfMemoryError) as exc:
            tripped = True
            reason = type(exc).__name__
            torch.cuda.empty_cache()
            break
        finally:
            del module, field
            torch.cuda.empty_cache()
    return {
        "layers": depth,
        "channels": channels,
        "completed_side": completed,
        "tripped": tripped,
        "trip_reason": reason,
        "peak_vram_mib": peak,
        "steps": seconds,
    }


def main():
    apply_env()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, _name = pick_5070()
    telemetry = gpu_telemetry_and_guard()
    identity = _run_side(17, device, tol=1e-8, maxiter=80)
    worst = max(row["full_gap"] for row in identity["pairs"])
    worst = max(worst, max(row["schur_gap"] for row in identity["pairs"]))
    gate = verification_gate(
        "resistance",
        {"full": identity["pairs"][0]["full"], "boundary": identity["pairs"][0]["schur"], "tol": 1e-4},
    )
    payload = {
        "experiment": "B",
        "host_gpu": True,
        "complete": False,
        "device": telemetry,
        "solver": "pcg-vcycle-jacobi",
        "identity_side": 17,
        "identity": identity,
        "identity_max_dense_gap": worst,
        "identity_gate": gate,
        "literature_target_note": "Source document states 10.9 ms to 220 ms. This file stores measured seconds only.",
    }
    path = ROOT / "runs" / "guidance_b" / "metrics.json"
    if worst > 1e-4 or not gate["pass"]:
        payload["stopped_before_large_grids"] = True
        write_metrics(path, payload)
        raise SystemExit(1)
    payload["grid_513"] = _run_side(513, device, tol=1e-6, maxiter=100)
    torch.cuda.empty_cache()
    payload["grid_2561"] = _run_side(2561, device, tol=1e-5, maxiter=120)
    torch.cuda.empty_cache()
    payload["bulk_mpn"] = _bulk(device)
    payload["complete"] = True
    digest = write_metrics(path, payload)
    print(f"DONE guidance_b sha256={digest} identity_gap={worst}", flush=True)


if __name__ == "__main__":
    main()
