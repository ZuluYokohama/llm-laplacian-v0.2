"""The four guidance tools. Experiments call these functions directly.

mcp_server.py exposes the same functions over stdio. Runs do not wait on a client.
"""

from __future__ import annotations

import torch

from laplacian.cubical import boundary_matrices, nilpotency_residual
from laplacian.device import enforce_guard, memory_mib, pick_5070
from laplacian.operators_torch import boundary_schur, effective_resistance


def gpu_telemetry_and_guard(name_substring="5070"):
    device, name = pick_5070()
    used = enforce_guard(name_substring)
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "device_name": name,
        "device_index": index,
        "memory_used_mib": used,
        "memory_total_mib": props.total_memory / (1024 * 1024),
        "capability": [props.major, props.minor],
        "l2_cache_bytes": getattr(props, "L2_cache_size", None),
    }


def cochain_builder(cells, device, dtype=torch.float64):
    matrices, counts = boundary_matrices(cells, dtype=dtype, device=device)
    residual = nilpotency_residual(cells, device=device, dtype=dtype)
    return {"matrices": matrices, "counts": counts, "nilpotency": residual}


def dtn_schur_solver(laplacian, boundary):
    """Dense Schur complement for toy graphs. Large grids use the sparse solver."""
    schur = boundary_schur(laplacian, boundary)
    return schur


def verification_gate(kind, payload):
    """Deterministic pass/fail for the three axioms and a GDV comparison."""
    if kind == "nilpotency":
        residual = float(payload["residual_l1"])
        return {"pass": residual < 1e-12, "residual_l1": residual, "kind": kind}
    if kind == "resistance":
        gap = abs(float(payload["full"]) - float(payload["boundary"]))
        return {"pass": gap < float(payload.get("tol", 1e-6)), "gap": gap, "kind": kind}
    if kind == "symplectic":
        residual = float(payload["residual"])
        return {"pass": residual < 1e-12, "residual": residual, "kind": kind}
    if kind == "gdv":
        return {
            "pass": float(payload["morse"]) < float(payload["mlp"]),
            "morse": float(payload["morse"]),
            "mlp": float(payload["mlp"]),
            "kind": kind,
        }
    raise ValueError(f"unknown gate {kind}")


def resistance_pair(laplacian, i, j, boundary):
    full = effective_resistance(laplacian, i, j)
    schur = dtn_schur_solver(laplacian, boundary)
    # Endpoints are the first two columns only when `boundary` is exactly [i, j].
    local_i = boundary.index(i)
    local_j = boundary.index(j)
    reduced = effective_resistance(schur, local_i, local_j)
    return full, reduced, memory_mib()
