"""One-shot stdio adapter for the four guidance tools.

Input is one JSON object: {"id", "tool", "arguments"}.
Output is one JSON object: {"id", "ok", "result"} or {"id", "ok", "error"}.
Training runs call the functions in mcp_tools.py directly and do not wait on this process.
"""

from __future__ import annotations

import json
import sys

import torch

from laplacian.mcp_tools import (
    cochain_builder,
    dtn_schur_solver,
    gpu_telemetry_and_guard,
    resistance_pair,
    verification_gate,
)

TOOLS = {
    "gpu_telemetry_and_guard": gpu_telemetry_and_guard,
    "cochain_builder": cochain_builder,
    "dtn_schur_solver": dtn_schur_solver,
    "verification_gate": verification_gate,
    "resistance_pair": resistance_pair,
}

_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "int64": torch.int64,
}


def _tensor_view(value):
    detached = value.detach()
    if detached.numel() <= 64:
        return detached.cpu().tolist()
    return {"shape": list(detached.shape), "dtype": str(detached.dtype).removeprefix("torch.")}


def _box(value):
    if isinstance(value, torch.Tensor):
        return _tensor_view(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"truncated": True, "type": type(value).__name__}


def _level3(value):
    if isinstance(value, torch.Tensor):
        return _tensor_view(value)
    if isinstance(value, dict):
        return {str(key): _box(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_box(item) for item in value]
    return _box(value)


def _level2(value):
    if isinstance(value, torch.Tensor):
        return _tensor_view(value)
    if isinstance(value, dict):
        return {str(key): _level3(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_level3(item) for item in value]
    return _box(value)


def _level1(value):
    if isinstance(value, torch.Tensor):
        return _tensor_view(value)
    if isinstance(value, dict):
        return {str(key): _level2(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_level2(item) for item in value]
    return _box(value)


def _at_depth(value, depth):
    """Four explicit container levels. No function calls its own name."""
    del depth
    if isinstance(value, torch.Tensor):
        return _tensor_view(value)
    if isinstance(value, dict):
        return {str(key): _level1(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_level1(item) for item in value]
    return _box(value)


def _arguments(tool, arguments):
    prepared = dict(arguments or {})
    if isinstance(prepared.get("dtype"), str):
        prepared["dtype"] = _DTYPES[prepared["dtype"]]
    if tool in {"dtn_schur_solver", "resistance_pair"} and "laplacian" in prepared:
        prepared["laplacian"] = torch.tensor(prepared["laplacian"], dtype=torch.float64)
    return prepared


def dispatch(message):
    ident = message.get("id")
    tool = message.get("tool")
    if tool not in TOOLS:
        return {"id": ident, "ok": False, "error": f"unknown tool {tool}"}
    try:
        result = TOOLS[tool](**_arguments(tool, message.get("arguments")))
    except Exception as exc:
        return {"id": ident, "ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    return {"id": ident, "ok": True, "result": _at_depth(result, 6)}


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"id": None, "ok": False, "error": "empty stdin"}))
        return 2
    print(json.dumps(dispatch(json.loads(raw))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
