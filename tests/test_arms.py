"""Arm ids, pulse schedule, and the one-shot MCP gate."""

import json
import os
import subprocess
import sys
from pathlib import Path

from laplacian.arms import PULSE_STEPS, alias_sources, all_configs, get_config, layers_0index
from laplacian.mcp_server import dispatch

ROOT = Path(__file__).resolve().parents[1]


def test_seven_pulses_match_the_phase_two_grid():
    assert PULSE_STEPS == list(range(30000, 240000, 30000))
    assert len(PULSE_STEPS) == 7


def test_layer_sets_are_zero_based_and_inside_five_layers():
    assert layers_0index(get_config("arm2")) == [1, 4]
    wider = get_config("grid_s0p25_l135")
    assert layers_0index(wider) == [0, 2, 4]


def test_alias_ids_are_not_second_copies():
    configs = all_configs()
    aliases = alias_sources()
    assert aliases == {"grid_s0p5_l25": "arm2", "scale_m8": "arm4"}
    assert "grid_s0p5_l25" not in configs
    assert "scale_m8" not in configs
    grids = [name for name in configs if name.startswith("grid_")]
    assert len(grids) == 8
    assert get_config("arm4")["m"] == 8
    assert get_config("scale_m2")["m"] == 2


def test_mcp_nilpotency_gate_accepts_a_zero_residual():
    body = dispatch(
        {
            "id": "gate",
            "tool": "verification_gate",
            "arguments": {"kind": "nilpotency", "payload": {"residual_l1": 0}},
        }
    )
    assert body["ok"] is True
    assert body["result"]["pass"] is True
    assert body["id"] == "gate"


def test_mcp_cochain_builder_returns_shapes_for_a_one_cell_complex():
    body = dispatch({"id": "cells", "tool": "cochain_builder", "arguments": {"cells": 1, "device": "cpu"}})
    assert body["ok"] is True
    assert body["result"]["nilpotency"]["residual_l1"] == 0.0


def test_mcp_stdio_process_returns_one_json_object():
    python = sys.executable
    message = {
        "id": "stdio",
        "tool": "verification_gate",
        "arguments": {"kind": "symplectic", "payload": {"residual": 0}},
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [str(python), "-m", "laplacian.mcp_server"],
        input=json.dumps(message),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["ok"] is True
    assert body["result"]["pass"] is True


def test_queue_lists_each_alias_once_and_fetches_last():
    import importlib.util

    path = ROOT / "scripts" / "run_queue.py"
    spec = importlib.util.spec_from_file_location("run_queue_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [item["id"] for item in module.build_items()]
    assert ids.count("grid_s0p5_l25") == 1
    assert ids.count("scale_m8") == 1
    assert ids.index("arm2") < ids.index("grid_s0p5_l25")
    assert ids.index("arm4") < ids.index("scale_m8")
    assert ids.index("arm3") < ids.index("arm3_masks")
    assert ids.index("smoke_pulse") < ids.index("eta")
    assert ids.index("verify_reference") < ids.index("host_invariants") < ids.index("arm0")
    invariant = next(item for item in module.build_items() if item["id"] == "host_invariants")
    assert invariant["gpu"] is False
    assert ids[-1] == "fetch_references"
    assert "guidance_a" not in ids


def test_bind_paper_stays_closed_and_rejects_unmeasured_targets():
    import importlib.util

    path = ROOT / "scripts" / "bind_paper.py"
    spec = importlib.util.spec_from_file_location("bind_paper_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.queue_is_done() is False
    assert "1.466" in module.bait_hits("measured val loss 1.466\n")
    assert module.bait_hits("literature_target val loss 1.466 was not measured\n") == []


def test_training_modules_import():
    import laplacian.data
    import laplacian.eval_run
    import laplacian.smol
    import laplacian.train

    assert laplacian.train.write_eta is not None
    assert laplacian.smol.REPO == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert callable(laplacian.data.prepare)
    assert callable(laplacian.eval_run.evaluate)
