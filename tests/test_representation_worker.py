"""Actual model-path CPU integration on synthetic checkpoints, not study results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/representation_audit"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import worker
from laplacian.model import SpectralTransformer


def record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(734)
    torch.manual_seed(734)
    config = {"run_id": "synthetic_cpu_fixture", "m": 1, "k": 0, "exponent": .5, "study": {"vocab_size": 64}, "seed": 0}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config": config}))
    tokens = tmp_path / "train.u16"
    tokens.write_bytes(torch.randint(0, 64, (2048,), generator=generator).numpy().astype("<u2").tobytes())
    snapshots = []
    for step in worker.audit.STEPS:
        model = SpectralTransformer(vocab=64, m=1, k=0, d_model=16, n_layers=2, context=256)
        checkpoint = tmp_path / f"state_step{step:07d}.pt"
        torch.save({"model": model.state_dict(), "step": step, "checkpoint_version": 2,
                    "training_config": config, "training_config_sha256": worker.audit.config_hash(config),
                    "reproducibility": {"resume_limitations": []}}, checkpoint)
        snapshots.append(record(checkpoint))
    manifest = {"config": record(path), "steps": worker.audit.STEPS, "snapshots": snapshots,
                "train_inputs": [record(tokens)], "sample_offsets": worker.audit.offsets(2048)}
    return manifest


def factory(**kwargs):
    return SpectralTransformer(**kwargs, d_model=16, n_layers=2)


def test_real_model_path_fixed_windows_and_repeatability(tiny, tmp_path, monkeypatch):
    # Local fixture checkpoints are allowed only through this integration seam.
    # Production admission/limits are tested separately, not bypassed in the CLI.
    monkeypatch.setattr(worker.audit, "ROOT", tmp_path)
    inputs = [tiny["config"], *tiny["snapshots"], *tiny["train_inputs"]]
    outputs = []
    for name in ("first", "second"):
        out = tmp_path / name
        out.mkdir()
        result = worker.measure(tiny, out, model_factory=factory)
        worker.audit.immutable(out / "worker_result.json", result)
        worker.audit.validate_outputs(out, tiny, expected_layers=2)
        assert result["microbatches"] == 32 and result["checkpoints"] == 4
        assert result["gpu_invocations"] == result["held_out_examples"] == 0
        rows = [json.loads(line) for line in (out / "measurements.jsonl").read_text().splitlines()]
        assert len(rows) == 32 and {r["step"] for r in rows} == set(worker.audit.STEPS)
        for batch in range(8):
            selected = [r for r in rows if r["batch"] == batch]
            assert len({r["window_sha256"] for r in selected}) == 1
        proposal = json.loads((out / "next_experiment.json").read_bytes())
        assert proposal["state"] == "proposal_only" and proposal["proposed_budget"]["gpu_hours"] == 0
        outputs.append((out / "measurements.jsonl").read_bytes())
    assert outputs[0] == outputs[1]
    for original in inputs:
        assert record(Path(original["path"])) == original


def test_incorrect_checkpoint_update_is_rejected(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(worker.audit, "ROOT", tmp_path)
    tiny["steps"] = [1, *tiny["steps"][1:]]
    out = tmp_path / "wrong_step"
    out.mkdir()
    with pytest.raises(ValueError, match="Checkpoint update/config"):
        worker.measure(tiny, out, model_factory=factory)
    assert not (out / "worker_result.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows production Job Object containment")
def test_real_job_enforces_memory_cap_and_one_cpu():
    code = """import sys,json
sys.path.insert(0,sys.argv[1])
import limits
held=limits.establish(64*1024*1024)
try:
    x=bytearray(128*1024*1024)
except MemoryError:
    print(json.dumps({'memory_enforced':True,'affinity':held['affinity']}))
else:
    raise AssertionError('Memory allocation should be denied')
"""
    result = subprocess.run([sys.executable, "-c", code, str(HERE)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["memory_enforced"] and observed["affinity"] & (observed["affinity"] - 1) == 0


@pytest.mark.skipif(os.name != "nt", reason="Windows production Job Object containment")
def test_owner_death_terminates_contained_worker(tmp_path):
    # Actual child process, no model and no GPU. The owner is killed abruptly.
    code = """import pathlib,subprocess,sys,time
sys.path.insert(0,sys.argv[1])
import representation_audit as audit
childcode='import sys,time;sys.stdin.buffer.readline();time.sleep(60)'
child=subprocess.Popen([sys.executable,'-c',childcode],stdin=subprocess.PIPE)
held=audit.contain_worker(child)
pathlib.Path(sys.argv[2]).write_text(str(child.pid))
time.sleep(60)
"""
    pidfile = tmp_path / "child.pid"
    owner = subprocess.Popen([sys.executable, "-c", code, str(ROOT / "scripts"), str(pidfile)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    import ctypes
    import time
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = None
    try:
        deadline = time.monotonic() + 15
        while not pidfile.exists() and time.monotonic() < deadline:
            if owner.poll() is not None:
                pytest.fail(owner.stderr.read().decode())
            time.sleep(.05)
        assert pidfile.exists()
        handle = kernel.OpenProcess(0x100000, False, int(pidfile.read_text()))
        assert handle
        owner.kill()
        owner.wait(timeout=5)
        assert kernel.WaitForSingleObject(handle, 5000) == 0
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if handle:
            kernel.CloseHandle(handle)
