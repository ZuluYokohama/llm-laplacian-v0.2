"""Experiment D: perimeter ReLU surrogate, tropical max-plus layer, FP8 storage.

Bytes, train seconds, and inference latency are measured. The 7 kB / 0.67 s /
0.003 ms figures from the source note are stored only as literature_target text.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from laplacian.device import enforce_guard, pick_5070
from laplacian.mcp_tools import gpu_telemetry_and_guard
from laplacian.runtime import apply_env, write_metrics


def ellipse(n, a=2.0, b=1.0):
    theta = torch.linspace(0, 2 * torch.pi, n + 1)[:-1]
    x = a * torch.cos(theta)
    y = b * torch.sin(theta)
    denom = (a * a * torch.sin(theta) ** 2 + b * b * torch.cos(theta) ** 2) ** 1.5
    kappa = (a * b) / denom
    features = torch.stack([x, y, kappa], dim=1)
    response = kappa * torch.exp(-0.25 * x)
    response = (response - response.mean()) / response.std().clamp_min(1e-6)
    return features.float(), response.float().unsqueeze(1)


class ReLUSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, features):
        hidden = torch.relu(self.fc1(features))
        hidden = torch.relu(self.fc2(hidden))
        return self.fc3(hidden), self.fc1(features)


class TropicalSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.tropical = nn.Parameter(torch.randn(32, 3) * 0.01)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, features):
        scores = self.tropical + features[:, None, :]
        hidden = scores.max(dim=-1).values
        top2 = torch.topk(scores, k=2, dim=-1).values
        gap = (top2[..., 0] - top2[..., 1]).abs()
        output = self.fc3(torch.relu(self.fc2(hidden)))
        return output, gap


def _bytes(module, dtype=None):
    total = 0
    count = 0
    for value in module.state_dict().values():
        if not value.is_floating_point() and dtype is None:
            continue
        tensor = value if dtype is None else value.detach().to(dtype)
        total += tensor.numel() * tensor.element_size()
        count += tensor.numel()
    return int(total), int(count)


def _train(module, features, target, steps, device):
    module.to(device)
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    started = time.perf_counter()
    last = None
    module.train()
    for _step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred, _aux = module(features)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
        enforce_guard()
    return last, time.perf_counter() - started


def _latency_ms(module, features):
    module.eval()
    for _warm in range(20):
        module(features)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _repeat in range(200):
        module(features)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / 200.0)


def _fit_stats(pred, target):
    residual = float(torch.mean((pred - target) ** 2).item())
    variance = float(torch.var(target).item())
    r2 = 1.0 - residual / max(variance, 1e-12)
    sign_hit = float((torch.sign(pred) == torch.sign(target)).float().mean().item())
    return {"mse": residual, "r2": r2, "sign_agreement": sign_hit}


def _fp8(module):
    try:
        stored = {key: value.detach().to(torch.float8_e4m3fn) for key, value in module.state_dict().items()}
        nbytes = sum(value.numel() * value.element_size() for value in stored.values())
        restored = {key: value.to(torch.float32) for key, value in stored.items()}
        return {"accepted": True, "bytes": int(nbytes), "dtype": "float8_e4m3fn", "roundtrip_keys": len(restored)}
    except Exception as exc:
        return {"accepted": False, "error": str(exc)[:300], "dtype": "float8_e4m3fn"}


def main():
    apply_env()
    device, _name = pick_5070()
    telemetry = gpu_telemetry_and_guard()
    features, target = ellipse(256)
    features = features.to(device)
    target = target.to(device)
    relu = ReLUSurrogate()
    tropical = TropicalSurrogate()
    relu_bytes, relu_params = _bytes(relu)
    tropical_bytes, tropical_params = _bytes(tropical)
    relu_loss, relu_seconds = _train(relu, features, target, 400, device)
    tropical_loss, tropical_seconds = _train(tropical, features, target, 400, device)
    relu.eval()
    tropical.eval()
    with torch.no_grad():
        relu_pred, preact = relu(features)
        tropical_pred, gaps = tropical(features)
        relu_stats = _fit_stats(relu_pred, target)
        tropical_stats = _fit_stats(tropical_pred, target)
        near_threshold = float((preact.abs() < 1e-3).float().mean().item())
        tropical_tie = float((gaps < 1e-3).float().mean().item())
    payload = {
        "experiment": "D",
        "host_gpu": True,
        "complete": True,
        "device": telemetry,
        "n_perimeter": 256,
        "response": "kappa * exp(-0.25 x), then mean-std normalized",
        "steps": 400,
        "relu": {
            "parameter_bytes": relu_bytes,
            "parameters": relu_params,
            "train_seconds": relu_seconds,
            "final_loss": relu_loss,
            "infer_latency_ms": _latency_ms(relu, features),
            "near_zero_preactivation_fraction": near_threshold,
            **relu_stats,
            "fp8": _fp8(relu),
        },
        "tropical": {
            "parameter_bytes": tropical_bytes,
            "parameters": tropical_params,
            "train_seconds": tropical_seconds,
            "final_loss": tropical_loss,
            "infer_latency_ms": _latency_ms(tropical, features),
            "maxplus_tie_fraction": tropical_tie,
            "semiring": "max-plus first layer, y_i = max_j (W_ij + x_j)",
            **tropical_stats,
            "fp8": _fp8(tropical),
        },
        "peak_vram_mib": enforce_guard(),
        "literature_target_note": "Source note states about 7 kB, 0.67 s train, and 0.003 ms inference. Those are not measurements.",
    }
    digest = write_metrics(ROOT / "runs" / "guidance_d" / "metrics.json", payload)
    print(
        f"DONE guidance_d sha256={digest} relu_ms={payload['relu']['infer_latency_ms']:.6f} "
        f"bytes={relu_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
