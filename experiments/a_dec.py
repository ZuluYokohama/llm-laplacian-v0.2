"""Experiment A: metricized DEC operator versus an unconstrained edge MLP.

The coboundary is the frozen cubical boundary. The TNO emits a potential and
the trainer applies that coboundary, so B2^T g is algebraically zero. The MLP
emits edge values and is free to leave that image.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from laplacian.device import enforce_guard, pick_5070
from laplacian.mcp_tools import cochain_builder, gpu_telemetry_and_guard, verification_gate
from laplacian.operators_torch import del_residual
from laplacian.runtime import apply_env, write_metrics


class HodgeTNO(nn.Module):
    def __init__(self, n_edges, n_vertices):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_edges, 128),
            nn.Tanh(),
            nn.Linear(128, n_vertices),
        )
        self.raw_metric = nn.Parameter(torch.zeros(n_edges))

    def metric(self):
        return F.softplus(self.raw_metric) + 1e-4

    def forward(self, observation, boundary_1):
        potential = self.encoder(observation)
        return potential @ boundary_1, self.metric()


class EdgeMLP(nn.Module):
    def __init__(self, n_edges):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_edges, 128),
            nn.Tanh(),
            nn.Linear(128, n_edges),
        )

    def forward(self, observation, _boundary):
        return self.net(observation), None


def closure_residual(edges, boundary_2):
    """||B2^T g||_1 averaged over the batch. edges is [batch, n_edges]."""
    return (edges @ boundary_2).abs().sum(dim=1).mean()


def train_one(module, boundary_1, boundary_2, steps, device, use_metric):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    last_loss = None
    last_residual = None
    started = time.perf_counter()
    peak = 0.0
    module.train()
    for step in range(steps):
        observation = torch.randn(64, boundary_1.shape[1], generator=generator, dtype=torch.float32).to(device)
        target = observation
        opt.zero_grad(set_to_none=True)
        produced, metric = module(target, boundary_1)
        if use_metric:
            weight = metric / metric.mean().clamp_min(1e-6)
            loss = (weight * (produced - target).pow(2)).mean() + (metric.mean() - 1.0).pow(2)
        else:
            loss = (produced - target).pow(2).mean()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
        with torch.no_grad():
            last_residual = float(closure_residual(produced.detach(), boundary_2).item())
        if step % 20 == 0 or step + 1 == steps:
            used = enforce_guard()
            peak = max(peak, used)
            print(
                f"PROGRESS model={module.__class__.__name__} step={step}/{steps} "
                f"loss={last_loss:.6f} residual={last_residual:.6e} vram_mib={used:.0f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "final_loss": last_loss,
        "closure_residual_l1": last_residual,
        "seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "peak_vram_mib": peak,
    }


def main():
    apply_env()
    torch.backends.cuda.matmul.allow_tf32 = False
    device, name = pick_5070()
    telemetry = gpu_telemetry_and_guard()
    built = cochain_builder(8, device=device, dtype=torch.float64)
    gate = verification_gate("nilpotency", built["nilpotency"])
    symplectic = verification_gate(
        "symplectic",
        {"residual": del_residual([0.0, 0.0], [1.0, -2.0], [2.0, -4.0], 0.5)},
    )
    b1 = built["matrices"][0].float()
    b2 = built["matrices"][1].float()
    n_vertices = b1.shape[0]
    n_edges = b1.shape[1]
    tno = HodgeTNO(n_edges, n_vertices).to(device)
    mlp = EdgeMLP(n_edges).to(device)
    tno_report = train_one(tno, b1, b2, 200, device, use_metric=True)
    mlp_report = train_one(mlp, b1, b2, 200, device, use_metric=False)
    payload = {
        "experiment": "A",
        "device_name": name,
        "telemetry": telemetry,
        "backend": "torch-fp64-matmul",
        "cells": 8,
        "nilpotency": built["nilpotency"],
        "nilpotency_gate": gate,
        "symplectic_gate": symplectic,
        "tno": tno_report,
        "mlp": mlp_report,
        "host_gpu": True,
    }
    path = ROOT / "runs" / "guidance_a" / "metrics.json"
    digest = write_metrics(path, payload)
    print(f"DONE guidance_a sha256={digest} residual={built['nilpotency']['residual_l1']}", flush=True)


if __name__ == "__main__":
    main()
