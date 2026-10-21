"""Frozen activation-graph check on the GTX 1660 Ti.

This process must be started with CUDA_VISIBLE_DEVICES set to the 1660.
It refuses a device whose name contains 5070. It is not part of run_queue.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from laplacian.culling import (
    adjacency_edge_index,
    correlation_adjacency,
    fiedler_from_adjacency,
    select_peripheral,
)
from laplacian.runtime import apply_env, write_metrics

MODELS = Path(r"D:\Models\Qwen")
RUNGS = {
    1: MODELS / "Qwen3-0.6B-Base",
    2: MODELS / "Qwen3-1.7B-Base",
}


def assert_frozen_device(name):
    """The 5070 belongs to the serial queue. This branch only accepts the 1660."""
    if "5070" in name or "1660" not in name:
        raise SystemExit(f"frozen branch refuses device {name}")


def memory_allows(free_mib, weight_bytes, reserve_mib=1024, overhead=4.05):
    """True when free memory covers the measured activation overhead plus a reserve.

    Rung 1 used about 4.05 MiB of device memory per MiB of safetensors on the 1660.
    A check that only adds the file size will admit a model that does not fit.
    """
    weight_mib = weight_bytes / (1024 * 1024)
    return float(free_mib) >= weight_mib * overhead + reserve_mib


def column_shuffle(activations):
    """Permute each unit independently so marginals stay and correlations break."""
    order = torch.rand(activations.shape, device=activations.device).argsort(dim=0)
    return activations.gather(0, order)


def rank_from_activations(activations):
    """Fiedler adjacency-edge index from token rows of one MLP layer."""
    count = max(int(activations.shape[0]), 1)
    mean = activations.sum(dim=0)
    second = activations.transpose(0, 1) @ activations
    adjacency = correlation_adjacency(second, mean, count)
    _vector, rayleigh, degenerate = fiedler_from_adjacency(adjacency, steps=30)
    scores = adjacency_edge_index(adjacency, _vector)
    return scores, bool(degenerate), float(rayleigh)


def quota_for(width, fraction=0.05):
    return max(1, int(width * fraction))


def _weight_bytes(folder):
    total = 0
    for path in Path(folder).glob("*.safetensors"):
        total += path.stat().st_size
    return total


def _gpu_row(name_part):
    import subprocess

    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "nvidia-smi failed")
    for line in proc.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 3 and name_part in parts[0]:
            return parts[0], float(parts[1]), float(parts[2])
    raise SystemExit(f"nvidia-smi has no device whose name contains {name_part}")


def _stories(count, context, seed):
    import numpy as np
    from laplacian.data import load_tokenizer

    path = Path(__file__).resolve().parents[2] / "data" / "tinystories_val.u16"
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    limit = len(tokens) - context - 1
    if limit <= 0:
        raise RuntimeError(f"{path} is too short")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, limit, size=count)
    gpt = load_tokenizer("gpt2")
    texts = []
    for start in starts.tolist():
        ids = tokens[int(start) : int(start) + context].astype(int).tolist()
        texts.append(gpt.decode(ids, skip_special_tokens=True))
    return texts


def _encode(tokenizer, texts, context):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=context,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _capture(model, layer, input_ids, attention_mask):
    mlp = model.model.layers[layer].mlp
    chunks = []

    def hook(_module, args, _output):
        hidden = args[0]
        intermediate = mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
        flat = intermediate.detach().float().reshape(-1, intermediate.shape[-1])
        chunks.append(flat.cpu())

    handle = mlp.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return torch.cat(chunks, dim=0)


def _cross_entropy(model, input_ids, attention_mask):
    losses = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], 2):
            batch = input_ids[start : start + 2]
            mask = attention_mask[start : start + 2]
            labels = batch.clone()
            labels = labels.masked_fill(mask == 0, -100)
            output = model(input_ids=batch, attention_mask=mask, labels=labels)
            losses.append(float(output.loss.detach().float().cpu()))
    return sum(losses) / max(len(losses), 1)


def _with_units_zeroed(model, layer, indices, input_ids, attention_mask):
    mlp = model.model.layers[layer].mlp
    original = mlp.forward

    def forward(hidden):
        intermediate = mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
        if indices.numel() > 0:
            intermediate = intermediate.clone()
            intermediate[..., indices] = 0
        return mlp.down_proj(intermediate)

    mlp.forward = forward
    try:
        return _cross_entropy(model, input_ids, attention_mask)
    finally:
        mlp.forward = original


def run(rung):
    apply_env()
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("frozen branch requires the 1660 to be visible")
    visible = torch.cuda.get_device_name(0)
    assert_frozen_device(visible)
    print(f"PROGRESS frozen device={visible}", flush=True)
    folder = RUNGS[rung]
    if rung == 2:
        prior = Path(__file__).resolve().parents[2] / "runs" / "frozen_qwen3_0p6b" / "metrics.json"
        if not prior.exists():
            raise SystemExit("rung 2 waits for a hashed rung 1 metrics file")
    weight_bytes = _weight_bytes(folder)
    smi_name, used_mib, total_mib = _gpu_row("1660")
    free_mib = total_mib - used_mib
    destination = Path(__file__).resolve().parents[2] / "runs" / (
        "frozen_qwen3_0p6b" if rung == 1 else "frozen_qwen3_1p7b"
    )
    if not memory_allows(free_mib, weight_bytes):
        payload = {
            "run_id": destination.name,
            "complete": False,
            "host_gpu": True,
            "device_name": smi_name,
            "error": "free memory below the weight plus reserve",
            "free_mib": free_mib,
            "weight_bytes": weight_bytes,
            "literature_target": "source PDF validation loss 1.466 and AutoInterp 0.866 were not measured",
        }
        digest = write_metrics(destination / "metrics.json", payload)
        print(f"FAILED {destination.name} sha256={digest} reason=memory", flush=True)
        raise SystemExit(1)
    started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        folder,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.to("cuda")
    model.eval()
    print(f"PROGRESS frozen loaded layer={int(model.config.num_hidden_layers) // 2}", flush=True)
    context = 128
    layer = int(model.config.num_hidden_layers) // 2
    calibration_text = _stories(64, context, seed=0)
    evaluation_text = _stories(32, context, seed=1)
    calibration_ids, calibration_mask = _encode(tokenizer, calibration_text, context)
    evaluation_ids, evaluation_mask = _encode(tokenizer, evaluation_text, context)
    activations = _capture(
        model, layer, calibration_ids.to("cuda"), calibration_mask.to("cuda")
    ).to("cuda")
    scores, degenerate, rayleigh = rank_from_activations(activations)
    shuffled_scores, shuffled_degenerate, shuffled_rayleigh = rank_from_activations(
        column_shuffle(activations)
    )
    width = int(scores.shape[0])
    quota = quota_for(width)
    alive = torch.ones(width, dtype=torch.bool)
    peripheral = select_peripheral(scores.cpu(), alive, quota)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    random_units = torch.randperm(width, generator=generator)[: peripheral.numel()]
    column_norm = model.model.layers[layer].mlp.down_proj.weight.detach().float().norm(dim=0)
    magnitude = torch.topk(column_norm.cpu(), peripheral.numel(), largest=False).indices
    evaluation_ids = evaluation_ids.to("cuda")
    evaluation_mask = evaluation_mask.to("cuda")
    baseline = _cross_entropy(model, evaluation_ids, evaluation_mask)
    peripheral_loss = _with_units_zeroed(model, layer, peripheral.to("cuda"), evaluation_ids, evaluation_mask)
    random_loss = _with_units_zeroed(model, layer, random_units.to("cuda"), evaluation_ids, evaluation_mask)
    magnitude_loss = _with_units_zeroed(model, layer, magnitude.to("cuda"), evaluation_ids, evaluation_mask)
    finite = all(
        value == value
        for value in (baseline, peripheral_loss, random_loss, magnitude_loss, rayleigh, shuffled_rayleigh)
    )
    _smi_name, used_after, _total = _gpu_row("1660")
    payload = {
        "run_id": destination.name,
        "complete": bool(finite),
        "host_gpu": True,
        "device_name": visible,
        "smi_name": smi_name,
        "runtime_dtype": "float16",
        "weight_file_dtype": "bfloat16",
        "model_path": str(folder),
        "layer": layer,
        "width": width,
        "units_zeroed": int(peripheral.numel()),
        "calibration_sequences": 64,
        "evaluation_sequences": 32,
        "context": context,
        "activation_rows": int(activations.shape[0]),
        "degenerate": degenerate,
        "shuffled_degenerate": shuffled_degenerate,
        "rayleigh": rayleigh,
        "shuffled_rayleigh": shuffled_rayleigh,
        "mean_aei": float(scores.mean().cpu()),
        "mean_shuffled_aei": float(shuffled_scores.mean().cpu()),
        "ce_baseline": baseline,
        "ce_peripheral": peripheral_loss,
        "ce_random": random_loss,
        "ce_magnitude": magnitude_loss,
        "peripheral_minus_random": peripheral_loss - random_loss,
        "seconds": time.perf_counter() - started,
        "vram_used_mib_before": used_mib,
        "vram_used_mib_after": used_after,
        "weight_bytes": weight_bytes,
        "literature_target": "source PDF validation loss 1.466 and AutoInterp 0.866 were not measured",
        "jovin_receipt": False,
    }
    digest = write_metrics(destination / "metrics.json", payload)
    print(
        f"DONE {destination.name} sha256={digest} device={visible} "
        f"ce_base={baseline:.4f} peripheral_minus_random={peripheral_loss - random_loss:.4f}",
        flush=True,
    )
    if not finite:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    run(args.rung)


if __name__ == "__main__":
    main()
