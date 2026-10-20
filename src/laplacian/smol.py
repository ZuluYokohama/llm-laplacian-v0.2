"""SmolLM2-135M-Instruct SFT. The cached 1.7B model is not a substitute."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from laplacian.culling import (
    adjacency_edge_index,
    correlation_adjacency,
    fiedler_from_adjacency,
    select_peripheral,
    sparsify_matrix,
    suzuki_recalibrate,
)
from laplacian.data import DATA, TokenStore, load_tokenizer, snapshot_dir
from laplacian.device import enforce_guard, memory_mib, pick_5070
from laplacian.losses import _affinity, _quadratic
from laplacian.runtime import RUNS, apply_env, read_json, write_metrics
from laplacian.schedule import gammas
from laplacian.train import learning_rate, log_line, read_meta, restore_generator

REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
PULSES = [60000, 120000, 180000, 240000]
CROSS = (8, 16, 30)


def _refuse(config):
    layers = int(config.num_hidden_layers)
    hidden = int(config.hidden_size)
    name = str(getattr(config, "_name_or_path", REPO))
    if "1.7B" in name or layers != 30 or hidden > 1024:
        raise SystemExit(f"refusing {name}: layers={layers} hidden={hidden}. Required SmolLM2-135M.")


def _load_model(device):
    from transformers import AutoConfig, AutoModelForCausalLM

    local = snapshot_dir(REPO)
    if local is not None:
        config = AutoConfig.from_pretrained(str(local), local_files_only=True)
        _refuse(config)
        model = AutoModelForCausalLM.from_pretrained(
            str(local), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    else:
        config = AutoConfig.from_pretrained(REPO)
        _refuse(config)
        model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.to(device)
    model.gradient_checkpointing_enable()
    return model


def _install(model):
    bank = []
    for layer in model.model.layers:
        mlp = layer.mlp
        inter = mlp.up_proj.out_features
        hidden = mlp.up_proj.in_features
        masks = {
            "neuron": torch.ones(inter, device=mlp.up_proj.weight.device),
            "up": torch.ones(inter, hidden, device=mlp.up_proj.weight.device),
            "gate": torch.ones_like(mlp.gate_proj.weight),
            "down": torch.ones_like(mlp.down_proj.weight),
            "capture": False,
            "sum": torch.zeros(inter, device=mlp.up_proj.weight.device),
            "gram": torch.zeros(inter, inter, device=mlp.up_proj.weight.device),
            "tokens": 0,
        }

        def forward(x, mlp=mlp, masks=masks):
            gate_w = mlp.gate_proj.weight * masks["gate"] * masks["neuron"][:, None]
            up_w = mlp.up_proj.weight * masks["up"] * masks["neuron"][:, None]
            down_w = mlp.down_proj.weight * masks["down"] * masks["neuron"][None, :]
            gated = mlp.act_fn(F.linear(x, gate_w, mlp.gate_proj.bias)) * F.linear(x, up_w, mlp.up_proj.bias)
            if masks["capture"]:
                flat = gated.detach().float().reshape(-1, gated.shape[-1])
                masks["sum"] = masks["sum"] + flat.sum(dim=0)
                masks["gram"] = masks["gram"] + flat.transpose(0, 1) @ flat
                masks["tokens"] += int(flat.shape[0])
            return F.linear(gated, down_w, mlp.down_proj.bias)

        mlp.forward = forward
        bank.append(masks)
    return bank


def _manifold(hidden_states, step):
    table = gammas(step)
    if table["manifold"] == 0 and table["fractional"] == 0:
        return hidden_states[0].new_zeros(())
    chosen = [hidden_states[index].mean(dim=1).float() for index in CROSS]
    total = chosen[0].new_zeros(())
    pairs = 0
    for source in chosen:
        lap = _affinity(source, 1.0)
        for target in chosen:
            count = target.shape[0]
            total = total + _quadratic(lap, target) / (count * count)
            pairs += 1
    return table["manifold"] * total / max(pairs, 1)


class _Down:
    def __init__(self, parameter):
        self.weight = parameter


def _pulse_layer(mlp, masks, fraction):
    count = max(int(masks["tokens"]), 1)
    adjacency = correlation_adjacency(masks["gram"], masks["sum"], count)
    fiedler, lam, degenerate = fiedler_from_adjacency(adjacency, steps=30)
    report = {"lambda2": lam, "degenerate": bool(degenerate)}
    if not degenerate:
        quota = max(1, int(math.floor(mlp.up_proj.out_features * fraction)))
        alive = masks["neuron"] > 0
        remove = select_peripheral(adjacency_edge_index(adjacency, fiedler), alive, quota)
        survivors = torch.where(alive)[0]
        if remove.numel():
            drop = torch.zeros(mlp.up_proj.out_features, dtype=torch.bool, device=remove.device)
            drop[remove] = True
            survivors = survivors[~drop[survivors]]
        report["suzuki"] = suzuki_recalibrate(_Down(mlp.down_proj), masks["gram"] / count, survivors, 1e-3)
        with torch.no_grad():
            masks["neuron"][remove] = 0
            mlp.up_proj.weight[remove] = 0
            mlp.gate_proj.weight[remove] = 0
            mlp.down_proj.weight[:, remove] = 0
        report["removed"] = int(remove.numel())
    else:
        report["removed"] = 0
    report["up"] = sparsify_matrix(mlp.up_proj.weight, masks["up"], 0.15, 0.40, "sample", "smol up")
    report["gate"] = sparsify_matrix(mlp.gate_proj.weight, masks["gate"], 0.15, 0.40, "sample", "smol gate")
    report["down"] = sparsify_matrix(mlp.down_proj.weight, masks["down"], 0.15, 0.40, "sample", "smol down")
    masks["sum"].zero_()
    masks["gram"].zero_()
    masks["tokens"] = 0
    masks["capture"] = False
    return report


def _calibrate(model, bank, store, generator, device, sequences):
    for masks in bank:
        masks["capture"] = True
        masks["sum"].zero_()
        masks["gram"].zero_()
        masks["tokens"] = 0
    seen = 0
    model.eval()
    with torch.no_grad():
        while seen < sequences:
            inputs, _targets = store.batch(2, generator)
            model(inputs.to(device), output_hidden_states=False, output_attentions=False)
            seen += int(inputs.shape[0])
            log_line(f"PULSE calibrate sequences={seen}/{sequences}")
    for masks in bank:
        masks["capture"] = False
    model.train()


def _strip_norms(model, bank_unused, step, done, store, generator, device):
    del bank_unused
    layers = len(model.model.layers)
    interval = 300_000 // layers
    due = {(index + 1) * interval: index for index in range(layers)}
    due.pop(300_000, None)
    due[299_000] = layers - 1
    updated = list(done)
    for when, index in due.items():
        if step < when or index in updated:
            continue
        layer = model.model.layers[index]
        inputs, _targets = store.batch(2, generator)
        with torch.no_grad():
            probe = model.model.embed_tokens(inputs.to(device))
            frozen = probe.float().pow(2).mean().sqrt().clamp_min(1e-6)
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            weight = norm.weight.detach().clone()
            scale = frozen.detach()

            def forward(x, weight=weight, scale=scale):
                return x * (weight / scale)

            norm.forward = forward
            norm.weight.requires_grad_(False)
        updated.append(index)
        log_line(f"PROGRESS smol_norm_strip layer={index} step={step}")
    return updated


def _save(path, model, optimizer, generator, bank, step, pulses, stripped):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "bank": [{key: value for key, value in masks.items() if torch.is_tensor(value)} for masks in bank],
        "pulses": pulses,
        "stripped": stripped,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_bank(bank, saved):
    for masks, previous in zip(bank, saved):
        for key, value in previous.items():
            if key in masks and torch.is_tensor(masks[key]) and torch.is_tensor(value):
                masks[key].copy_(value.to(masks[key].device))


def train():
    metrics = RUNS / "arm5" / "metrics.json"
    if metrics.exists() and read_json(metrics).get("complete"):
        log_line("SKIP arm5 complete")
        return
    apply_env()
    device, name = pick_5070()
    meta = read_meta()
    if "smol_train_tokens" not in meta:
        raise SystemExit("SmolLM tokenizer memmap is missing from data/meta.json")
    store = TokenStore(DATA / "tinystories_smol_train.u16", context=256)
    model = _load_model(device)
    bank = _install(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    folder = RUNS / "arm5"
    folder.mkdir(parents=True, exist_ok=True)
    step = 0
    pulses = []
    stripped = []
    last = folder / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        bank = _install(model)
        _restore_bank(bank, checkpoint["bank"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_generator(generator, checkpoint["generator"])
        step = int(checkpoint["step"])
        pulses = list(checkpoint["pulses"])
        stripped = list(checkpoint["stripped"])
        log_line(f"PROGRESS resume arm5 step={step}")
    total = 300_000
    microbatch = 2
    accum = 8
    started = time.perf_counter()
    started_step = step
    peak = memory_mib()
    last_loss = 0.0
    while step < total:
        stripped = _strip_norms(model, bank, step, stripped, store, generator, device)
        rate = learning_rate(step, total, 2000, 3e-4, 3e-5)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _micro in range(accum):
            inputs, _targets = store.batch(microbatch, generator)
            inputs = inputs.to(device)
            outputs = model(
                inputs,
                labels=inputs,
                output_hidden_states=True,
                output_attentions=False,
            )
            extra = _manifold(outputs.hidden_states, step)
            loss = outputs.loss + extra
            (loss / accum).backward()
            loss_sum += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        last_loss = loss_sum / accum
        finite = math.isfinite(last_loss)
        if step in PULSES and step not in pulses:
            _calibrate(model, bank, store, generator, device, 512)
            reports = []
            for index, layer in enumerate(model.model.layers):
                reports.append(_pulse_layer(layer.mlp, bank[index], 0.075))
                log_line(f"PULSE layer={index} nodes={reports[-1]['removed']}")
            pulses.append(step)
        if step % 10 == 0 or not finite:
            used = memory_mib()
            peak = max(peak, used)
            log_line(f"STEP {step}/{total} loss={last_loss:.6f} ce={last_loss:.6f} vram_mb={used:.0f} nan={0 if finite else 1}")
        if not finite:
            _save(last, model, optimizer, generator, bank, step, pulses, stripped)
            raise SystemExit(1)
        if step % 1000 == 0:
            _save(last, model, optimizer, generator, bank, step, pulses, stripped)
        if step % 10 == 0:
            enforce_guard()
    _save(last, model, optimizer, generator, bank, step, pulses, stripped)
    torch.save({"step": step, "model": model.state_dict(), "bank": [{k: v for k, v in m.items() if torch.is_tensor(v)} for m in bank]}, folder / "weights_final.pt")
    elapsed = time.perf_counter() - started
    payload = {
        "run_id": "arm5",
        "complete": True,
        "host_gpu": True,
        "device_name": name,
        "model": REPO,
        "steps": step,
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(step - started_step, 1),
        "microbatch": microbatch,
        "accum": accum,
        "tokens_per_step": microbatch * accum * 256,
        "final_loss": last_loss,
        "peak_vram_mib": peak,
        "pulses": pulses,
        "cross_layer_hidden_state_index": list(CROSS),
        "note": "Hugging Face causal LM loss shifts labels internally. tokens_per_step counts input tokens.",
    }
    digest = write_metrics(metrics, payload)
    log_line(f"DONE arm5 sha256={digest}")


def _checks(text, specs):
    results = []
    for spec in specs:
        kind = spec["type"]
        words = text.split()
        if kind == "max_words":
            ok = len(words) <= int(spec["value"])
        elif kind == "min_words":
            ok = len(words) >= int(spec["value"])
        elif kind == "must_include":
            ok = str(spec["value"]).lower() in text.lower()
        elif kind == "must_exclude":
            ok = str(spec["value"]).lower() not in text.lower()
        elif kind == "lowercase":
            ok = text == text.lower()
        elif kind == "uppercase":
            ok = text == text.upper()
        elif kind == "json_key":
            try:
                start = text.find("{")
                end = text.rfind("}")
                obj = json.loads(text[start : end + 1])
                ok = spec["value"] in obj
            except (ValueError, json.JSONDecodeError):
                ok = False
        elif kind == "sentences":
            parts = [part for part in text.replace("!", ".").replace("?", ".").split(".") if part.strip()]
            ok = len(parts) == int(spec["value"])
        else:
            ok = False
            kind = "unsupported"
        results.append({"type": kind, "pass": bool(ok)})
    return results


def _generate(model, tokenizer, prompt, device, new_tokens):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors="pt").to(device)
    start = torch.cuda.Event(enable_timing=True)
    first = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=new_tokens, do_sample=False)
    end.record()
    torch.cuda.synchronize()
    produced = out[0, tokens["input_ids"].shape[1] :]
    # A one-token prefill timing, separate from the full generate.
    first.record()
    with torch.no_grad():
        model(tokens["input_ids"], use_cache=True)
    stop = torch.cuda.Event(enable_timing=True)
    stop.record()
    torch.cuda.synchronize()
    ttft = float(first.elapsed_time(stop))
    seconds = float(start.elapsed_time(end)) / 1000.0
    return tokenizer.decode(produced, skip_special_tokens=True), ttft, produced.shape[0] / max(seconds, 1e-6)


def evaluate():
    apply_env()
    device, name = pick_5070()
    path = RUNS / "arm5" / "weights_final.pt"
    if not path.exists():
        raise SystemExit("arm5 weights are missing")
    tokenizer = load_tokenizer(REPO)
    model = _load_model(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    bank = _install(model)
    _restore_bank(bank, checkpoint["bank"])
    model.eval()
    rows = []
    with (DATA / "ifeval_local.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    checked = []
    passed = 0
    total = 0
    for row in rows:
        text, _ttft, _rate = _generate(model, tokenizer, row["prompt"], device, 64)
        results = _checks(text, row["checks"])
        checked.append({"prompt": row["prompt"], "output": text, "checks": results})
        for item in results:
            total += 1
            passed += int(item["pass"])
    commonsense = json.loads((DATA / "commonsense.json").read_text(encoding="utf-8"))
    hits = 0
    samples = []
    for row in commonsense:
        text, _ttft, _rate = _generate(model, tokenizer, row["prompt"], device, 16)
        ok = row["answer"].lower() in text.lower()
        hits += int(ok)
        if len(samples) < 5:
            samples.append({"prompt": row["prompt"], "output": text, "hit": ok})
    speeds = []
    for length in (200, 400, 600, 800):
        _text, ttft, rate = _generate(model, tokenizer, "Tell a very short story about a kite.", device, length)
        speeds.append({"new_tokens": length, "ttft_ms": ttft, "output_tokens_per_second": rate})
        log_line(f"PROGRESS arm5_speed tokens={length} otps={rate:.3f}")
    payload = {
        "run_id": "arm5_eval",
        "complete": True,
        "host_gpu": True,
        "device_name": name,
        "model": REPO,
        "ifeval_source": "data/ifeval_local.jsonl",
        "ifeval_official": False,
        "ifeval_checks_passed": passed,
        "ifeval_checks_total": total,
        "ifeval_accuracy": passed / max(total, 1),
        "ifeval_rows": checked,
        "commonsense_source": "data/commonsense.json",
        "commonsense_hits": hits,
        "commonsense_total": len(commonsense),
        "commonsense_accuracy": hits / max(len(commonsense), 1),
        "commonsense_samples": samples,
        "speed": speeds,
        "ttft_definition": "eager prefill of the prompt, timed after the generate call",
        "judge_model": None,
    }
    digest = write_metrics(RUNS / "arm5_eval" / "metrics.json", payload)
    log_line(f"DONE arm5_eval sha256={digest} ifeval={payload['ifeval_accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.eval:
        evaluate()
    else:
        raise SystemExit("pass --train or --eval")


if __name__ == "__main__":
    main()
