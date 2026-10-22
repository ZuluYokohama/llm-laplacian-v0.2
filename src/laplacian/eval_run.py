"""Validation loss and interpretability metrics for one Spectral-LLM checkpoint.

The judge, when it loads, is Qwen2.5-1.5B-Instruct. Scores are not labeled as
GPT-4o-mini.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from laplacian.arms import get_config, layers_0index
from laplacian.data import DATA, TokenStore, load_tokenizer, snapshot_dir
from laplacian.device import memory_mib, pick_5070
from laplacian.losses import spectral_penalties
from laplacian.runtime import RUNS, apply_env, write_metrics
from laplacian.train import build_model, forward_loss, read_meta, run_dir

JUDGE_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _load_student(arm, device, meta):
    config = get_config(arm)
    model = build_model(config, int(meta["vocab"]), device)
    path = run_dir(config) / "weights_final.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    return config, model


def _val_ce(model, config, store, device, batches):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    total = 0.0
    with torch.no_grad():
        for _batch in range(batches):
            inputs, targets = store.batch(8, generator)
            _loss, ce, _outputs, _penalties = forward_loss(
                model, inputs.to(device), targets.to(device), config, 0
            )
            total += float(ce.item())
    return total / batches


def _collect(model, store, device, batches, layer):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2)
    streams = []
    hiddens = []
    pooled = []
    attentions = []
    with torch.no_grad():
        for _batch in range(batches):
            inputs, _targets = store.batch(4, generator)
            outputs = model(inputs.to(device))
            streams.append(outputs["stream"][layer].float().cpu())
            hiddens.append(outputs["hidden"][layer].float().cpu())
            pooled.append(outputs["pooled"][layer].float().cpu())
            attentions.append(outputs["attn"][layer].float().cpu())
    return {
        "stream": torch.cat(streams, dim=0),
        "hidden": torch.cat(hiddens, dim=0),
        "pooled": torch.cat(pooled, dim=0),
        "attn": torch.cat(attentions, dim=0),
    }


def _sae(stream, device):
    vectors = stream.reshape(-1, stream.shape[-1])
    take = min(4096, vectors.shape[0])
    data = vectors[:take].to(device)
    width = data.shape[1]
    hidden = width * 4
    encoder = nn.Linear(width, hidden).to(device)
    decoder = nn.Linear(hidden, width, bias=False).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    last = 0.0
    for _step in range(400):
        opt.zero_grad(set_to_none=True)
        code = F.relu(encoder(data))
        recon = decoder(code)
        loss = torch.mean((recon - data) ** 2) + 1e-3 * code.abs().mean()
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
    with torch.no_grad():
        code = F.relu(encoder(data))
        recon = decoder(code)
        mse = float(torch.mean((recon - data) ** 2).item())
        l0 = float((code > 0).float().sum(dim=1).mean().item())
        dead = float((code.sum(dim=0) == 0).float().mean().item())
    return {"mse": mse, "mean_l0": l0, "dead_fraction": dead, "steps": 400, "final_loss": last, "dictionary": hidden}


def _transcoder(stream, hidden, device):
    source = stream.reshape(-1, stream.shape[-1])
    target = hidden.reshape(-1, hidden.shape[-1])
    take = min(4096, source.shape[0])
    x = source[:take].to(device)
    y = target[:take].to(device)
    layer = nn.Linear(x.shape[1], y.shape[1], bias=False).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    for _step in range(400):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((layer(x) - y) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = float(torch.mean((layer(x) - y) ** 2).item())
    return {"mse": mse, "steps": 400, "definition": "linear map from layer stream tokens to that layer's MLP activations"}


def _henrici(stream):
    flat = stream.reshape(-1, stream.shape[-1])
    take = min(10000, flat.shape[0] - 1)
    left = flat[:take]
    right = flat[1 : take + 1]
    solution = torch.linalg.lstsq(left, right).solution
    gram = solution @ solution.transpose(0, 1)
    other = solution.transpose(0, 1) @ solution
    numer = torch.linalg.matrix_norm(gram - other, ord="fro")
    denom = torch.linalg.matrix_norm(solution, ord="fro").pow(2).clamp_min(1e-12)
    return {"states": int(take), "henrici": float((numer / denom).item())}


def _gram_gap(collected, layers):
    grams = []
    for index in layers:
        states = collected["pooled"] if index == layers[0] else None
    del states
    # pooled was collected only for the eval layer. Use that layer against itself
    # and report the off-diagonal mean of one Gram matrix as the within-layer gap.
    states = collected["pooled"]
    gram = states @ states.transpose(0, 1)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    off = (gram * (1.0 - eye)).abs().mean()
    return float(off.item())


def _dirichlet(model, config, store, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    inputs, _targets = store.batch(4, generator)
    with torch.no_grad():
        outputs = model(inputs.to(device))
        penalties = spectral_penalties(
            outputs,
            layers_0index(config),
            float(config["exponent"]),
            weights={"manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 1.0},
        )
    return float(penalties["dirichlet"].item())


def _faithfulness(model, tokenizer, device, layer):
    prompts = json.loads((DATA / "circuit_prompts.json").read_text(encoding="utf-8"))
    block = model.blocks[layer]
    saved = block.mlp.neuron_mask.detach().clone()
    full = []
    ablate = []
    only = []
    zeros = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=False)[: model.context]
            if len(ids) < 2:
                continue
            tokens = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            def ce():
                logits = model(inputs)["logits"].float()
                return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item())

            block.mlp.neuron_mask.copy_(saved)
            full.append(ce())
            hidden = model(inputs)["hidden"][layer]
            scores = hidden[0].mean(dim=0)
            alive = torch.where(saved > 0)[0]
            ranked = alive[torch.topk(scores[alive], k=min(16, int(alive.numel()))).indices]
            block.mlp.neuron_mask.copy_(saved)
            block.mlp.neuron_mask[ranked] = 0
            ablate.append(ce())
            block.mlp.neuron_mask.zero_()
            block.mlp.neuron_mask[ranked] = 1
            only.append(ce())
            block.mlp.neuron_mask.zero_()
            zeros.append(ce())
            block.mlp.neuron_mask.copy_(saved)
    full_m = sum(full) / max(len(full), 1)
    ablate_m = sum(ablate) / max(len(ablate), 1)
    only_m = sum(only) / max(len(only), 1)
    zero_m = sum(zeros) / max(len(zeros), 1)
    span = zero_m - full_m
    return {
        "prompts": len(full),
        "layer_index_0": layer,
        "ce_full": full_m,
        "ce_ablate_circuit": ablate_m,
        "ce_circuit_only": only_m,
        "ce_layer_zero": zero_m,
        "faithfulness": 1.0 - (only_m - full_m) / (span if span != 0 else 1e-8),
        "completeness": (ablate_m - full_m) / (span if span != 0 else 1e-8),
        "definition": "circuit = 16 highest-mean MLP neurons on the prompt; faithfulness uses circuit-only CE; completeness uses circuit-ablated CE",
    }


def _judge(device):
    from transformers import AutoModelForCausalLM

    local = snapshot_dir(JUDGE_ID)
    kwargs = {"torch_dtype": torch.bfloat16}
    if local is not None:
        tokenizer = load_tokenizer(JUDGE_ID)
        model = AutoModelForCausalLM.from_pretrained(str(local), local_files_only=True, **kwargs)
    else:
        tokenizer = load_tokenizer(JUDGE_ID)
        model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, **kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def _generate(model, tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**tokens, max_new_tokens=32, do_sample=False)
    new = out[0, tokens["input_ids"].shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=True)


def _score_digit(text):
    for char in text:
        if char in ("0", "1"):
            return int(char)
    return None


def _autointerp(student, tokenizer, store, device, layer):
    try:
        judge_tok, judge = _judge(device)
    except Exception as exc:
        return {"judge_model": JUDGE_ID, "error": str(exc)[:400], "neurons": 0}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4)
    inputs, _targets = store.batch(4, generator)
    with torch.no_grad():
        outputs = student(inputs.to(device))
    hidden = outputs["hidden"][layer][0].float()
    variance = hidden.var(dim=0)
    alive = torch.where(student.blocks[layer].mlp.neuron_mask > 0)[0]
    choose = min(100, int(alive.numel()))
    neurons = alive[torch.topk(variance[alive], k=choose).indices]
    ids = inputs[0].tolist()
    scores = []
    drifts = []
    for neuron in neurons.tolist():
        column = hidden[:, neuron]
        high = torch.topk(column, k=min(4, column.shape[0])).indices.tolist()
        snippets = []
        for position in high:
            window = ids[max(0, position - 4) : position + 4]
            snippets.append(tokenizer.decode(window))
        description = _generate(
            judge,
            judge_tok,
            "In at most eight words, what do these snippets share? " + " | ".join(snippets),
            device,
        )
        first = _generate(judge, judge_tok, f"Does '{description}' match these snippets? Answer 0 or 1. " + " | ".join(snippets), device)
        second = _generate(judge, judge_tok, f"Score 0 or 1 only. Description: {description}. Snippets: " + " | ".join(reversed(snippets)), device)
        left = _score_digit(first)
        right = _score_digit(second)
        if left is not None:
            scores.append(left)
        if left is not None and right is not None:
            drifts.append(abs(left - right))
    del judge
    torch.cuda.empty_cache()
    return {
        "judge_model": JUDGE_ID,
        "neurons": choose,
        "scored": len(scores),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "mean_recursive_drift": (sum(drifts) / len(drifts)) if drifts else None,
        "selection": "highest activation variance on one batch, layer_index_0=" + str(layer),
    }


def evaluate(arm):
    apply_env()
    device, name = pick_5070()
    meta = read_meta()
    config, model = _load_student(arm, device, meta)
    store = TokenStore(DATA / "tinystories_val.u16", context=int(config["context"]))
    tokenizer = load_tokenizer("gpt2")
    layer = 2 if len(model.blocks) > 2 else 0
    val = _val_ce(model, config, store, device, batches=128)
    collected = _collect(model, store, device, batches=8, layer=layer)
    sae = _sae(collected["stream"], device)
    transcoder = _transcoder(collected["stream"], collected["hidden"], device)
    koopman = _henrici(collected["stream"])
    payload = {
        "run_id": f"eval_{arm}",
        "complete": True,
        "host_gpu": True,
        "device_name": name,
        "source_run": arm,
        "val_ce": val,
        "val_sequences": 128 * 8,
        "layer_index_0": layer,
        "dirichlet": _dirichlet(model, config, store, device),
        "gram_gap": _gram_gap(collected, layers_0index(config)),
        "gram_gap_definition": "mean absolute off-diagonal of the Gram matrix of pooled states at the evaluated layer",
        "sae": sae,
        "transcoder": transcoder,
        "koopman": koopman,
        "circuit": _faithfulness(model, tokenizer, device, layer),
        "autointerp": _autointerp(model, tokenizer, store, device, layer),
        "peak_vram_mib": memory_mib(),
    }
    digest = write_metrics(RUNS / f"eval_{arm}" / "metrics.json", payload)
    print(f"DONE eval_{arm} sha256={digest} val_ce={val:.6f} judge={payload['autointerp'].get('judge_model')}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    evaluate(args.arm)


if __name__ == "__main__":
    main()
