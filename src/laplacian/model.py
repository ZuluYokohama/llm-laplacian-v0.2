"""5-layer decoder-only Transformer used by the Spectral-LLM arms."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 5
CONTEXT = 256


class RMSNorm(nn.Module):
    """RMSNorm that can be replaced by a frozen calibration scale."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("stripped", torch.zeros((), dtype=torch.int32))
        self.register_buffer("frozen_scale", torch.ones(dim))
        self.register_buffer("frozen_rms", torch.ones(()))

    def strip(self):
        with torch.no_grad():
            self.frozen_scale.copy_(self.weight.detach())
            self.stripped.fill_(1)
        self.weight.requires_grad_(False)

    def forward(self, x):
        if int(self.stripped.item()) == 1:
            return x * (self.frozen_scale / self.frozen_rms.clamp_min(self.eps))
        square = x.float().pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(square + self.eps)
        if self.training:
            mean_rms = rms.detach().mean()
            self.frozen_rms.mul_(0.99).add_(mean_rms, alpha=0.01)
        return (x.float() * torch.rsqrt(square + self.eps) * self.weight).to(dtype=x.dtype)


def apply_rope(query, key):
    """Half-rotation RoPE. query and key are [batch, heads, time, head_dim]."""
    length = query.shape[-2]
    half = query.shape[-1] // 2
    positions = torch.arange(length, device=query.device, dtype=torch.float32)
    exponents = torch.arange(half, device=query.device, dtype=torch.float32) / half
    frequency = torch.exp(-math.log(10000.0) * exponents)
    angles = positions[:, None] * frequency[None, :]
    cosine = torch.cos(angles)[None, None, :, :]
    sine = torch.sin(angles)[None, None, :, :]
    q1, q2 = query.float()[..., :half], query.float()[..., half:]
    k1, k2 = key.float()[..., :half], key.float()[..., half:]
    rotated_q = torch.cat([q1 * cosine - q2 * sine, q1 * sine + q2 * cosine], dim=-1)
    rotated_k = torch.cat([k1 * cosine - k2 * sine, k1 * sine + k2 * cosine], dim=-1)
    return rotated_q.to(dtype=query.dtype), rotated_k.to(dtype=key.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_mask", torch.ones(d_model, d_model))
        self.register_buffer("k_mask", torch.ones(d_model, d_model))
        self.register_buffer("v_mask", torch.ones(d_model, d_model))
        self.register_buffer("o_mask", torch.ones(d_model, d_model))

    def _masked(self, linear, mask):
        return linear.weight * mask

    def forward(self, x):
        batch, length, width = x.shape
        q_weight, k_weight, v_weight = self._masked(self.qkv, self._qkv_mask()).split(width, dim=0)
        query = F.linear(x, q_weight).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = F.linear(x, k_weight).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = F.linear(x, v_weight).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rope(query, key)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, value.float())
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, width)
        out = F.linear(mixed.to(dtype=x.dtype), self.proj.weight * self.o_mask)
        return out, weights

    def _qkv_mask(self):
        return torch.cat([self.q_mask, self.k_mask, self.v_mask], dim=0)


class MLP(nn.Module):
    def __init__(self, d_model, width_multiplier, topk):
        super().__init__()
        hidden = width_multiplier * d_model
        self.hidden = hidden
        self.topk = int(topk or 0)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.register_buffer("neuron_mask", torch.ones(hidden))
        self.register_buffer("up_mask", torch.ones(hidden, d_model))
        self.register_buffer("down_mask", torch.ones(d_model, hidden))

    def forward(self, x):
        alive = self.neuron_mask
        up_weight = self.up.weight * self.up_mask * alive[:, None]
        hidden = F.relu(F.linear(x, up_weight))
        fired = (hidden > 0).to(hidden.dtype)
        if 0 < self.topk < hidden.shape[-1]:
            _values, indices = torch.topk(hidden, self.topk, dim=-1)
            gate = torch.zeros_like(hidden)
            gate.scatter_(-1, indices, 1.0)
            hidden = hidden * gate
            fired = gate
        hidden = hidden * alive
        down_weight = self.down.weight * self.down_mask * alive[None, :]
        return F.linear(hidden, down_weight), hidden, fired


class Block(nn.Module):
    def __init__(self, d_model, width_multiplier, topk):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, width_multiplier, topk)

    def forward(self, x):
        attn_out, weights = self.attn(self.norm1(x))
        x = x + attn_out
        mlp_out, hidden, fired = self.mlp(self.norm2(x))
        return x + mlp_out, hidden, fired, weights


class SpectralTransformer(nn.Module):
    def __init__(self, vocab, m=4, k=0, d_model=D_MODEL, n_layers=N_LAYERS, context=CONTEXT):
        super().__init__()
        self.vocab = vocab
        self.m = m
        self.k = k
        self.d_model = d_model
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, m, k) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def strip_layer(self, index):
        block = self.blocks[index]
        block.norm1.strip()
        block.norm2.strip()

    def stripped_layers(self):
        flags = []
        for block in self.blocks:
            flags.append(int(block.norm1.stripped.item()) == 1 and int(block.norm2.stripped.item()) == 1)
        return flags

    def forward(self, tokens):
        x = self.tok(tokens)
        embedded = x
        pooled = []
        streams = []
        hiddens = []
        fired = []
        attention = []
        for block in self.blocks:
            x, hidden, fire, weights = block(x)
            streams.append(x)
            pooled.append(x.mean(dim=1))
            hiddens.append(hidden)
            fired.append(fire)
            attention.append(weights)
        final = self.norm(x)
        return {
            "logits": self.head(final),
            "embedded": embedded,
            "stream": streams,
            "pooled": pooled,
            "hidden": hiddens,
            "fired": fired,
            "attn": attention,
            "final": final,
        }

    def neuron_nnz(self):
        alive = [int(block.mlp.neuron_mask.sum().item()) for block in self.blocks]
        edges = []
        for block in self.blocks:
            edges.append(int((block.mlp.up_mask * block.mlp.neuron_mask[:, None] > 0).sum().item()))
        return {"alive": alive, "up_edges": edges}
