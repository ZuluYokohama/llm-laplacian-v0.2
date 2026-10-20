"""Host checks for token accounting, JL sketch shapes, and QKV write-through."""

import torch

from laplacian.culling import collect_moments, resistance_sketch
from laplacian.model import SpectralTransformer
from laplacian.operators_torch import jl_dimension


def test_collect_moments_counts_tokens_not_the_last_width():
    model = SpectralTransformer(vocab=32, m=1, k=0, d_model=32, n_layers=1, context=8)
    batches = [torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (1, 4))]
    _grams, _sums, count = collect_moments(model, iter(batches), sequences=3, microbatch=1)
    assert count == 2 * 8 + 1 * 4


def test_resistance_sketch_cross_term_has_matrix_shape():
    weight = torch.randn(6, 4)
    mask = torch.ones(6, 4)
    resistance, width = resistance_sketch(weight, mask, eps=1.5, batch_columns=4, log_prefix="sketch")
    assert resistance.shape == (6, 4)
    assert width == jl_dimension(10, 1.5)
    assert torch.isfinite(resistance).all()


def test_layer_pulse_updates_qkv_and_neurons():
    from laplacian.culling import layer_pulse

    model = SpectralTransformer(vocab=20, m=1, k=0, d_model=8, n_layers=1, context=4)
    batch = torch.randint(0, 20, (4, 4))
    grams, sums, count = collect_moments(model, iter([batch, batch]), sequences=8, microbatch=4)
    before = model.blocks[0].attn.qkv.weight.detach().clone()
    report = layer_pulse(
        model.blocks[0], grams[0], sums[0], count, quota=1, eps=2.0, retain_floor=0.25, mode="topk", layer_index=0
    )
    assert report["removed"] >= 0
    assert int(model.blocks[0].mlp.neuron_mask.sum().item()) <= model.blocks[0].mlp.hidden
    assert model.blocks[0].attn.qkv.weight.shape == before.shape
    assert torch.isfinite(model.blocks[0].attn.qkv.weight).all()


def test_qkv_row_slice_writes_through_to_the_parameter():
    attention = SpectralTransformer(vocab=16, m=1, k=0, d_model=8, n_layers=1).blocks[0].attn
    before_tail = attention.qkv.weight[8:].detach().clone()
    with torch.no_grad():
        attention.qkv.weight[:8].copy_(torch.zeros_like(attention.qkv.weight[:8]))
    assert torch.count_nonzero(attention.qkv.weight[:8]) == 0
    assert torch.allclose(attention.qkv.weight[8:], before_tail)


def test_zero_row_sketch_uses_live_order_and_stays_finite():
    weight = torch.rand(6, 4) + 0.1
    weight[0] = 0
    mask = torch.ones(6, 4)
    resistance, width = resistance_sketch(weight, mask, eps=1.5, batch_columns=4, log_prefix="zero-row")
    assert resistance.shape == (6, 4)
    assert width == jl_dimension(9, 1.5)
    assert torch.isfinite(resistance).all()
    assert torch.count_nonzero(resistance[0]) == 0


def test_cg_restores_the_callers_tf32_flag():
    from laplacian.culling import _cg_bipartite

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    weight = torch.rand(5, 4) + 0.1
    rhs_src = torch.randn(4, 2)
    rhs_dst = torch.randn(5, 2)
    _src, _dst, residual = _cg_bipartite(weight, rhs_src, rhs_dst, maxiter=8)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert residual == residual
    torch.backends.cuda.matmul.allow_tf32 = previous


def test_cull_pulse_keeps_each_layers_moments(monkeypatch):
    import laplacian.culling as culling

    model = SpectralTransformer(vocab=20, m=1, k=0, d_model=8, n_layers=3, context=4)
    batches = [torch.randint(0, 20, (2, 4))]
    grams, sums, _count = culling.collect_moments(model, iter(batches), sequences=2, microbatch=1)
    expected = [(float(grams[i].sum()), float(sums[i].sum())) for i in range(3)]
    seen = []

    def stub(block, second, mean, count, quota, eps, retain_floor, mode, layer_index, tau=1e-3):
        seen.append((layer_index, float(second.sum()), float(mean.sum())))
        return {"layer": layer_index}

    monkeypatch.setattr(culling, "layer_pulse", stub)
    reports = culling.cull_pulse(
        model, iter(batches), sequences=2, fraction=0.05, eps=2.0, retain_floor=0.25, mode="topk", microbatch=1
    )
    assert [row[0] for row in seen] == [0, 1, 2]
    assert len(reports) == 3
    for index, (layer, gram_sum, mean_sum) in enumerate(seen):
        assert layer == index
        assert gram_sum == expected[index][0]
        assert mean_sum == expected[index][1]


def test_sample_mode_keeps_weights_finite_when_resistance_is_not(monkeypatch):
    import laplacian.culling as culling

    def fake_sketch(weight, mask, eps, batch_columns, log_prefix):
        return torch.full(weight.shape, float("nan")), 3

    monkeypatch.setattr(culling, "resistance_sketch", fake_sketch)
    weight = torch.randn(4, 3)
    mask = torch.ones(4, 3)
    before = weight.detach().clone()
    report = culling.sparsify_matrix(
        weight, mask, eps=1.0, retain_floor=0.25, mode="sample", log_prefix="degenerate", batch_columns=2
    )
    assert report["degenerate_sample"] is True
    assert torch.isfinite(weight).all()
    assert int(mask.sum().item()) > 0
    assert torch.isfinite(before).all()
