"""λ2 used by the Dirichlet band is the Fiedler value, not the spectral radius."""

import torch
from torch import nn

from laplacian.host_invariants import check
from laplacian.losses import _lambda2
from laplacian.train import resample_dead


def _path_adjacency(nodes):
    adjacency = torch.zeros(nodes, nodes)
    for index in range(nodes - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    return adjacency


def _normalized_eigenvalues(adjacency):
    degree = adjacency.sum(dim=1)
    inv = degree.clamp_min(1e-8).rsqrt()
    normal = inv[:, None] * adjacency * inv[None, :]
    lap = torch.eye(adjacency.shape[0]) - normal
    return torch.linalg.eigvalsh(lap)


def test_host_lambda2_agrees_with_the_isolated_reference():
    report = check(nodes=12)
    assert report["jovin_receipt"] is False
    assert report["host_gpu"] is False
    assert report["abs_gap"] < 1e-3
    assert report["resample_write_through"] is True


def test_lambda2_matches_the_fiedler_value_of_a_path():
    adjacency = _path_adjacency(12)
    values = _normalized_eigenvalues(adjacency)
    measured = float(_lambda2(adjacency, steps=40))
    assert abs(measured - float(values[1])) < 1e-3
    assert measured < float(values[-1]) * 0.5


class _Mlp:
    def __init__(self):
        self.neuron_mask = torch.ones(4)
        self.up = nn.Linear(2, 4, bias=False)
        self.down = nn.Linear(4, 2, bias=False)
        self.up_mask = torch.ones(4)
        self.down_mask = torch.ones(2, 4)


class _Block:
    def __init__(self):
        self.mlp = _Mlp()


class _Model:
    def __init__(self):
        self.blocks = [_Block()]


def test_resample_uses_the_given_rate_not_a_single_batch_shape():
    model = _Model()
    before = model.blocks[0].mlp.up.weight.detach().clone()
    alive = resample_dead(model, [torch.ones(4)])
    assert alive == 0
    assert torch.equal(model.blocks[0].mlp.up.weight, before)
    changed = resample_dead(model, [torch.zeros(4)])
    assert changed == 4
    assert not torch.equal(model.blocks[0].mlp.up.weight, before)
