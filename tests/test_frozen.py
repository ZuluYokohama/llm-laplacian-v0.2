import torch

from laplacian.frozen import assert_frozen_device, column_shuffle, memory_allows, quota_for, rank_from_activations


def test_device_guard_rejects_the_5070():
    try:
        assert_frozen_device("NVIDIA GeForce RTX 5070")
    except SystemExit as exc:
        assert "5070" in str(exc)
    else:
        raise AssertionError("5070 was accepted")
    try:
        assert_frozen_device("NVIDIA GeForce RTX 1660 Ti")
    except SystemExit:
        raise AssertionError("1660 was refused")


def test_memory_guard_keeps_a_reserve():
    assert memory_allows(8000, 1200 * 1024 * 1024)
    assert not memory_allows(5000, 1200 * 1024 * 1024)
    assert not memory_allows(5100, 3452692285)


def test_ranking_is_finite_and_a_column_shuffle_can_change_it():
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(32, 6, generator=generator)
    base[:, 1] = base[:, 0]
    scores, degenerate, rayleigh = rank_from_activations(base)
    shuffled, _shuffled_degenerate, shuffled_rayleigh = rank_from_activations(column_shuffle(base))
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(shuffled).all()
    assert rayleigh == rayleigh
    assert shuffled_rayleigh == shuffled_rayleigh
    assert not degenerate
    assert not torch.allclose(scores, shuffled)
    assert quota_for(3072) == 153
