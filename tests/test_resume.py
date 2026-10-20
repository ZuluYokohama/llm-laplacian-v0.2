import torch

from laplacian.train import restore_generator


def test_restore_generator_matches_a_cpu_byte_state(tmp_path):
    source = torch.Generator(device="cpu")
    source.manual_seed(7)
    path = tmp_path / "generator.pt"
    torch.save({"generator": source.get_state()}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)["generator"]
    expected = torch.Generator(device="cpu")
    expected.set_state(loaded.contiguous())
    target = torch.Generator(device="cpu")
    restore_generator(target, loaded)
    draw = torch.randint(0, 1_000_000, (4,), generator=target)
    reference = torch.randint(0, 1_000_000, (4,), generator=expected)
    assert torch.equal(draw, reference)
