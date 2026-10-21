import torch
import json
import numpy as np
import pytest

from laplacian import train as training
from laplacian.arms import get_config
from laplacian.model import SpectralTransformer
from laplacian.reproducibility import seed_everything
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


def tiny_state(seed=3):
    seed_everything(seed)
    model = SpectralTransformer(vocab=13, m=1, d_model=16, n_layers=1, context=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    return model, optimizer, generator


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_checkpoint_preserves_training_flags_and_elapsed_time(tmp_path):
    model, optimizer, generator = tiny_state()
    model.strip_layer(0)
    config = {"seed": 3, "steps": 20}
    path = tmp_path / "last.pt"
    fired = [torch.arange(16, dtype=torch.float32)]
    training.save_checkpoint(path, model, optimizer, generator, 4, [30000], [0], fired, 18,
                             config=config, elapsed_seconds_total=12.5, last_loss=2.0, last_ce=1.5)
    loaded, opt, gen = tiny_state(900)
    metadata = {}
    step, pulses, stripped, sums, tokens = training.load_checkpoint(
        path, loaded, opt, gen, "cpu", config=config, resume_metadata=metadata)
    assert step == 4 and pulses == [30000] and stripped == [0] and tokens == 18
    assert_nested_equal(sums, fired)
    assert not loaded.blocks[0].norm1.weight.requires_grad
    assert metadata["elapsed_seconds_total"] == 12.5
    assert metadata["elapsed_seconds_complete"]
    assert metadata["resume_status"] == "state_complete"
    assert metadata["last_loss"] == 2.0


def test_resume_rejects_different_training_config_before_loading_weights(tmp_path):
    model, optimizer, generator = tiny_state()
    path = tmp_path / "last.pt"
    training.save_checkpoint(path, model, optimizer, generator, 4, [], [], config={"seed": 3, "steps": 20})
    loaded, opt, gen = tiny_state(900)
    before = loaded.tok.weight.detach().clone()
    with pytest.raises(ValueError, match="configuration differs"):
        training.load_checkpoint(path, loaded, opt, gen, "cpu", config={"seed": 3, "steps": 21})
    assert torch.equal(loaded.tok.weight, before)


def test_legacy_resume_warning_persists_in_descendant_checkpoints(tmp_path):
    model, optimizer, generator = tiny_state()
    old = tmp_path / "legacy.pt"
    torch.save({"step": 4, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "generator": generator.get_state()}, old)
    metadata = {}
    config = {"seed": 3, "steps": 20}
    with pytest.warns(RuntimeWarning, match="Non-exact"):
        training.load_checkpoint(old, model, optimizer, generator, "cpu", config=config, resume_metadata=metadata)
    assert metadata["resume_status"] == "non_exact"
    assert not metadata["elapsed_seconds_complete"]
    new = tmp_path / "descendant.pt"
    training.save_checkpoint(new, model, optimizer, generator, 5, [], [], config=config,
                             resume_metadata=metadata, elapsed_seconds_total=8.0)
    with pytest.warns(RuntimeWarning, match="Non-exact"):
        training.load_checkpoint(new, model, optimizer, generator, "cpu", config=config, resume_metadata=metadata)
    assert metadata["resume_status"] == "non_exact"
    assert not metadata["elapsed_seconds_complete"]
    assert metadata["elapsed_seconds_total"] == 8.0


def test_nonfinite_checkpoint_preserves_last_valid_resume_artifact(tmp_path):
    model, optimizer, generator = tiny_state()
    path = tmp_path / "last.pt"
    training.save_checkpoint(path, model, optimizer, generator, 0, [], [], config={"seed": 3})
    previous = path.read_bytes()
    optimizer.state[model.tok.weight]["exp_avg"] = torch.full_like(model.tok.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="checkpoint was not written"):
        training.save_checkpoint(path, model, optimizer, generator, 1, [], [], config={"seed": 3})
    assert path.read_bytes() == previous
    assert not path.with_suffix(".pt.tmp").exists()


@pytest.fixture
def cpu_training(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    monkeypatch.setattr(training, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "DATA", tmp_path)
    monkeypatch.setattr(training, "apply_env", lambda: None)
    monkeypatch.setattr(training, "pick_5070", lambda: (torch.device("cpu"), "CPU test"))
    monkeypatch.setattr(training, "read_meta", lambda: {"vocab": 13})
    monkeypatch.setattr(training, "memory_mib", lambda: 0.0)
    monkeypatch.setattr(training, "enforce_guard", lambda: None)
    monkeypatch.setattr(training, "log_line", lambda line: None)
    monkeypatch.setattr(training, "build_model", lambda config, vocab, device:
                        SpectralTransformer(vocab=vocab, m=1, d_model=16, n_layers=1, context=config["context"]))
    np.tile(np.arange(13, dtype=np.uint16), 40).tofile(tmp_path / "tinystories_train.u16")
    config = get_config("arm0")
    config.update(steps=2002, microbatch=1, accum=1, context=4, seed=3)
    yield config
    torch.set_num_threads(previous_threads)
    torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def test_actual_training_resume_matches_across_dead_resampling_boundary(cpu_training, tmp_path, monkeypatch):
    # Force resampling at the real 1000-step cadence, so the test cannot pass
    # merely because no neuron happened to be dead at the second boundary.
    original_resample = training.resample_dead
    resampled = []

    def resample(model, rates):
        changed = original_resample(model, rates, threshold=2.0)
        assert changed > 0
        resampled.append(changed)
        return changed

    monkeypatch.setattr(training, "resample_dead", resample)
    training.train({**cpu_training, "run_id": "whole"})
    whole = torch.load(tmp_path / "runs/whole/last.pt", weights_only=False)
    original_save = training.save_checkpoint

    class Interruption(Exception):
        pass

    def interrupt_after_checkpoint(path, model, optimizer, generator, step, *args, **kwargs):
        original_save(path, model, optimizer, generator, step, *args, **kwargs)
        if step == 1000:
            raise Interruption()

    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_checkpoint)
    with pytest.raises(Interruption):
        training.train({**cpu_training, "run_id": "split"})
    assert not (tmp_path / "runs/split/weights_final.pt").exists()
    partial = torch.load(tmp_path / "runs/split/last.pt", weights_only=False)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    seed_everything(999)  # Simulate arbitrary RNG use in a fresh process.
    training.train({**cpu_training, "run_id": "split"})
    resumed = torch.load(tmp_path / "runs/split/last.pt", weights_only=False)
    for key in ("step", "model", "optimizer", "generator", "rng_state", "fired_sums", "fired_tokens"):
        assert_nested_equal(whole[key], resumed[key])
    assert len(resampled) == 4
    metrics = json.loads((tmp_path / "runs/split/metrics.json").read_text())
    assert metrics["resumed_from_step"] == 1000
    assert metrics["segment_steps"] == 1002
    assert metrics["elapsed_seconds_total"] == pytest.approx(partial["reproducibility"]["elapsed_seconds_total"] + metrics["segment_seconds"])
    assert metrics["elapsed_seconds_complete"]


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_training_failure_does_not_write_a_final_checkpoint(cpu_training, tmp_path, monkeypatch, failure):
    original = training.forward_loss

    def fail(model, inputs, targets, config, step):
        loss, ce, outputs, penalties = original(model, inputs, targets, config, step)
        if failure == "loss":
            loss = loss * float("nan")
        else:
            loss.register_hook(lambda gradient: gradient * float("nan"))
        return loss, ce, outputs, penalties

    monkeypatch.setattr(training, "forward_loss", fail)
    config = {**cpu_training, "run_id": "failure", "steps": 2}
    with pytest.raises(FloatingPointError):
        training.train(config)
    folder = tmp_path / "runs/failure"
    assert not (folder / "weights_final.pt").exists()
    assert not (folder / "last.pt").exists()
    assert not (folder / "metrics.json").exists()
