"""Synthetic CPU contracts only; no real dataset, GPU or held-out evaluation."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from laplacian.model import SpectralTransformer
from laplacian.reproducibility import config_hash, rng_state, seed_everything

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study_pair_test", ROOT / "scripts/benchmark_study_pair.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
PRODUCTION_WORK = pair.validate_work


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    study, training = pair.study, pair.training
    old_threads, old_tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    monkeypatch.setattr(study, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(training, "build_model", lambda cfg, vocab, device:
                        SpectralTransformer(vocab=vocab, m=1, d_model=16, n_layers=1, context=cfg["context"]).to(device))
    monkeypatch.setattr(training, "read_meta", lambda: pytest.fail("historical data used"))
    monkeypatch.setattr(training, "TokenStore", lambda *args: pytest.fail("historical data used"))
    monkeypatch.setattr(pair, "validate_work", lambda cfg: None)  # Explicit tiny-vocab fixture substitute.
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.tile(np.arange(13, dtype="<u2"), 40).tofile(bundle / "train.u16")
    (bundle / "train.stories.jsonl").write_text('{"story_id":"synthetic","offset":0,"length":520}\n')
    manifest = {"tokenizer": {"vocab_size": 13}, "files": {
        name: {key: value for key, value in study.file_record(bundle / name).items() if key != "path"}
        for name in ("train.u16", "train.stories.jsonl")}}
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    def validate(path):
        assert Path(path).resolve() == manifest_path.resolve()
        for name, expected in manifest["files"].items():
            actual = pair.file_record(bundle / name)
            if any(actual[key] != value for key, value in expected.items()):
                raise ValueError("Synthetic token input changed")
        return copy.deepcopy(manifest)

    monkeypatch.setattr(study, "validate_bundle", validate)
    config = study.make_pretrain_config(manifest_path, "synthetic-parent", 9002, steps=1)
    config.update(microbatch=2, accum=2, context=4, layers_1index=[1])
    config_path = study.prepare_pretrain(config)
    seed_everything(9002)
    model = training.build_model(config, 13, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000123, betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
    inputs = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    pair.update(model, optimizer, [(inputs, inputs + 1)] * 2, config, torch.device("cpu"))
    generator = torch.Generator().manual_seed(83)
    checkpoint = config_path.parent / "last.pt"
    training.save_checkpoint(checkpoint, model, optimizer, generator, 1, [], [],
                             fired_sums=[torch.arange(16, dtype=torch.float32)], fired_tokens=8, config=config)
    study.write_metrics(config_path.parent / "metrics.json", {"complete": True, "config": config, "steps": 1,
                        "study": {"final_checkpoint": pair.file_record(checkpoint), "timing_states": []}})
    monkeypatch.setattr(pair, "require_gpu", lambda run_id: torch.device("cpu"))
    yield {"checkpoint": checkpoint, "config": config, "bundle": bundle, "path": tmp_path}
    torch.set_num_threads(old_threads)
    torch.backends.cuda.matmul.allow_tf32 = old_tf32


def prepare(fixture, *, name="pair", order="AB", phase=0, warmup=1, steps=2):
    output = fixture["path"] / name
    plan = pair.prepare(fixture["checkpoint"], output, pair_id=name, penalty_step=phase,
                        batch_seed=123, order=order, warmup=warmup, steps=steps)
    return output / "plan.json", plan


def run_worker(path, implementation):
    return pair.worker(path, implementation, path.parent / implementation,
                       f"pair-{path.parent.name}-{implementation}")


def test_preparation_is_cpu_only_and_exact_stream_oracle(fixture, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried by prepare"))
    original = fixture["checkpoint"].read_bytes()
    path, plan = prepare(fixture)
    batches = pair.verified_json(plan["batch_manifest"]["path"])
    oracle = torch.Generator().manual_seed(123)
    tokens = np.tile(np.arange(13, dtype=np.int64), 40)
    assert len(batches["batches"]) == 6
    for row in batches["batches"]:
        starts = torch.randint(0, 515, (2,), generator=oracle).tolist()
        assert row["starts"] == starts
        packed = np.stack([tokens[start:start + 5] for start in starts])
        assert row["inputs_sha256"] == hashlib.sha256(packed[:, :-1].copy().tobytes()).hexdigest()
        assert row["targets_sha256"] == hashlib.sha256(packed[:, 1:].copy().tobytes()).hexdigest()
    assert batches["sampler_sha256_after"] == pair.tensor_hash(oracle.get_state())
    assert fixture["checkpoint"].read_bytes() == original
    assert not list(fixture["checkpoint"].parent.glob("samples-*.jsonl"))
    assert plan["kind"] != "confirmatory_timing_v1" and plan["confirmatory_claim_permitted"] is False
    raw = path.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(fixture)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("phase", [0, 135000])
def test_paired_workers_restore_full_state_hold_lr_and_apply_explicit_override(fixture, phase, monkeypatch):
    path, plan = prepare(fixture, phase=phase)
    config_before = copy.deepcopy(fixture["config"])
    seen = []
    real_update = pair.update

    def observe(model, optimizer, batches, cfg, device):
        seen.append((cfg["study"]["loss_implementation"], cfg["study"]["penalty_step"],
                     optimizer.param_groups[0]["lr"], float(next(iter(optimizer.state.values()))["step"])))
        return real_update(model, optimizer, batches, cfg, device)

    monkeypatch.setattr(pair, "update", observe)
    first = run_worker(path, "baseline")
    random.random()
    np.random.random()
    torch.rand(3)  # The second worker must recover checkpoint globals, not this caller's state.
    second = run_worker(path, "optimized")
    means = pair.validate_pair([first, second], plan)
    assert means["baseline"] > 0 and means["optimized"] > 0
    assert first["restored_state_sha256"] == second["restored_state_sha256"]
    assert first["prepared_state_sha256"] == second["prepared_state_sha256"]
    assert first["local_sampler_sha256_before_override"] != first["local_sampler_sha256_after_preparation"]
    assert [row[3] for row in seen] == [1., 2., 3., 1., 2., 3.]
    assert all(row[2] == .0000123 and row[1] == phase for row in seen)
    assert fixture["config"] == config_before
    assert first["benchmark_config"]["study"]["stage"] == "ce_pretrain"  # Clearly a separate benchmark overlay.
    assert first["benchmark_config"]["spectral"] and not first["benchmark_config"]["dead_units"]
    assert first["finite_checks_passed"] and first["numerical_acceptance"] == "not_evaluated"
    assert first["eligibility"] == "not_evaluated" and first["confirmatory_claim_permitted"] is False
    assert not (path.parent / "baseline/error.json").exists()
    with pytest.raises(FileExistsError):
        run_worker(path, "baseline")


def test_state_hash_sensitive_to_every_rng_optimizer_buffer_and_flag(fixture):
    payload = torch.load(fixture["checkpoint"], weights_only=False)
    state = {key: payload.get(key) for key in pair.STATE_KEYS}
    expected = pair.state_hash(state)
    assert pair.state_hash(copy.deepcopy(state)) == expected
    mutations = [
        lambda x: x["model"].__setitem__(next(iter(x["model"])), x["model"][next(iter(x["model"]))] + 1),
        lambda x: x["optimizer"]["param_groups"][0].__setitem__("lr", 3e-5),
        lambda x: x["generator"].__setitem__(0, x["generator"][0] ^ 1),
        lambda x: x["rng_state"]["torch_cpu"].__setitem__(0, x["rng_state"]["torch_cpu"][0] ^ 1),
        lambda x: x["rng_state"]["numpy"][1].__setitem__(0, 42),
        lambda x: x["rng_state"].__setitem__("python", (3, (1, 2), None)),
        lambda x: x["requires_grad"].__setitem__(next(iter(x["requires_grad"])), False),
        lambda x: x["module_training"].__setitem__("", False),
        lambda x: x.__setitem__("fired_tokens", 9),
        lambda x: x["fired_sums"][0].__setitem__(0, 1),
    ]
    for mutate in mutations:
        candidate = copy.deepcopy(state)
        mutate(candidate)
        assert pair.state_hash(candidate) != expected
    assert pair.state_hash(torch.tensor(2., dtype=torch.bfloat16)) != pair.state_hash(torch.tensor(2.))
    assert pair.state_hash([1, 2]) != pair.state_hash((1, 2))


@pytest.mark.parametrize("target", ["checkpoint", "train", "manifest", "config", "baseline", "batch", "source"])
def test_worker_input_or_source_change_fails_immutably(fixture, target, monkeypatch):
    path, plan = prepare(fixture)
    if target == "source":
        monkeypatch.setattr(pair, "sources", lambda: [])
    else:
        targets = {"checkpoint": fixture["checkpoint"], "train": fixture["bundle"] / "train.u16",
                   "manifest": fixture["bundle"] / "manifest.json", "config": fixture["checkpoint"].parent / "study_config.json",
                   "batch": Path(plan["batch_manifest"]["path"])}
        if target == "baseline":
            # Never mutate the shared historical source. Replace only the synthetic plan reference.
            wrong = fixture["path"] / "wrong.py"
            wrong.write_text("# changed baseline\n")
            plan["inputs"][-1] = {**plan["inputs"][-1], "path": str(wrong)}
            path.unlink()
            path.with_name("plan.json.sha256").unlink()
            pair.immutable_json(path, plan)
        else:
            with targets[target].open("ab") as handle:
                handle.write(b"\n")
    with pytest.raises((ValueError, RuntimeError), match="changed|stale"):
        run_worker(path, "baseline")
    error = pair.verified_json(path.parent / "baseline/error.json")
    assert not error["complete"] and not (path.parent / "baseline/metrics.json").exists()
    assert (path.parent / "baseline/attempt.json").exists()


def test_mid_update_mutation_revokes_completion(fixture, monkeypatch):
    path, _ = prepare(fixture)
    actual = pair.update
    def mutate(*args):
        result = actual(*args)
        with fixture["checkpoint"].open("ab") as handle:
            handle.write(b"\n")
        return result
    monkeypatch.setattr(pair, "update", mutate)
    with pytest.raises(ValueError, match="Bound input changed"):
        run_worker(path, "optimized")
    error = pair.verified_json(path.parent / "optimized/error.json")
    assert len(error["updates"]) == 3 and not error["complete"]
    assert error["inputs_before"] != error["inputs_after"]


@pytest.mark.parametrize("stage", ["prepare", "worker"])
def test_checkpoint_change_during_final_bundle_validation_cannot_complete(fixture, monkeypatch, stage):
    path = None
    if stage == "worker":
        path, _ = prepare(fixture)
    actual = pair.study.validate_config
    calls = []
    def change_after_validation(config):
        result = actual(config)
        calls.append(1)
        if len(calls) == 2:
            with fixture["checkpoint"].open("ab") as handle:
                handle.write(b"\n")
        return result
    monkeypatch.setattr(pair.study, "validate_config", change_after_validation)
    with pytest.raises(ValueError, match="Bound input changed"):
        if stage == "worker":
            run_worker(path, "optimized")
        else:
            prepare(fixture)
    error_path = fixture["path"] / ("pair/optimized/error.json" if stage == "worker" else "pair/prepare_error.json")
    error = pair.verified_json(error_path)
    assert len(calls) == 2 and error["inputs_before"] != error["inputs_after"]
    assert not error["complete"]
    assert not error_path.with_name("metrics.json" if stage == "worker" else "plan.json").exists()


def test_nonfinite_before_optimizer_mutation(fixture, monkeypatch):
    cfg = fixture["config"]
    model = pair.training.build_model(cfg, 13, torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters())
    before = pair.state_hash(model.state_dict())
    monkeypatch.setattr(pair.training, "forward_loss", lambda *args:
                        (torch.tensor(float("nan"), requires_grad=True), torch.tensor(1.), {}, None))
    with pytest.raises(FloatingPointError, match="before optimizer mutation"):
        pair.update(model, optimizer, [(torch.zeros((2, 4), dtype=torch.long),) * 2] * 2, cfg, torch.device("cpu"))
    assert not optimizer.state and pair.state_hash(model.state_dict()) == before


@pytest.mark.parametrize("field", ["generator", "module_training", "optimizer"])
def test_restore_detects_state_load_loss(fixture, monkeypatch, field):
    checkpoint = fixture["checkpoint"]
    actual = pair.capture_state
    def altered(*args):
        result = actual(*args)
        result[field] = {}
        return result
    monkeypatch.setattr(pair, "capture_state", altered)
    with pytest.raises(ValueError, match="Loaded full training state differs"):
        pair.restore(checkpoint, fixture["config"], torch.device("cpu"))


@pytest.mark.parametrize("change,message", [
    ("missing", "canonical budget"), ("wrong_run", "canonical budget"), ("wrong_ledger", "canonical budget"),
    ("device1", "canonical budget"), ("blank", "canonical budget"), ("wrong_gpu", "RTX 5070"), ("bf16", "BF16")])
def test_cuda_admission_cpu_fakes(monkeypatch, change, message):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "run")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(pair.LEDGER))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if change == "missing":
        monkeypatch.delenv("STUDY_BUDGET_RUN_ID")
    elif change == "wrong_run":
        monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "different")
    elif change == "wrong_ledger":
        monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(ROOT / "wrong.jsonl"))
    elif change == "device1":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    elif change == "blank":
        monkeypatch.setenv("STUDY_BUDGET_RUN_ID", " ")
    early = change not in ("wrong_gpu", "bf16")
    monkeypatch.setattr(pair.budget, "read_ledger", lambda path: [])
    monkeypatch.setattr(pair.budget, "accounting", lambda rows:
                        {"run": {"settled": False, "launch_authorized": True}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Early CUDA query") if early else True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "RTX 4090" if change == "wrong_gpu" else "RTX 5070")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: change != "bf16")
    with pytest.raises(RuntimeError, match=message):
        pair.require_gpu(" " if change == "blank" else "run")


@pytest.mark.parametrize("state", [None, {"settled": True, "launch_authorized": True},
                                  {"settled": False, "launch_authorized": False}])
def test_env_strings_cannot_replace_active_reservation(monkeypatch, state):
    monkeypatch.setenv("STUDY_BUDGET_RUN_ID", "run")
    monkeypatch.setenv("STUDY_BUDGET_LEDGER", str(pair.LEDGER))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(pair.budget, "read_ledger", lambda path: [])
    monkeypatch.setattr(pair.budget, "accounting", lambda rows: {} if state is None else {"run": state})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA before reservation admission"))
    with pytest.raises(RuntimeError, match="actually launched open canonical reservation"):
        pair.require_gpu("run")


def test_checkpoint_final_role_step_is_checked(fixture):
    payload = torch.load(fixture["checkpoint"], weights_only=False)
    payload["step"] = 0
    torch.save(payload, fixture["checkpoint"])
    with pytest.raises(ValueError, match="completed final/snapshot role"):
        pair.restore(fixture["checkpoint"], fixture["config"], torch.device("cpu"))


def test_source_disappearing_still_writes_error(fixture, monkeypatch):
    path, _ = prepare(fixture)
    def gone():
        raise FileNotFoundError("Synthetic source deletion")
    monkeypatch.setattr(pair, "sources", gone)
    with pytest.raises(FileNotFoundError, match="Synthetic source deletion"):
        run_worker(path, "baseline")
    assert not pair.verified_json(path.parent / "baseline/error.json")["complete"]


def test_mid_read_configuration_change_retains_original_attempt_identity(fixture, monkeypatch):
    config_path = fixture["checkpoint"].parent / "study_config.json"
    before = pair.file_record(config_path)
    original = pair.study.read_config
    def changed(path):
        result = original(path)
        with Path(path).open("ab") as handle:
            handle.write(b"\n")
        return result
    monkeypatch.setattr(pair.study, "read_config", changed)
    with pytest.raises(ValueError, match="Bound input changed"):
        prepare(fixture)
    error = pair.verified_json(fixture["path"] / "pair/prepare_error.json")
    assert before in error["attempted_inputs_before"]
    assert before not in error["attempted_inputs_after"]
    assert not error["complete"]


@pytest.mark.parametrize("kwargs", [{"warmup": 0}, {"warmup": True}, {"steps": 13}, {"steps": 1.5},
                                   {"penalty_step": True}, {"batch_seed": True}, {"pair_id": "../escape"}])
def test_invalid_options_never_create_attempt(fixture, kwargs):
    args = dict(pair_id="invalid", penalty_step=0, batch_seed=1, order="AB", warmup=1, steps=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        pair.prepare(fixture["checkpoint"], fixture["path"] / "invalid", **args)
    assert not (fixture["path"] / "invalid").exists()


def test_production_configuration_rejects_fixture_dimensions(fixture):
    with pytest.raises(ValueError, match="production M4"):
        PRODUCTION_WORK(fixture["config"])


def fake_receipt(plan, implementation):
    value = pair.receipt_base()
    value.update(complete=True, implementation=implementation, finite_checks_passed=True, freshness_checks_passed=True,
                 updates=[{"index": index, "warmup": index < plan["warmup"], "seconds": 1., "loss": 1., "ce": 1.}
                          for index in range(plan["warmup"] + plan["steps"])],
                 batch_manifest=plan["batch_manifest"], source_checkpoint=plan["checkpoint"], execution_id=implementation,
                 restored_state_sha256="a" * 64, prepared_state_sha256="b" * 64, environment={"synthetic": True},
                 cudnn_allow_tf32=True, optimizer_parameter_groups=[{"lr": .0000123}], source_step=1,
                 spectral_weights={key: float(pair.gammas(plan["penalty_step"])[key])
                                   for key in ("manifold", "fractional", "hyper", "dirichlet")},
                 timer_boundary=pair.BOUNDARY,
                 local_sampler_sha256_before_override="c" * 64, local_sampler_sha256_after_preparation="d" * 64)
    return value


@pytest.mark.parametrize("mutation,message", [
    (lambda x: x[1].__setitem__("prepared_state_sha256", "f" * 64), "prepared_state"),
    (lambda x: x[1].__setitem__("execution_id", "baseline"), "not distinct"),
    (lambda x: x[1]["updates"].pop(), "population differs"),
    (lambda x: x[0]["updates"][0].__setitem__("index", False), "record is invalid"),
    (lambda x: x[0]["updates"][1].__setitem__("seconds", float("nan")), "record is invalid"),
    (lambda x: x[0].__setitem__("numerical_acceptance", "passed"), "exploratory checks"),
])
def test_pair_mismatch_or_false_acceptance_fails(fixture, mutation, message):
    _, plan = prepare(fixture)
    receipts = [fake_receipt(plan, name) for name in ("baseline", "optimized")]
    mutation(receipts)
    with pytest.raises(ValueError, match=message):
        pair.validate_pair(receipts, plan)


def test_matching_phase1_weights_fail_when_plan_uses_phase_135000(fixture):
    _, plan = prepare(fixture, name="weights", phase=135000)
    receipts = [fake_receipt(plan, name) for name in ("baseline", "optimized")]
    phase1 = {key: float(pair.gammas(0)[key]) for key in ("manifold", "fractional", "hyper", "dirichlet")}
    assert receipts[0]["spectral_weights"] == receipts[1]["spectral_weights"] != phase1
    for item in receipts:
        item["spectral_weights"] = phase1
    with pytest.raises(ValueError, match="penalty schedule"):
        pair.validate_pair(receipts, plan)


@pytest.fixture
def dispatch_fixture(fixture, monkeypatch):
    ledger = fixture["path"] / "budget/budget.jsonl"
    decisions = fixture["path"] / "decisions.json"
    decisions.write_text(json.dumps({"recommended_core_gpu_hours": {"approved": True, "hard_cap": .02, "ledger": str(ledger)}}))
    monkeypatch.setattr(pair, "LEDGER", ledger)
    monkeypatch.setattr(pair, "DECISIONS", decisions)
    path, plan = prepare(fixture, order="BA")
    calls = []
    def command(plan_path, implementation, output, run_id):
        # Real fresh Python subprocesses, CPU only, under real temporary-grant budget containment.
        value = fake_receipt(plan, implementation)
        value.update(run_id=run_id, plan=pair.file_record(plan_path))
        code = ("import os,json,pathlib,hashlib; os.environ['CUDA_VISIBLE_DEVICES']=''; "
                "p=pathlib.Path(" + repr(str(output)) + "); p.mkdir(); "
                "v=json.loads(" + repr(json.dumps(value)) + "); v['pid']=os.getpid(); "
                "r=(json.dumps(v,sort_keys=True)+'\\n').encode(); (p/'metrics.json').write_bytes(r); "
                "(p/'metrics.json.sha256').write_text(hashlib.sha256(r).hexdigest()+'\\n')")
        calls.append(implementation)
        return [sys.executable, "-c", code]
    monkeypatch.setattr(pair, "worker_command", command)
    return path, plan, ledger, calls


def test_dispatch_real_cpu_fresh_processes_and_canonical_budget_proofs(dispatch_fixture):
    path, plan, ledger, calls = dispatch_fixture
    result = pair.run_pair(path, 8.)
    assert result["complete"] and result["confirmatory_claim_permitted"] is False
    assert calls == ["optimized", "baseline"]
    worker_rows = [pair.verified_json(row["metrics"]["path"]) for row in result["workers"]]
    assert worker_rows[0]["pid"] != worker_rows[1]["pid"]
    assert result["charged_seconds"] > 0
    records = pair.budget.read_ledger(ledger)
    assert [row["event"] for row in records] == ["reserved", "launch_authorized", "finished"] * 2
    for item in result["workers"]:
        proof = item["budget"]
        prefix = [json.loads(line) for line in Path(proof["ledger_prefix"]["path"]).read_text().splitlines()]
        assert prefix == records[:proof["settlement"]["sequence"] + 1]
    with pytest.raises(FileExistsError):
        pair.run_pair(path, 8.)
    assert len(pair.budget.read_ledger(ledger)) == 6


def test_dispatch_failed_first_worker_never_starts_second_or_promotes(dispatch_fixture, monkeypatch):
    path, _, ledger, calls = dispatch_fixture
    monkeypatch.setattr(pair, "worker_command", lambda *args: [sys.executable, "-c", "raise SystemExit(7)"])
    with pytest.raises(RuntimeError, match="cannot be promoted or retried"):
        pair.run_pair(path, 8.)
    assert len(pair.budget.read_ledger(ledger)) == 3
    error = pair.verified_json(path.parent / "dispatch/error.json")
    assert not error["complete"] and not (path.parent / "dispatch/metrics.json").exists()
    assert pair.budget.read_ledger(ledger)[-1]["charge_seconds"] >= 8.


def test_dispatch_cannot_cover_pair_or_reuse_attempt(dispatch_fixture):
    path, _, ledger, calls = dispatch_fixture
    pair.DECISIONS.write_text(json.dumps({"recommended_core_gpu_hours": {"approved": True, "hard_cap": 10/3600, "ledger": str(ledger)}}))
    with pytest.raises(ValueError, match="both worker reservations"):
        pair.run_pair(path, 8.)
    assert calls == [] and not ledger.exists()
    assert not pair.verified_json(path.parent / "dispatch/error.json")["complete"]


def test_budget_success_without_settlement_is_not_evidence(dispatch_fixture, monkeypatch):
    path, _, _, calls = dispatch_fixture
    monkeypatch.setattr(pair.budget, "run_study", lambda args: 0)
    with pytest.raises(ValueError, match="completed canonical budget settlement"):
        pair.run_pair(path, 8.)
    assert calls == ["optimized"]


def test_worker_receipt_replaced_after_parse_cannot_bind_new_bytes(dispatch_fixture, monkeypatch):
    path, _, ledger, calls = dispatch_fixture
    original = pair.verified_json
    def replaced(receipt_path):
        result = original(receipt_path)
        receipt_path = Path(receipt_path)
        if receipt_path.name == "metrics.json" and receipt_path.parent.name == "optimized":
            raw = receipt_path.read_bytes() + b"\n"
            receipt_path.write_bytes(raw)
            receipt_path.with_name("metrics.json.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
        return result
    monkeypatch.setattr(pair, "verified_json", replaced)
    with pytest.raises(ValueError, match="Bound input changed"):
        pair.run_pair(path, 8.)
    assert calls == ["optimized"] and len(pair.budget.read_ledger(ledger)) == 3
    assert not (path.parent / "dispatch/metrics.json").exists()
