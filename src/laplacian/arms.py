"""Named runs for the Spectral-LLM queue.

Layer sets in the source note are 1-based. `layers_0index` is what the loss uses.
The grid cell s=0.5 on layers {2,5} aliases arm2. Scaling M=8 aliases arm4.
"""

from __future__ import annotations

PULSE_STEPS = [30000, 60000, 90000, 120000, 150000, 180000, 210000]


def _base(**overrides):
    config = {
        "steps": 300_000,
        "m": 4,
        "k": 0,
        "spectral": False,
        "pulses": False,
        "norm_removal": False,
        "dead_units": True,
        "layers_1index": [2, 5],
        "exponent": 0.5,
        "microbatch": 8,
        "accum": 8,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup": 2000,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "clip": 1.0,
        "calibration_sequences": 512,
        "pulse_fraction": 0.05,
        "eps": 0.15,
        "retain_floor": 0.25,
        "pulse_mode": "sample",
        "seed": 0,
        "context": 256,
        "spectral_probe_steps": 0,
        "pulse_only": False,
        "masks": False,
        "teacher": None,
        "manifold_constant": None,
    }
    config.update(overrides)
    return config


def layers_0index(config):
    converted = [int(layer) - 1 for layer in config["layers_1index"]]
    if any(layer < 0 or layer > 4 for layer in converted):
        raise ValueError(f"layer set out of range for a 5-layer model: {config['layers_1index']}")
    return converted


def all_configs():
    configs = {
        "arm0": _base(run_id="arm0"),
        "arm1": _base(run_id="arm1", m=32, k=8),
        "arm2": _base(run_id="arm2", spectral=True, exponent=0.5, layers_1index=[2, 5]),
        "arm3": _base(run_id="arm3", m=8, k=16),
        "arm3_masks": _base(run_id="arm3_masks", m=8, k=16, masks=True, steps=0),
        "arm4": _base(
            run_id="arm4", m=8, k=8, spectral=True, pulses=True, norm_removal=True, exponent=0.5
        ),
        "smoke_arm0": _base(run_id="smoke_arm0", steps=100, spectral_probe_steps=10, dead_units=False),
        "smoke_pulse": _base(
            run_id="smoke_pulse",
            m=8,
            k=8,
            steps=0,
            pulse_only=True,
            pulses=True,
            calibration_sequences=512,
            dead_units=False,
        ),
        "distill_plain": _base(
            run_id="distill_plain", steps=30_000, m=4, k=0, teacher="arm1", microbatch=4, dead_units=True
        ),
        "distill_manifold": _base(
            run_id="distill_manifold",
            steps=30_000,
            m=4,
            k=0,
            teacher="arm1",
            microbatch=4,
            manifold_constant=0.05,
            dead_units=True,
        ),
    }
    for exponent in (0.25, 0.5, 0.75):
        for layers in ([1, 5], [2, 5], [1, 3, 5]):
            tag = str(exponent).replace(".", "p")
            layer_tag = "".join(str(layer) for layer in layers)
            run_id = f"grid_s{tag}_l{layer_tag}"
            if exponent == 0.5 and layers == [2, 5]:
                continue
            configs[run_id] = _base(
                run_id=run_id, spectral=True, exponent=exponent, layers_1index=list(layers)
            )
    for width in (2, 4, 16):
        configs[f"scale_m{width}"] = _base(
            run_id=f"scale_m{width}",
            m=width,
            k=8,
            spectral=True,
            pulses=True,
            norm_removal=True,
            exponent=0.5,
        )
    return configs


def get_config(name):
    configs = all_configs()
    if name not in configs:
        known = ", ".join(sorted(configs))
        raise KeyError(f"unknown arm {name}. known: {known}")
    return configs[name]


def alias_sources():
    return {
        "grid_s0p5_l25": "arm2",
        "scale_m8": "arm4",
    }
