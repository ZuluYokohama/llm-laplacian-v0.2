"""CPU-only analytic-spectrum cutoff sensitivity; no training/quality inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laplacian.operators_torch import fractional_matrix, fractional_policy
from laplacian.runtime import write_metrics

CUTOFFS = (1e-5, 1e-4, 1e-3)
EXPONENTS = (.25, .5, .75, 1.)


def scalar(values, exponent, cutoff):
    """Independent scalar polynomial expression, not the production helper."""
    answer = np.empty_like(values)
    for index, value in enumerate(values):
        if value < 0:
            answer[index] = (2-exponent)*cutoff**(exponent-1)*value
        elif value < cutoff:
            t = value/cutoff
            answer[index] = cutoff**exponent*((2-exponent)*t+(exponent-1)*t*t)
        else:
            answer[index] = value**exponent
    return answer


def slope(value, exponent, cutoff):
    if value < 0:
        return (2-exponent)*cutoff**(exponent-1)
    if value < cutoff:
        return cutoff**(exponent-1)*((2-exponent)+2*(exponent-1)*(value/cutoff))
    return exponent*value**(exponent-1)


def oracle(spectrum, basis, features, exponent, cutoff):
    values = scalar(spectrum, exponent, cutoff)
    powered = (basis*values) @ basis.T
    secants = np.empty((len(spectrum), len(spectrum)))
    for i, left in enumerate(spectrum):
        for j, right in enumerate(spectrum):
            secants[i, j] = (slope(left, exponent, cutoff) if left == right
                             else (values[i]-values[j])/(left-right))
    upstream = .5*features @ features.T
    derivative = basis @ (secants*(basis.T @ upstream @ basis)) @ basis.T
    energy = .5*np.sum(features*(powered @ features))
    return powered, derivative, energy


def dense_energy(matrix, features, exponent, cutoff):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    powered = (eigenvectors*scalar(eigenvalues, exponent, cutoff)) @ eigenvectors.T
    return .5*np.sum(features*(powered @ features))


def fixtures():
    spectra = {
        "zero": [0., 0., 0., 0.],
        "repeated_positive": [0., .5, .5, .5],
        "disconnected_nullity_two": [0., 0., 2., 2.],
        "small_gap": [0., 1e-8, 1e-6, 1.],
        "repeated_inside_default_cap": [0., 5e-5, 5e-5, 1.],
        "admitted_negative_roundoff": [-5e-8, 0., 1e-4, .5],
    }
    for cutoff in CUTOFFS:
        spectra[f"join_{cutoff:g}"] = [0., .5*cutoff, cutoff, 2*cutoff]
    for scale in (.01, 1., 100.):
        spectra[f"scale_{scale:g}"] = [scale*x for x in (0., 1e-5, 1e-4, 1e-3)]
    return {key: np.asarray(values, dtype=np.float64) for key, values in spectra.items()}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve(), ROOT / "src/laplacian/operators_torch.py",
             ROOT / "src/laplacian/runtime.py", ROOT / "paper/planning/FRACTIONAL_POLICY.md"]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for source in paths:
        destination = output / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != source_hashes[source.relative_to(ROOT).as_posix()]:
            raise RuntimeError("Source changed while creating immutable snapshot")
    report = {
        "complete": False, "run_id": output.name, "host_gpu": False, "device": "cpu",
        "scope": "Analytic symmetric 4x4 spectra, fixed features, operator and matrix-gradient checks; no model/training/quality inference",
        "policy": fractional_policy(), "cutoffs": CUTOFFS, "exponents": EXPONENTS,
        "precision": "float64", "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "source_sha256": source_hashes,
        "oracle_tolerances": {"matrix_atol": 2e-9, "matrix_rtol": 1e-9,
                              "gradient_atol": 2e-7, "gradient_rtol": 1e-8,
                              "directional_relative_scaled": 5e-5},
        "limitations": ["These spectra need not be realizable by the model's Gaussian normalized graph",
                        "Gradient is with respect to the matrix, not model parameters or held-out CE",
                        "Relative sensitivity is undefined at zero reference norm; absolute differences are retained",
                        "Large sensitivity inside the changed band is a method effect, not an implementation failure"],
        "cases": [],
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        generator = np.random.default_rng(20261021)
        basis, _ = np.linalg.qr(generator.normal(size=(4, 4)))
        features = generator.normal(size=(4, 3))
        direction = generator.normal(size=(4, 4))
        direction = (direction+direction.T)/2
        direction /= np.linalg.norm(direction)
        report["fixture_seed"] = 20261021
        report["basis"] = basis.tolist()
        report["features"] = features.tolist()
        report["finite_difference_direction"] = direction.tolist()
        for name, spectrum in fixtures().items():
            matrix = (basis*spectrum) @ basis.T
            for exponent in EXPONENTS:
                default_power, default_gradient, default_energy = oracle(spectrum, basis, features, exponent, 1e-4)
                for cutoff in CUTOFFS:
                    expected, gradient, energy = oracle(spectrum, basis, features, exponent, cutoff)
                    lap = torch.tensor(matrix, dtype=torch.float64, requires_grad=True)
                    points = torch.tensor(features, dtype=torch.float64)
                    actual = fractional_matrix(lap, exponent, cutoff=cutoff)
                    actual_energy = .5*(points*(actual @ points)).sum()
                    actual_gradient = torch.autograd.grad(actual_energy, lap)[0].numpy()
                    actual_matrix = actual.detach().numpy()
                    if not np.isfinite(actual_matrix).all() or not np.isfinite(actual_gradient).all():
                        raise RuntimeError(f"nonfinite operator/gradient: {name}/{exponent}/{cutoff}")
                    np.testing.assert_allclose(actual_matrix, expected, atol=2e-9, rtol=1e-9)
                    np.testing.assert_allclose(actual_gradient, gradient, atol=2e-7, rtol=1e-8)
                    h = cutoff*1e-5
                    finite_difference = (dense_energy(matrix+h*direction, features, exponent, cutoff)
                                         - dense_energy(matrix-h*direction, features, exponent, cutoff))/(2*h)
                    analytic_direction = float(np.sum(gradient*direction))
                    directional_error = abs(finite_difference-analytic_direction)/max(1., abs(analytic_direction))
                    if directional_error > 5e-5:
                        raise RuntimeError(f"finite difference mismatch: {name}/{exponent}/{cutoff}: {directional_error}")
                    gradient_delta = float(np.linalg.norm(gradient-default_gradient))
                    matrix_delta = float(np.linalg.norm(expected-default_power, ord=2))
                    row = {"fixture": name, "spectrum": spectrum.tolist(), "exponent": exponent, "cutoff": cutoff,
                           "finite": True, "energy": float(actual_energy.detach()), "matrix_gradient_frobenius": float(np.linalg.norm(actual_gradient)),
                           "energy_delta_vs_default": float(energy-default_energy),
                           "matrix_operator_delta_vs_default": matrix_delta,
                           "matrix_gradient_delta_vs_default": gradient_delta,
                           "matrix_gradient_relative_delta_vs_default": gradient_delta/float(np.linalg.norm(default_gradient)) if np.linalg.norm(default_gradient) > 0 else None,
                           "maximum_scalar_slope": (2-exponent)*cutoff**(exponent-1),
                           "matrix_max_abs_oracle_error": float(np.max(np.abs(actual_matrix-expected))),
                           "gradient_max_abs_oracle_error": float(np.max(np.abs(actual_gradient-gradient))),
                           "finite_difference_step": h, "finite_difference_directional_error_scaled": directional_error,
                           "oracle_passed": True}
                    if np.all(spectrum >= 0):
                        exact = (basis*spectrum**exponent) @ basis.T
                        original_energy = .5*np.sum(features*(exact @ features))
                        gap = float(original_energy-energy)
                        bound = .5*cutoff**exponent*float(np.square(features).sum())
                        if gap < -1e-10 or gap > bound+1e-10:
                            raise RuntimeError("Declared PSD energy-error bound failed")
                        row.update(unsmoothed_forward_energy=float(original_energy),
                                   unsmoothed_forward_energy_gap=gap, absolute_psd_energy_gap_bound=bound,
                                   unsmoothed_derivative_compared=False)
                    report["cases"].append(row)
        after = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report["source_sha256_after"] = after
        if after != source_hashes:
            raise RuntimeError("Source changed during CPU cutoff diagnostic")
        report["summary"] = {
            "case_count": len(report["cases"]),
            "maximum_matrix_oracle_error": max(c["matrix_max_abs_oracle_error"] for c in report["cases"]),
            "maximum_gradient_oracle_error": max(c["gradient_max_abs_oracle_error"] for c in report["cases"]),
            "maximum_directional_error_scaled": max(c["finite_difference_directional_error_scaled"] for c in report["cases"]),
            "zero_matrix_gradient_cutoff_dependence_expected": True,
        }
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        torch.set_num_threads(previous_threads)
        write_metrics(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
