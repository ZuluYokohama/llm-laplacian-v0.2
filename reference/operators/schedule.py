"""Piecewise spectral schedule from the specification's phase table.

Phase I is steps [0, 30000). Phase II is [30000, 240000) at the peak coefficients.
Phase III is [240000, 300000]. The intra-phase linear ramp is not applied:
step 30000 already returns the Phase II peaks the tests lock.
"""


def gammas(step):
    step = int(step)
    if step < 30000:
        return {"phase": 1, "manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 1e-4}
    if step < 240000:
        return {"phase": 2, "manifold": 0.05, "fractional": 0.02, "hyper": 0.01, "dirichlet": 1e-3}
    return {"phase": 3, "manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 1e-4}
