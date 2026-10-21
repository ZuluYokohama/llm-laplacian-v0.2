"""Linear Phase II ramp and Phase III decay from the Spectral-LLM specification.

Phase I  [0, 30000): geometric weights 0, Dirichlet 1e-4.
Phase II [30000, 240000): linear ramp to the peak table.
Phase III [240000, 300000]: linear decay of the geometric weights to 0, Dirichlet 1e-4.
"""

PEAK = {"manifold": 0.05, "fractional": 0.02, "hyper": 0.01, "dirichlet": 1e-3}
PHASE2_START = 30_000
PHASE3_START = 240_000
PHASE3_END = 300_000


def gammas(step):
    step = int(step)
    if step < PHASE2_START:
        return {"phase": 1, "manifold": 0.0, "fractional": 0.0, "hyper": 0.0, "dirichlet": 1e-4}
    if step < PHASE3_START:
        alpha = (step - PHASE2_START) / (PHASE3_START - PHASE2_START)
        return {
            "phase": 2,
            "manifold": PEAK["manifold"] * alpha,
            "fractional": PEAK["fractional"] * alpha,
            "hyper": PEAK["hyper"] * alpha,
            "dirichlet": 1e-4 + (PEAK["dirichlet"] - 1e-4) * alpha,
        }
    if step >= PHASE3_END:
        beta = 0.0
    else:
        beta = (PHASE3_END - step) / (PHASE3_END - PHASE3_START)
    return {
        "phase": 3,
        "manifold": PEAK["manifold"] * beta,
        "fractional": PEAK["fractional"] * beta,
        "hyper": PEAK["hyper"] * beta,
        "dirichlet": 1e-4,
    }
