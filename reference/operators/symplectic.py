"""Discrete Euler-Lagrange residual for a free particle.

L_d(q0, q1, h) = ||q1 - q0||^2 / (2h).
Axiom 3 holds on a uniform step: D2 L_d(q_prev, q, h) + D1 L_d(q, q_next, h) = 0.
"""


def _d1(q0, q1, h):
    return [(q0[i] - q1[i]) / h for i in range(len(q0))]


def _d2(q0, q1, h):
    return [(q1[i] - q0[i]) / h for i in range(len(q0))]


def del_residual(q_prev, q_curr, q_next, h):
    left = _d2(q_prev, q_curr, h)
    right = _d1(q_curr, q_next, h)
    return sum(abs(left[i] + right[i]) for i in range(len(left)))
