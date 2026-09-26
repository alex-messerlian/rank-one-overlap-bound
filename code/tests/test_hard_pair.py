"""Tests of the hard pair and the proof's main identities. Each test compares two independent routes (construction vs closed form, or dense numerics vs
analytic bound); none calls the same formula twice.

Run from code/:  python -m pytest -q tests
"""

from __future__ import annotations

import numpy as np
import pytest

from src import rankone as r1


@pytest.mark.parametrize("d,N", [(3, 2), (3, 3), (4, 2), (4, 3)])
def test_nested_subspace_identity_matches_haar_expansion(d: int, N: int) -> None:
    for which in ("c", "m"):
        assert np.abs(r1.omega_expansion(d, N, which) - r1.omega_projector_form(d, N, which)).max() < 1e-13


def test_haar_expansion_matches_monte_carlo() -> None:
    mc_c, mc_m = r1.omega_monte_carlo(4, 2, 20000, seed=7)
    assert np.abs(mc_c - r1.omega_expansion(4, 2, "c")).max() < 0.01
    assert np.abs(mc_m - r1.omega_expansion(4, 2, "m")).max() < 0.01


def test_q_values_and_single_copy_average() -> None:
    rng = np.random.default_rng(1)
    d = 8
    P = np.zeros((d, d))
    P[0, 0] = 1
    for _ in range(10):
        rc, rm = r1.rho_pair(r1.haar_u(d, rng))
        assert abs(np.trace(P @ rc @ rc) - 0.5) < 1e-12 and abs(np.trace(P @ rm @ rm) - 0.25) < 1e-12
    assert np.allclose(r1.omega_expansion(d, 1, "c"), r1.sigma(d))
    assert np.allclose(r1.omega_expansion(d, 1, "m"), r1.sigma(d))


@pytest.mark.parametrize("d", [3, 5, 8])
def test_two_copy_difference_is_Y_over_4D(d: int) -> None:
    Delta = r1.omega_expansion(d, 2, "c") - r1.omega_expansion(d, 2, "m")
    assert np.abs(Delta - r1.Y_operator(d) / (4 * (d - 1))).max() < 1e-14


@pytest.mark.parametrize("d", [8, 32, 128])
def test_two_copy_adaptive_protocol_attains_ppt_bound(d: int) -> None:
    closed = 1 / (4 * np.sqrt(d - 1))
    assert abs(r1.tv_adaptive_two_copy(d) - closed) < 1e-12
    ppt = 0.5 * r1.trace_norm(r1.partial_transpose_last(r1.Y_operator(d) / (4 * (d - 1)), d, 2))
    assert abs(ppt - closed) < 1e-12


def test_coherence_blind_basis_sees_nothing() -> None:
    for d, N in [(4, 2), (4, 3)]:
        Delta = r1.omega_expansion(d, N, "c") - r1.omega_expansion(d, N, "m")
        assert r1.tv_product_basis(Delta, np.eye(d), N) < 1e-15


def test_pointwise_ratio_counterexample_closed_form() -> None:
    d = 8
    D = d - 1
    r = (D + 1) / (2 * D)
    a2 = 1 / (1 + r * D)
    a, b = np.sqrt(a2), np.sqrt(1 - a2)
    e0, w = np.eye(d)[0], np.eye(d)[1]
    x = np.kron(a * e0 + b * w, a * e0 - b * w)
    pc = x @ r1.omega_expansion(d, 2, "c") @ x
    ps = x @ np.kron(r1.sigma(d), r1.sigma(d)) @ x
    assert abs(pc / ps - 2 * D / (3 * D + 1)) < 1e-12


@pytest.mark.parametrize("d,m", [(3, 3), (4, 3), (4, 4), (8, 2)])
def test_coherence_step_trace_norm_equals_closed_form(d: int, m: int) -> None:
    B = r1.register_parts(d, m)["coh"]
    numeric = r1.trace_norm(r1.partial_transpose_last(B, d, m))
    assert abs(numeric - 2 * r1.W_trace_norm_closed(d, m)) < 1e-10
    assert r1.W_trace_norm_closed(d, m) <= np.sqrt((m - 1) / (2 * (d - 1))) + 1e-15


@pytest.mark.parametrize("d,m", [(3, 4), (4, 3), (5, 3), (8, 3)])
def test_purity_type_steps_within_proved_bound(d: int, m: int) -> None:
    D = d - 1
    Pi = np.eye(d)
    Pi[0, 0] = 0
    A = r1.register_parts(d, m)["uu"] - np.kron(r1.omega_expansion(d, m - 1, "c"), Pi / D)
    Ap = r1.register_parts_mixed(d, m)["uu"] - np.kron(r1.omega_expansion(d, m - 1, "m"), Pi / D)
    bound = 2 * r1.E_m_closed(d, m)
    assert r1.trace_norm(r1.partial_transpose_last(A, d, m)) <= bound + 1e-12
    assert r1.trace_norm(r1.partial_transpose_last(Ap, d, m)) <= bound + 1e-12
    assert r1.E_m_closed(d, m) <= (m - 1) / (2 * D) + 1e-15


def test_mixed_ensemble_one_sided_ratio_on_random_and_structured_product_vectors() -> None:
    """Floating check of the proved bound <x|omega_m|x> >= (1 - N(N-1)/(2D)) <x|sigma^N|x>."""
    rng = np.random.default_rng(3)
    d, N = 4, 3
    D = d - 1
    om = r1.omega_expansion(d, N, "m")
    sg = r1.kron_power(r1.sigma(d), N)
    worst = np.inf
    for trial in range(400):
        vecs = []
        for t in range(N):
            v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            if trial % 2:  # structured: shared complement direction, varying |0> weight
                v = np.concatenate([[rng.random()], np.ones(d - 1) * np.exp(2j * np.pi * t / N)])
            vecs.append(v / np.linalg.norm(v))
        x = vecs[0]
        for v in vecs[1:]:
            x = np.kron(x, v)
        worst = min(worst, np.real(x.conj() @ om @ x) / np.real(x.conj() @ sg @ x))
    assert worst >= 1 - N * (N - 1) / (2 * D) - 1e-12


@pytest.mark.parametrize("q", [5, 10, 20, 30])
def test_certified_copy_lower_bound(q: int) -> None:
    d = 2 ** q
    n = int(np.floor((d - 1) ** (1 / 3)))
    assert r1.ppt_bias_bound_simple(d, n) < 1 / 3
    if n <= 2000:
        assert r1.ppt_bias_bound(d, n) <= r1.ppt_bias_bound_simple(d, n) + 1e-12
