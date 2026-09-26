"""Numerical checks of the explicit numbers stated in paper/paper.tex.

Checks (all cheap; no dense d^2 x d^2 matrix beyond d = 12):
  A. D = 31 value of D^(-1/3)/4 + 1/(3 sqrt 2), and monotonicity in D   (proof of Theorem 3)
  B. d = 64 finite thresholds: exact beta_N vs simplified bound           (Remark 12)
  C. chain beta_N <= middle expression <= simplified expression           (Theorem 8)
  D. Lemma 11: sum form == binomial-expectation form; Jensen bound        (Lemma 11)
  E. Proposition 14 constants and the "d >= 6" consequence                (Proposition 14 + following paragraph)
  F. Fourier protocols at d = 3..12: adaptive = 1/(4 sqrt D), PPT value, nonadaptive = 1/(2d)   (Thm 4, Remark 13)
  G. Simplified threshold / (2D)^(1/3) -> 1                                (Remark 12, last sentence)

These supplement the proofs; they do not prove anything for all d.
Run from code/:  python checks/verify_numbers.py results/verify_numbers.json
"""

from __future__ import annotations

import json
import math
import sys
import numpy as np


def _lncomb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _pmf(n: int) -> np.ndarray:
    """Bin(n, 1/2) probabilities, computed in log space (no overflow for large n)."""
    return np.array([math.exp(_lncomb(n, k) - n * math.log(2)) for k in range(n + 1)])


def E_m(m: int, D: int) -> float:
    n = m - 1
    k = np.arange(n + 1)
    return float(np.dot(_pmf(n), k / (D + k)))


def W_m(m: int, D: int) -> float:
    n = m - 1
    l = np.arange(n + 1)
    return float(np.dot(_pmf(n), np.sqrt((n - l) / (D + l))))


def W_m_sum_form(m: int, D: int) -> float:
    """The first expression of Lemma 11: 2^-(m-1) sum_l sqrt(c_{l+1} c_l (l+1)/(D+l))."""
    n = m - 1
    return sum(math.exp(0.5 * (_lncomb(n, l + 1) + _lncomb(n, l)) - n * math.log(2)) * math.sqrt((l + 1) / (D + l))
               for l in range(n))


def beta(N: int, D: int) -> float:
    return sum(E_m(m, D) + 0.5 * W_m(m, D) for m in range(2, N + 1))


def middle(N: int, D: int) -> float:
    return N * (N - 1) / (4 * D) + sum(math.sqrt(j) for j in range(1, N)) / (2 * math.sqrt(2 * D))


def simplified(N: int, D: int) -> float:
    return N * (N - 1) / (4 * D) + N**1.5 / (3 * math.sqrt(2 * D))


def first_N(f, D: int, start: int = 2) -> int:
    N = start
    while f(N, D) < 1 / 3:
        N += 1
    return N


def fourier_checks(d: int) -> dict:
    D = d - 1
    F = np.array([[np.exp(2j * np.pi * j * k / d) / np.sqrt(d) for k in range(d)] for j in range(d)])  # row j = f_j
    Y = np.zeros((d * d, d * d), dtype=complex)
    for i in range(1, d):
        Y[i * d + 0, 0 * d + i] += 1  # |i0><0i|
        Y[0 * d + i, i * d + 0] += 1  # |0i><i0|
    Delta = Y / (4 * D)
    # PPT value: half trace norm of the partial transpose on copy 2
    T = Delta.reshape(d, d, d, d).transpose(0, 3, 2, 1).reshape(d * d, d * d)
    ppt = 0.5 * np.abs(np.linalg.eigvalsh(T)).sum()
    # adaptive protocol
    adv_adaptive = 0.0
    hjk_ok = True
    adv_nonadaptive = 0.0
    for j in range(d):
        f = F[j]
        H = np.einsum("a,abcd,c->bd", f.conj(), Y.reshape(d, d, d, d), f)  # (<f|x I) Y (|f> x I)
        w, V = np.linalg.eigh(H)
        adv_adaptive += sum(w[w > 1e-12]) / (4 * D)
        for k in range(d):
            val = np.vdot(F[k], H @ F[k]).real
            hjk_ok &= abs(val - (2 / d) * ((1 if j == k else 0) - 1 / d)) < 1e-12
            adv_nonadaptive += max(val, 0.0) / (4 * D)
    return {
        "d": d,
        "ppt_half_trace_norm": ppt,
        "adaptive_advantage": adv_adaptive,
        "target_1_over_4sqrtD": 1 / (4 * math.sqrt(D)),
        "nonadaptive_fourier_advantage": adv_nonadaptive,
        "target_1_over_2d": 1 / (2 * d),
        "fk_Hj_fk_formula_ok": bool(hjk_ok),
        "pass": bool(abs(ppt - 1 / (4 * math.sqrt(D))) < 1e-12 and abs(adv_adaptive - 1 / (4 * math.sqrt(D))) < 1e-12
                     and abs(adv_nonadaptive - 1 / (2 * d)) < 1e-12 and hjk_ok),
    }


def main(out: str) -> None:
    res: dict = {}
    # A
    f = lambda D: D ** (-1 / 3) / 4 + 1 / (3 * math.sqrt(2))
    vals = [f(D) for D in range(31, 5001)]
    res["A_D31"] = {"value": f(31), "below_0.316": f(31) < 0.316, "decreasing_31_to_5000": all(a > b for a, b in zip(vals, vals[1:]))}
    # B
    D = 63
    res["B_d64"] = {
        "exact_beta_threshold": first_N(beta, D),
        "middle_threshold": first_N(middle, D),
        "simplified_threshold": first_N(simplified, D),
        "beta_values": {N: beta(N, D) for N in range(2, 8)},
        "simplified_values": {N: simplified(N, D) for N in range(2, 8)},
    }
    # C
    chain_ok = True
    for D in (2, 3, 10, 31, 63, 100, 1000, 10**5):
        for N in range(2, 120):
            b, mdl, s = beta(N, D), middle(N, D), simplified(N, D)
            chain_ok &= (b <= mdl + 1e-15) and (mdl <= s + 1e-15)
    res["C_chain_beta_le_middle_le_simplified"] = chain_ok
    # D
    lem11_ok = True
    jensen_ok = True
    for D in (2, 3, 10, 63, 1000):
        for m in range(2, 60):
            a, b = W_m(m, D), W_m_sum_form(m, D)
            lem11_ok &= abs(a - b) <= 1e-12 * max(1.0, abs(a))
            jensen_ok &= a <= math.sqrt((m - 1) / (2 * D)) + 1e-15
    res["D_lemma11_forms_agree"] = lem11_ok
    res["D_lemma11_jensen_bound"] = jensen_ok
    # E
    prop_ok = True
    sum_ok = True
    for D in (2, 3, 5, 6, 10, 31, 63, 100, 1000, 3000):
        Ev = {m: E_m(m, D) for m in range(2, D + 1)}
        Wv = {m: W_m(m, D) for m in range(2, D + 1)}
        for m in range(2, D + 1):
            prop_ok &= Wv[m] >= math.sqrt(m - 1) / (2 * math.sqrt(2 * D)) - 1e-15
            prop_ok &= Ev[m] + Wv[m] <= 2 * math.sqrt((m - 1) / (2 * D)) + 1e-15
        acc_W = 0.0
        acc_EW = 0.0
        for N in range(2, D + 1):
            acc_W += Wv[N]
            acc_EW += Ev[N] + Wv[N]
            sum_ok &= (N - 1) ** 1.5 / (6 * math.sqrt(2 * D)) <= 0.5 * acc_W + 1e-15
            sum_ok &= 0.5 * acc_EW <= 2 * N**1.5 / (3 * math.sqrt(2 * D)) + 1e-15
    res["E_prop14_stepwise"] = prop_ok
    res["E_prop14_sum_bounds"] = sum_ok
    consequence = []
    for D in range(5, 3001):
        Nstar = math.ceil(2 * D ** (1 / 3) + 1)
        consequence.append(Nstar <= D and (Nstar - 1) ** 1.5 / (6 * math.sqrt(2 * D)) >= 1 / 3 and Nstar <= 2 * D ** (1 / 3) + 2)
    res["E_consequence_d_ge_6"] = all(consequence)
    # F
    res["F_fourier"] = [fourier_checks(d) for d in range(3, 13)]
    # G
    res["G_simplified_threshold_ratio"] = {str(D): first_N(simplified, D) / (2 * D) ** (1 / 3) for D in (10**3, 10**6, 10**9, 10**12)}
    res["G_exact_threshold_ratio_info"] = {str(D): first_N(beta, D) / (2 * D) ** (1 / 3) for D in (10**3, 10**4, 10**5)}
    res["all_pass"] = bool(
        res["A_D31"]["below_0.316"] and res["A_D31"]["decreasing_31_to_5000"]
        and res["B_d64"]["exact_beta_threshold"] == 6 and res["B_d64"]["simplified_threshold"] == 5
        and chain_ok and lem11_ok and jensen_ok and prop_ok and sum_ok and res["E_consequence_d_ge_6"]
        and all(r["pass"] for r in res["F_fourier"])
    )
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("F_fourier",)}, indent=1, default=float))
    print("F_fourier pass:", [r["pass"] for r in res["F_fourier"]])
    print("ALL_PASS", res["all_pass"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "verify_numbers.json")
