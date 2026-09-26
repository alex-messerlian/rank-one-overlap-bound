"""Numerical evidence (not a proof): an explicit adaptive single-copy protocol for the hard pair, at larger d.

Learn-then-test with N = K + T copies, K = T = N/2:
  learn: measure K copies in the Fourier basis (every vector has |<0|f>|^2 = 1/d); an outcome f_j tilts the
         posterior of u towards the complement part of f_j, so aggregate w = sum_t Pi f_{j_t} and normalise;
  test:  measure T copies in a basis containing (|0> +- w)/sqrt(2): under rho_c,
         P(+) - P(-) = Re<w|u>; under rho_m the two probabilities are equal. Decide c iff n_+ > n_- (ties: coin).
The reported advantage P_c(decide c) - P_m(decide c) lower-bounds the optimal total-variation distance.
If the learned sum w is exactly zero (all Fourier outcome counts equal), w falls back to |1>; the run counts how often
this happens (never, with this seed).

Run from code/:  python checks/check_learn_then_test.py
Writes results/learn_then_test.json.
"""

from __future__ import annotations

import json
import time

import numpy as np

SEED = 20260926
TRIALS = 4000
STATS = {"fallback_used": 0, "min_spread_over_K": float("inf")}


def one_trial(d: int, K: int, T: int, hyp: str, rng: np.random.Generator) -> bool:
    D = d - 1
    g = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    u = np.zeros(d, dtype=complex)
    u[1:] = g / np.linalg.norm(g)
    fu = np.fft.fft(u) / np.sqrt(d)          # <f_j|u> for f_j[k] = exp(2 pi i j k / d)/sqrt(d)
    if hyp == "c":
        p = np.abs(1 / np.sqrt(d) + fu) ** 2 / 2
    else:
        p = (1 / d + np.abs(fu) ** 2) / 2
    p = p / p.sum()
    js = rng.choice(d, size=K, p=p)
    # w[k] = (1/sqrt d) sum_t exp(2 pi i j_t k / d), k >= 1
    counts = np.bincount(js, minlength=d).astype(float)
    w = np.fft.ifft(counts) * d / np.sqrt(d)
    w[0] = 0
    if np.all(counts == counts[0]):  # projected sum is exactly zero: use the fixed fallback direction |1>
        STATS["fallback_used"] += 1
        w = np.zeros(d, dtype=complex)
        w[1] = 1.0
    else:
        STATS["min_spread_over_K"] = min(STATS["min_spread_over_K"], float(counts @ counts - K * K / d) / K)
        w = w / np.linalg.norm(w)
    ov = np.vdot(w, u)                        # <w|u>
    if hyp == "c":
        pp, pm = abs(1 + ov) ** 2 / 4, abs(1 - ov) ** 2 / 4
    else:
        pp = pm = 0.25 + abs(ov) ** 2 / 4
    n = rng.multinomial(T, [pp, pm, max(0.0, 1 - pp - pm)])
    s = n[0] - n[1]
    return bool(s > 0 or (s == 0 and rng.random() < 0.5))


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    for d in (256, 1024, 4096):
        D = d - 1
        for N in (4, 8, 16, 32, 64, 128, 256, 512):
            K = T = N // 2
            pc = np.mean([one_trial(d, K, T, "c", rng) for _ in range(TRIALS)])
            pm = np.mean([one_trial(d, K, T, "m", rng) for _ in range(TRIALS)])
            adv = pc - pm
            se = np.sqrt(pc * (1 - pc) / TRIALS + pm * (1 - pm) / TRIALS)
            rows.append({"d": d, "N": N, "N_over_sqrtD": N / np.sqrt(D), "P_c_decide_c": pc, "P_m_decide_c": pm,
                         "advantage": adv, "se": se, "trials_per_hypothesis": TRIALS})
            print(rows[-1])
    out = {"seed": SEED, "rows": rows, "runtime_s": time.time() - t0, "fallback_stats": STATS,
           "note": "Monte Carlo lower bound on the optimal TV for one explicit adaptive protocol; not a proof"}
    with open("results/learn_then_test.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
