"""Part A/B checks: basic facts about the hard pair and the nested-subspace identity for the averaged moments.

Run from code/:  PYTHONPATH=. python checks/check_identities.py
Writes results/identities.json.
"""

from __future__ import annotations

import json
import time
from math import comb

import numpy as np

from src import rankone as r1

SEED_BASIC = 20260925
SEED_MC = 404


def basic_facts() -> dict:
    rng = np.random.default_rng(SEED_BASIC)
    rows = []
    for d in (2, 4, 8, 16, 64):
        e0 = np.zeros(d)
        e0[0] = 1
        P = np.outer(e0, e0)
        worst = {"q_c_err": 0.0, "q_m_err": 0.0, "trace_err": 0.0, "min_eig": 1.0}
        ranks = set()
        for _ in range(50):
            rc, rm = r1.rho_pair(r1.haar_u(d, rng))
            worst["q_c_err"] = max(worst["q_c_err"], abs(np.trace(P @ rc @ rc).real - 0.5))
            worst["q_m_err"] = max(worst["q_m_err"], abs(np.trace(P @ rm @ rm).real - 0.25))
            worst["trace_err"] = max(worst["trace_err"], abs(np.trace(rc).real - 1), abs(np.trace(rm).real - 1))
            worst["min_eig"] = min(worst["min_eig"], np.linalg.eigvalsh(rc).min(), np.linalg.eigvalsh(rm).min())
            ranks.add((int(np.linalg.matrix_rank(rc, tol=1e-10)), int(np.linalg.matrix_rank(rm, tol=1e-10))))
            # population of |0> is 1/2 for both
            worst["pop_err"] = max(worst.get("pop_err", 0.0), abs(rc[0, 0].real - 0.5), abs(rm[0, 0].real - 0.5))
        rows.append({"d": d, **{k: float(v) for k, v in worst.items()}, "ranks(c,m)": sorted(ranks)})
    return {"samples_per_d": 50, "seed": SEED_BASIC, "rows": rows}


def identity_checks() -> dict:
    rows = []
    for d, N in [(2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (5, 2), (8, 1), (8, 2), (8, 3)]:
        t0 = time.time()
        rec = {"d": d, "N": N}
        for which in ("c", "m"):
            A = r1.omega_expansion(d, N, which)
            B = r1.omega_projector_form(d, N, which)
            ev = np.linalg.eigvalsh(A)
            rec[f"max_abs_diff_{which}"] = float(np.abs(A - B).max())
            rec[f"trace_{which}"] = float(np.trace(A))
            rec[f"min_eig_{which}"] = float(ev.min())
            if which == "c":
                Ac = A
            else:
                Am = A
        rec["single_copy_equals_sigma"] = bool(N == 1 and np.allclose(Ac, r1.sigma(d)) and np.allclose(Am, r1.sigma(d)))
        if N == 2:
            rec["Delta2_minus_Y_over_4D_maxabs"] = float(np.abs((Ac - Am) - r1.Y_operator(d) / (4 * (d - 1))).max())
            rec["joint_TV_half_trace_norm"] = 0.5 * r1.trace_norm(Ac - Am)
        rec["seconds"] = time.time() - t0
        rows.append(rec)
    return {"rows": rows}


def monte_carlo_checks() -> dict:
    rows = []
    for d, N, samples in [(4, 2, 40000), (4, 3, 40000), (8, 2, 20000)]:
        mc_c, mc_m = r1.omega_monte_carlo(d, N, samples, SEED_MC + d + N)
        ex_c = r1.omega_expansion(d, N, "c")
        ex_m = r1.omega_expansion(d, N, "m")
        rows.append({"d": d, "N": N, "samples": samples, "seed": SEED_MC + d + N,
                     "max_abs_dev_c": float(np.abs(mc_c - ex_c).max()), "max_abs_dev_m": float(np.abs(mc_m - ex_m).max()),
                     "max_entry_c": float(np.abs(ex_c).max()), "stat_scale_1_over_sqrt_samples": 1 / np.sqrt(samples)})
    return {"rows": rows, "note": "floating-point Monte Carlo sanity check only; the identity itself is proved analytically"}


def main() -> None:
    t0 = time.time()
    out = {"basic": basic_facts(), "identity": identity_checks(), "monte_carlo": monte_carlo_checks()}
    out["runtime_s"] = time.time() - t0
    with open("results/identities.json", "w") as f:
        json.dump(out, f, indent=1)
    for r in out["identity"]["rows"]:
        print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in out["monte_carlo"]["rows"]:
        print(r)
    for r in out["basic"]["rows"]:
        print(r)
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
