"""N = 2: exact single-copy optimum, PPT bound, nonadaptive baselines, and the pointwise-ratio counterexample leaf.

Run from code/:  PYTHONPATH=. python checks/check_two_copy.py
Writes results/two_copy.json.
"""

from __future__ import annotations

import json
import time

import numpy as np

from src import rankone as r1


def adaptive_rows() -> list[dict]:
    rows = []
    for d in (4, 8, 16, 32, 64, 128, 256):
        tv = r1.tv_adaptive_two_copy(d)
        rows.append({"d": d, "tv_adaptive_protocol": tv, "closed_form_1_over_4sqrtD": 1 / (4 * np.sqrt(d - 1)),
                     "ratio": tv * 4 * np.sqrt(d - 1),
                     "tv_nonadaptive_same_fourier": r1.tv_nonadaptive_fourier_two_copy(d),
                     "d_times_tv_nonadaptive": d * r1.tv_nonadaptive_fourier_two_copy(d)})
    return rows


def cross_check_full_operators() -> list[dict]:
    """Recompute the adaptive protocol's TV from the full omega operators (independent of H_of / Y)."""
    rows = []
    for d in (4, 8):
        oc = r1.omega_expansion(d, 2, "c")
        om = r1.omega_expansion(d, 2, "m")
        F = r1.fourier_basis(d)
        tv = 0.0
        for j in range(d):
            x1 = F[:, j]
            # second-step basis: eigenvectors of the partial matrix element of (oc - om) on copy 1
            Dm = (oc - om).reshape(d, d, d, d)
            H = np.einsum("a,abcd,c->bd", x1.conj(), Dm, x1)
            _, E = np.linalg.eigh((H + H.conj().T) / 2)
            for k in range(d):
                x = np.kron(x1, E[:, k])
                tv += abs(np.real(x.conj() @ (oc - om) @ x))
        tv *= 0.5
        Gam = r1.partial_transpose_last(oc - om, d, 2)
        rows.append({"d": d, "tv_protocol_full_operators": tv, "closed_form": 1 / (4 * np.sqrt(d - 1)),
                     "half_trace_norm_Delta2_Gamma": 0.5 * r1.trace_norm(Gam),
                     "half_trace_norm_Delta2": 0.5 * r1.trace_norm(oc - om)})
    return rows


def ppt_upper_rows() -> list[dict]:
    rows = []
    for d in (4, 8, 16, 32):
        Y = r1.Y_operator(d) / (4 * (d - 1))
        rows.append({"d": d, "half_trace_norm_Delta2_Gamma": 0.5 * r1.trace_norm(r1.partial_transpose_last(Y, d, 2)),
                     "closed_form_1_over_4sqrtD": 1 / (4 * np.sqrt(d - 1))})
    return rows


def coherence_blind_and_fourier_N3() -> list[dict]:
    rows = []
    for d in (4, 8):
        for N in (2, 3):
            Delta = r1.omega_expansion(d, N, "c") - r1.omega_expansion(d, N, "m")
            rows.append({"d": d, "N": N,
                         "tv_computational_basis": r1.tv_product_basis(Delta, np.eye(d), N),
                         "tv_same_fourier_basis": r1.tv_product_basis(Delta, r1.fourier_basis(d), N)})
    return rows


def pointwise_counterexample() -> list[dict]:
    """Leaf x1 = a|0> + b|w>, x2 = a|0> - b|w> with |b|^2 = r D |a|^2, r = 1/c, c = 2D/(D+1).
    Closed forms: p_c/p_sigma = 2D/(3D+1), p_c/p_m = (3D+1)/(5D+3)."""
    rows = []
    for d in (4, 8):
        D = d - 1
        c = 2 * D / (D + 1)
        r = 1 / c
        a2 = 1 / (1 + r * D)
        a, b = np.sqrt(a2), np.sqrt(1 - a2)
        w = np.zeros(d, dtype=complex)
        w[1] = 1
        e0 = np.zeros(d, dtype=complex)
        e0[0] = 1
        x = np.kron(a * e0 + b * w, a * e0 - b * w)
        oc = r1.omega_expansion(d, 2, "c")
        om = r1.omega_expansion(d, 2, "m")
        sg = np.kron(r1.sigma(d), r1.sigma(d))
        pc, pm, ps = (float(np.real(x.conj() @ M @ x)) for M in (oc, om, sg))
        rows.append({"d": d, "pc_over_psigma": pc / ps, "closed_2D_over_3D+1": 2 * D / (3 * D + 1),
                     "pc_over_pm": pc / pm, "closed_(3D+1)_over_(5D+3)": (3 * D + 1) / (5 * D + 3),
                     "psigma_leaf": ps})
    return rows


def main() -> None:
    t0 = time.time()
    out = {"adaptive": adaptive_rows(), "cross_check": cross_check_full_operators(), "ppt_upper": ppt_upper_rows(),
           "nonadaptive_small": coherence_blind_and_fourier_N3(), "pointwise_counterexample": pointwise_counterexample()}
    out["runtime_s"] = time.time() - t0
    with open("results/two_copy.json", "w") as f:
        json.dump(out, f, indent=1)
    for key in ("adaptive", "cross_check", "ppt_upper", "nonadaptive_small", "pointwise_counterexample"):
        print("==", key)
        for row in out[key]:
            print({k: (round(v, 12) if isinstance(v, float) else v) for k, v in row.items()})
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
