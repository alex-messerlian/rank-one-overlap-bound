"""PPT-BOTH telescoping route: exact per-step trace norms against the proved closed-form bounds, and the resulting
copy lower bound.

Per step m (last register transposed), with sigma = (|0><0| + Pi/D)/2:
    X^c_m = omega_c^(m) - omega_c^(m-1) (x) sigma = (A_m + B_m)/2,
    A_m = E[|v><v|^{m-1} (x) |u><u|] - omega_c^(m-1) (x) Pi/D,   B_m = E[|v><v|^{m-1} (x) (|0><u| + |u><0|)],
    X^m_m = omega_m^(m) - omega_m^(m-1) (x) sigma = A'_m / 2.
Proved: ||A_m^G||_1, ||A'_m^G||_1 <= 2 E_m and ||B_m^G||_1 = 2 ||W_m||_1 (closed forms in src/rankone.py).

Run from code/:  PYTHONPATH=. python checks/check_ppt_telescoping.py
Writes results/ppt_telescoping.json.
"""

from __future__ import annotations

import json
import time

import numpy as np

from src import rankone as r1


def step_rows() -> list[dict]:
    rows = []
    for d, m in [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4), (5, 3), (8, 2), (8, 3), (16, 2)]:
        t0 = time.time()
        D = d - 1
        s = r1.sigma(d)
        Pi = np.eye(d)
        Pi[0, 0] = 0
        oc_m = r1.omega_expansion(d, m, "c")
        oc_prev = r1.omega_expansion(d, m - 1, "c")
        om_m = r1.omega_expansion(d, m, "m")
        om_prev = r1.omega_expansion(d, m - 1, "m")
        parts = r1.register_parts(d, m)
        e00 = np.zeros((d, d))
        e00[0, 0] = 1
        A = parts["uu"] - np.kron(oc_prev, Pi / D)
        B = parts["coh"]
        Xc = oc_m - np.kron(oc_prev, s)
        Xm = om_m - np.kron(om_prev, s)
        pm = r1.register_parts_mixed(d, m)
        Ap = pm["uu"] - np.kron(om_prev, Pi / D)
        rec = {
            "d": d, "m": m,
            "check_P00_equals_prev_x_00": float(np.abs(parts["00"] - np.kron(oc_prev, e00)).max()),
            "check_Xc_equals_(A+B)/2": float(np.abs(Xc - (A + B) / 2).max()),
            "check_Xm_equals_A'/2": float(np.abs(Xm - Ap / 2).max()),
            "trace_norm_Xc_Gamma": r1.trace_norm(r1.partial_transpose_last(Xc, d, m)),
            "trace_norm_Xm_Gamma": r1.trace_norm(r1.partial_transpose_last(Xm, d, m)),
            "trace_norm_A_Gamma": r1.trace_norm(r1.partial_transpose_last(A, d, m)),
            "trace_norm_Aprime_Gamma": r1.trace_norm(r1.partial_transpose_last(Ap, d, m)),
            "bound_2E_m": 2 * r1.E_m_closed(d, m),
            "trace_norm_B_Gamma": r1.trace_norm(r1.partial_transpose_last(B, d, m)),
            "closed_2W_m": 2 * r1.W_trace_norm_closed(d, m),
            "bound_Xc_E_plus_W": r1.E_m_closed(d, m) + r1.W_trace_norm_closed(d, m),
            "bound_Xm_E": r1.E_m_closed(d, m),
            "jensen_sqrt((m-1)/(2D))": np.sqrt((m - 1) / (2 * D)),
            "seconds": time.time() - t0,
        }
        rows.append(rec)
    return rows


def first_n_simple(d: int) -> int:
    """Smallest n with the simplified proved bound >= 1/3 (the simplified bound dominates the exact telescoping
    bound, so every n below this is certified insufficient)."""
    D = d - 1
    n_max = int(4 * (d - 1) ** 0.5) + 10
    j = np.arange(1, n_max + 1, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(np.sqrt(j))])   # csum[n-1] = sum_{j=1}^{n-1} sqrt(j)
    n = np.arange(2, n_max + 1)
    vals = n * (n - 1) / (4 * D) + csum[n - 1] / (2 * np.sqrt(2 * D))
    return int(n[np.argmax(vals >= 1 / 3)])


def lower_bound_table() -> list[dict]:
    rows = []
    for q in range(3, 41, 1):
        d = 2 ** q
        n_s = first_n_simple(d)
        rec = {"qubits": q, "d": d, "n_min_certified_simple": n_s, "(d-1)^(1/3)": (d - 1) ** (1 / 3),
               "sqrt(d-1)": (d - 1) ** 0.5, "ratio_n_min/(d-1)^(1/3)": n_s / (d - 1) ** (1 / 3)}
        if n_s <= 3000:
            n = 2
            while r1.ppt_bias_bound(d, n) < 1 / 3:
                n += 1
            rec["n_min_certified_exact"] = n
        rows.append(rec)
    return rows


def simple_bound_rows() -> list[dict]:
    rows = []
    for q in (5, 8, 12, 16, 20, 30):
        d = 2 ** q
        n = int(np.floor((d - 1) ** (1 / 3)))
        rows.append({"d": d, "n=floor((d-1)^(1/3))": n, "exact_bound": r1.ppt_bias_bound(d, n) if n <= 5000 else None,
                     "simple_bound": r1.ppt_bias_bound_simple(d, n), "below_1/3": r1.ppt_bias_bound_simple(d, n) < 1 / 3})
    return rows


def main() -> None:
    t0 = time.time()
    out = {"steps": step_rows(), "lower_bound": lower_bound_table(), "simple_bound": simple_bound_rows()}
    out["runtime_s"] = time.time() - t0
    with open("results/ppt_telescoping.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    for key in ("steps", "lower_bound", "simple_bound"):
        print("==", key)
        for row in out[key]:
            print({k: (round(float(v), 10) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main()
