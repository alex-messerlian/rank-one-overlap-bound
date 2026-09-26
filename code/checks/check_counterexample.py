"""Exact check of the Appendix A counterexample to the posterior-coherence inequality.

d = 4. Copies 1 and 2 are measured in the basis {x+ = (|0>+|1>)/sqrt2, x- = (|0>-|1>)/sqrt2, |2>, |3>}; condition on
the history (+, -); copy 3 is measured with {|1><1|, I - |1><1|}.
Claimed: P_c(+,-) = 7/96, P_m(+,-) = 11/96; conditional states diag(1/2, 13/70, 11/70, 11/70) and
diag(1/2, 23/110, 8/55, 8/55); chi^2(P_c || P_m) = 108/32683 while the coherent posterior mean is mu = 0.

Route: exact rational omega^(3) built with sympy from the Haar vector-moment expansion. This is a separate exact
implementation, independent of the floating-point code in src/rankone.py. The mu = 0 claim is checked by symmetry and by seeded
Monte Carlo.
Run from code/:  python checks/check_counterexample.py
"""

from __future__ import annotations

import json
from itertools import combinations, permutations, product

import numpy as np
import sympy as sp

d, N = 4, 3
D = d - 1


def rising(k: int) -> int:
    out = 1
    for j in range(k):
        out *= D + j
    return out


def omega_exact(which: str) -> sp.Matrix:
    M = sp.zeros(d ** N, d ** N)
    for k in range(N + 1):
        w = sp.Rational(1, rising(k)) / 2 ** N
        for S in combinations(range(N), k):
            for T in (combinations(range(N), k) if which == "c" else [S]):
                for pi in permutations(range(k)):
                    for i in product(range(D), repeat=k):
                        ket, bra = [0] * N, [0] * N
                        for a, s in enumerate(S):
                            ket[s] = i[a] + 1
                        for a in range(k):
                            bra[T[pi[a]]] = i[a] + 1
                        ki = sum(ket[t] * d ** (N - 1 - t) for t in range(N))
                        bi = sum(bra[t] * d ** (N - 1 - t) for t in range(N))
                        M[ki, bi] += w
    return M


def main() -> None:
    s2 = sp.sqrt(2)
    e = [sp.Matrix([1 if j == i else 0 for j in range(d)]) for i in range(d)]
    xp, xm = (e[0] + e[1]) / s2, (e[0] - e[1]) / s2
    out = {}
    for h in ("c", "m"):
        W = omega_exact(h)
        # conditional (unnormalised) state of copy 3: (<x+| (x) <x-| (x) I) W (|x+> (x) |x-> (x) I)
        L = sp.kronecker_product(xp, xm)  # 16 x 1
        blocks = sp.zeros(d, d)
        for a in range(d):
            for b in range(d):
                col = sp.kronecker_product(L, e[b])
                row = sp.kronecker_product(L, e[a])
                blocks[a, b] = sp.nsimplify(sp.simplify((row.T * W * col)[0, 0]))
        p_hist = sp.simplify(blocks.trace())
        rho = sp.simplify(blocks / p_hist)
        out[h] = {"P_history": str(p_hist), "rho_bar_diag": [str(rho[i, i]) for i in range(d)],
                  "rho_bar_offdiag_max_abs": str(max(abs(sp.nsimplify(rho[i, j])) for i in range(d) for j in range(d) if i != j))}
        out[h]["_p1"] = rho[1, 1]
    pc, pm = out["c"].pop("_p1"), out["m"].pop("_p1")
    chi2 = sp.nsimplify((pc - pm) ** 2 / pm + (pc - pm) ** 2 / (1 - pm))
    out["chi2_binary"] = str(chi2)
    out["matches_paper"] = {
        "P_c": out["c"]["P_history"] == "7/96", "P_m": out["m"]["P_history"] == "11/96",
        "rho_c": out["c"]["rho_bar_diag"] == ["1/2", "13/70", "11/70", "11/70"],
        "rho_m": out["m"]["rho_bar_diag"] == ["1/2", "23/110", "8/55", "8/55"],
        "chi2": out["chi2_binary"] == "108/32683"}
    # mu = 0: the coherent likelihood |<x+|v>|^2 |<x-|v>|^2 is invariant under u -> -u (it swaps the two factors),
    # and Haar measure is invariant under u -> -u, so E_c[u | history] = 0. Monte Carlo check (seeded):
    rng = np.random.default_rng(4040)
    g = rng.standard_normal((400000, D)) + 1j * rng.standard_normal((400000, D))
    u = np.zeros((400000, d), dtype=complex)
    u[:, 1:] = g / np.linalg.norm(g, axis=1, keepdims=True)
    lik = np.abs((1 + u[:, 1]) / 2) ** 2 * np.abs((1 - u[:, 1]) / 2) ** 2   # <x+-|v> = (1 +- <1|u>)/2
    mu = (lik[:, None] * u).sum(0) / lik.sum()
    out["mu_monte_carlo_max_abs"] = float(np.abs(mu).max())
    out["mu_monte_carlo_scale"] = float(1 / np.sqrt(400000))
    with open("results/counterexample.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
