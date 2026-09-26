"""Exact small-d objects for the rank-one nonlinear overlap q_P(rho) = Tr(P rho^2), P = |0><0|.

Hard pair (u Haar on the unit sphere of C^D, D = d - 1, the complement of |0>):
    rho_c(u) = |v><v|,  v = (|0> + u)/sqrt(2);      rho_m(u) = (|0><0| + |u><u|)/2.
Averaged N-copy states:  omega_c^(N) = E_u rho_c^{(x)N},  omega_m^(N) = E_u rho_m^{(x)N}.

Two independent constructions are provided:
  * `omega_expansion`: expand every copy over (ket, bra) in {0, u}^2 and apply the Haar vector-moment formula
    E[u_{i_1}..u_{i_k} conj(u_{j_1})..conj(u_{j_k})] = sum_{pi in S_k} prod_a delta(i_a, j_{pi(a)}) / (D (D+1) ... (D+k-1));
  * `omega_projector_form`: the claimed closed form
        omega_c^(N) = 2^{-N} sum_k binom(N,k) P_{H_k} / D_k,   omega_m^(N) = 2^{-N} sum_k P_{V_k} / D_k,
    built from the full-space symmetrizer and position projectors (no Haar formula used).
Dense matrices are (d^N x d^N); callers keep d^N <= 4096.
"""

from __future__ import annotations

from itertools import combinations, permutations, product
from math import comb, factorial

import numpy as np


# ------------------------------------------------------------------------------------------------------------------
# basic tensor utilities
# ------------------------------------------------------------------------------------------------------------------

def digits(d: int, N: int) -> np.ndarray:
    """(d^N, N) array of base-d digits, most significant first (register 1 first)."""
    idx = np.arange(d ** N)
    out = np.empty((d ** N, N), dtype=np.int64)
    for t in range(N - 1, -1, -1):
        out[:, t] = idx % d
        idx //= d
    return out


def index_of(dig: np.ndarray, d: int) -> np.ndarray:
    idx = np.zeros(dig.shape[0], dtype=np.int64)
    for t in range(dig.shape[1]):
        idx = idx * d + dig[:, t]
    return idx


def perm_operator(d: int, N: int, perm: tuple[int, ...]) -> np.ndarray:
    """P_perm |x_1 .. x_N> = |x_{perm^-1(1)} .. >: register t of the output carries the input register perm^-1(t)."""
    dig = digits(d, N)
    inv = np.argsort(perm)
    out_idx = index_of(dig[:, inv], d)
    P = np.zeros((d ** N, d ** N))
    P[out_idx, np.arange(d ** N)] = 1.0
    return P


def symmetrizer(d: int, N: int, positions: tuple[int, ...] | None = None) -> np.ndarray:
    """Average of the permutation operators permuting `positions` (all registers if None)."""
    pos = tuple(range(N)) if positions is None else tuple(positions)
    acc = np.zeros((d ** N, d ** N))
    count = 0
    for p in permutations(pos):
        perm = list(range(N))
        for a, b in zip(pos, p):
            perm[a] = b
        acc += perm_operator(d, N, tuple(perm))
        count += 1
    return acc / count


def rising(D: int, k: int) -> int:
    out = 1
    for j in range(k):
        out *= D + j
    return out


def dim_sym(D: int, k: int) -> int:
    """D_k = dim Sym^k(C^D) = binom(D + k - 1, k)."""
    return comb(D + k - 1, k)


def partial_transpose_last(X: np.ndarray, d: int, N: int) -> np.ndarray:
    """Transpose of the last register in the computational basis (which contains |0>)."""
    a = d ** (N - 1)
    T = X.reshape(a, d, a, d).transpose(0, 3, 2, 1)
    return T.reshape(a * d, a * d)


def trace_norm(X: np.ndarray) -> float:
    return float(np.abs(np.linalg.eigvalsh((X + X.conj().T) / 2)).sum())


# ------------------------------------------------------------------------------------------------------------------
# construction 1: Haar-moment expansion
# ------------------------------------------------------------------------------------------------------------------

def _op_ST(d: int, N: int, S: tuple[int, ...], T: tuple[int, ...]) -> np.ndarray:
    """E_u |0^{S^c} u^S><0^{T^c} u^T| (kets u at positions S, bras u at positions T, |S| = |T|)."""
    D, k = d - 1, len(S)
    M = np.zeros((d ** N, d ** N))
    w = 1.0 / rising(D, k)
    base_ket = [0] * N
    for pi in permutations(range(k)):
        for i in product(range(D), repeat=k):
            ket = base_ket.copy()
            bra = base_ket.copy()
            for a, s in enumerate(S):
                ket[s] = i[a] + 1
            # strand a joins ket position S[a] to bra position T[pi[a]]
            for a in range(k):
                bra[T[pi[a]]] = i[a] + 1
            ki = 0
            bi = 0
            for t in range(N):
                ki = ki * d + ket[t]
                bi = bi * d + bra[t]
            M[ki, bi] += w
    return M


def omega_expansion(d: int, N: int, which: str) -> np.ndarray:
    """omega_c (all S, T with |S| = |T|) or omega_m (S = T only), times 2^{-N}."""
    acc = np.zeros((d ** N, d ** N))
    for k in range(N + 1):
        for S in combinations(range(N), k):
            Ts = combinations(range(N), k) if which == "c" else [S]
            for T in Ts:
                acc += _op_ST(d, N, S, T)
    return acc / 2 ** N


def register_parts(d: int, m: int) -> dict[str, np.ndarray]:
    """Split omega_c^(m) by what the LAST register carries (ket, bra in {0, u}), with the 2^{-(m-1)} prefactor of
    the first m-1 registers, so that |v><v| = (|0><0| + |0><u| + |u><0| + |u><u|)/2 gives
        omega_c^(m) = (P00 + Pcoh + Puu)/2.
    Returns P00 (= omega_c^(m-1) (x) |0><0|), Puu (= E[|v><v|^{(m-1)} (x) |u><u|]) and Pcoh."""
    acc = {"00": np.zeros((d ** m, d ** m)), "uu": np.zeros((d ** m, d ** m)), "coh": np.zeros((d ** m, d ** m))}
    last = m - 1
    for k in range(m + 1):
        for S in combinations(range(m), k):
            for T in combinations(range(m), k):
                key = ("uu" if last in S else "coh") if last in T else ("coh" if last in S else "00")
                acc[key] += _op_ST(d, m, S, T)
    return {key: val / 2 ** (m - 1) for key, val in acc.items()}


def register_parts_mixed(d: int, m: int) -> dict[str, np.ndarray]:
    """Same split for omega_m^(m): only S = T terms, so there is no coherent part."""
    acc = {"00": np.zeros((d ** m, d ** m)), "uu": np.zeros((d ** m, d ** m))}
    last = m - 1
    for k in range(m + 1):
        for S in combinations(range(m), k):
            acc["uu" if last in S else "00"] += _op_ST(d, m, S, S)
    return {key: val / 2 ** (m - 1) for key, val in acc.items()}


# ------------------------------------------------------------------------------------------------------------------
# construction 2: nested-subspace projector form
# ------------------------------------------------------------------------------------------------------------------

def Q_positions(d: int, N: int, S: tuple[int, ...]) -> np.ndarray:
    """Projector: registers in S in the complement of |0>, all other registers equal to |0>."""
    dig = digits(d, N)
    mask = np.ones(d ** N, dtype=bool)
    for t in range(N):
        mask &= (dig[:, t] != 0) if t in S else (dig[:, t] == 0)
    return np.diag(mask.astype(float))


def omega_projector_form(d: int, N: int, which: str) -> np.ndarray:
    D = d - 1
    Psym = symmetrizer(d, N)
    acc = np.zeros((d ** N, d ** N))
    for k in range(N + 1):
        if which == "c":
            Qk = sum((Q_positions(d, N, S) for S in combinations(range(N), k)), np.zeros((d ** N, d ** N)))
            acc += comb(N, k) * (Psym @ Qk) / dim_sym(D, k)
        else:
            for S in combinations(range(N), k):
                acc += (Q_positions(d, N, S) @ symmetrizer(d, N, S)) / dim_sym(D, k)
    return acc / 2 ** N


# ------------------------------------------------------------------------------------------------------------------
# states, reference and Monte Carlo
# ------------------------------------------------------------------------------------------------------------------

def sigma(d: int) -> np.ndarray:
    """Common single-copy average (|0><0| + Pi_perp/D)/2."""
    D = d - 1
    s = np.eye(d) / (2 * D)
    s[0, 0] = 0.5
    return s


def haar_u(d: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.standard_normal(d - 1) + 1j * rng.standard_normal(d - 1)
    u = np.zeros(d, dtype=complex)
    u[1:] = g / np.linalg.norm(g)
    return u


def rho_pair(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = u.size
    e0 = np.zeros(d, dtype=complex)
    e0[0] = 1
    v = (e0 + u) / np.sqrt(2)
    return np.outer(v, v.conj()), (np.outer(e0, e0) + np.outer(u, u.conj())) / 2


def kron_power(M: np.ndarray, N: int) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for _ in range(N):
        out = np.kron(out, M)
    return out


def omega_monte_carlo(d: int, N: int, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    acc_c = np.zeros((d ** N, d ** N), dtype=complex)
    acc_m = np.zeros((d ** N, d ** N), dtype=complex)
    for _ in range(samples):
        rc, rm = rho_pair(haar_u(d, rng))
        acc_c += kron_power(rc, N)
        acc_m += kron_power(rm, N)
    return acc_c / samples, acc_m / samples


# ------------------------------------------------------------------------------------------------------------------
# closed forms used in the derivation
# ------------------------------------------------------------------------------------------------------------------

def _binom_half_pmf(n: int) -> np.ndarray:
    """P(L = l), L ~ Binomial(n, 1/2), l = 0..n, via log-gamma (stable for large n)."""
    from scipy.special import gammaln
    l = np.arange(n + 1)
    return np.exp(gammaln(n + 1) - gammaln(l + 1) - gammaln(n - l + 1) - n * np.log(2.0))


def W_trace_norm_closed(d: int, m: int) -> float:
    """||W_m||_1 = 2^{-(m-1)} sum_{l=0}^{m-2} sqrt(binom(m-1,l+1) binom(m-1,l) (l+1)/(D+l))
                = E_{L ~ Bin(m-1,1/2)} sqrt((m-1-L)/(D+L))   (using binom(m-1,l+1)(l+1) = binom(m-1,l)(m-1-l))."""
    D = d - 1
    pmf = _binom_half_pmf(m - 1)
    l = np.arange(m)
    return float(np.sum(pmf * np.sqrt((m - 1 - l) / (D + l))))


def E_m_closed(d: int, m: int) -> float:
    """E_m = E_{K ~ Bin(m-1,1/2)} K/(D+K) (half the proved bound on ||A_m^Gamma||_1); E_m <= (m-1)/(2D)."""
    D = d - 1
    pmf = _binom_half_pmf(m - 1)
    k = np.arange(m)
    return float(np.sum(pmf * k / (D + k)))


def ppt_bias_bound(d: int, n: int) -> float:
    """Proved PPT-BOTH bias bound sum_{m=2}^n [E_m + ||W_m||_1 / 2] for omega_c^(n) vs omega_m^(n) (n <= 5000)."""
    if n > 5000:
        raise ValueError("use ppt_bias_bound_simple for n > 5000")
    return sum(E_m_closed(d, m) + W_trace_norm_closed(d, m) / 2 for m in range(2, n + 1))


def ppt_bias_bound_simple(d: int, n: int) -> float:
    """Simplified proved bound n(n-1)/(4D) + (1/(2 sqrt(2D))) sum_{j=1}^{n-1} sqrt(j)."""
    D = d - 1
    return n * (n - 1) / (4 * D) + sum(np.sqrt(j) for j in range(1, n)) / (2 * np.sqrt(2 * D))


# ------------------------------------------------------------------------------------------------------------------
# N = 2 objects and protocols
# ------------------------------------------------------------------------------------------------------------------

def Y_operator(d: int) -> np.ndarray:
    """Y = sum_{i>=1} (|i0><0i| + |0i><i0|); claimed Delta_2 = omega_c^(2) - omega_m^(2) = Y/(4D)."""
    Y = np.zeros((d * d, d * d))
    for i in range(1, d):
        Y[i * d + 0, 0 * d + i] += 1
        Y[0 * d + i, i * d + 0] += 1
    return Y


def fourier_basis(d: int) -> np.ndarray:
    """Columns f_j with |<0|f_j>|^2 = 1/d (unbiased with respect to |0>)."""
    j = np.arange(d)
    return np.exp(2j * np.pi * np.outer(j, j) / d) / np.sqrt(d)


def H_of(x: np.ndarray) -> np.ndarray:
    """H_x = (<x| (x) I) Y (|x> (x) I) = <0|x> |0><Pi x| + <x|0> |Pi x><0|, with Pi the complement projector."""
    d = x.size
    e0 = np.zeros(d, dtype=complex)
    e0[0] = 1
    px = x.copy()
    px[0] = 0
    a = np.vdot(e0, x)  # <0|x>
    return a * np.outer(e0, px.conj()) + np.conj(a) * np.outer(px, e0.conj())


def tv_adaptive_two_copy(d: int) -> float:
    """Copy 1 in the Fourier basis; copy 2 in the eigenbasis of H_{x_1}. TV = (1/2) sum |<x1 x2|Delta_2|x1 x2>|."""
    D = d - 1
    F = fourier_basis(d)
    total = 0.0
    for j in range(d):
        H = H_of(F[:, j])
        evals = np.linalg.eigvalsh(H)
        total += np.abs(evals).sum()  # sum over the eigenbasis of |<e|H|e>| = trace norm of H
    return 0.5 * total / (4 * D)


def tv_nonadaptive_fourier_two_copy(d: int) -> float:
    D = d - 1
    F = fourier_basis(d)
    total = 0.0
    for j in range(d):
        H = H_of(F[:, j])
        total += np.abs(np.real(np.einsum("ik,ij,jk->k", F.conj(), H, F))).sum()
    return 0.5 * total / (4 * D)


def tv_product_basis(Delta: np.ndarray, B: np.ndarray, N: int) -> float:
    """Nonadaptive: the same orthonormal basis (columns of B) on every copy; TV = (1/2) sum_x |<x|Delta|x>|."""
    BN = kron_power(B, N)
    diag = np.real(np.einsum("ij,ik,kj->j", BN.conj(), Delta, BN))
    return 0.5 * float(np.abs(diag).sum())
