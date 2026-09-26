"""Checks of explicit values in the paper: the two-copy PPT value at large d (sparse), the D = 31 number in the proof
of Theorem 3, and the chi^2 of Appendix A. Nothing here allocates a dense d^2 x d^2 operator at large d.

Run from code/:  python -m pytest -q tests
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh


def ppt_two_copy_value_sparse(d: int) -> float:
    """Half trace norm of Delta_2^Gamma = (|Phi><00| + |00><Phi|)/(4D), with Phi = sum_{i>=1} |ii>, built as a sparse
    d^2 x d^2 matrix (2D nonzeros). The two nonzero eigenvalues come from a Lanczos solver."""
    D = d - 1
    rows, cols = [], []
    for i in range(1, d):
        rows += [i * d + i, 0]
        cols += [0, i * d + i]
    M = sps.coo_matrix((np.full(len(rows), 1 / (4 * D)), (rows, cols)), shape=(d * d, d * d)).tocsr()
    vals = eigsh(M, k=2, which="LM", return_eigenvectors=False)
    return 0.5 * float(np.abs(vals).sum())


@pytest.mark.parametrize("d", [128, 1024])
def test_two_copy_ppt_value_large_d_sparse(d: int) -> None:
    """Theorem 4 at large d. Sparse replacement for the dense d = 128 case in test_hard_pair.py (a 2 GiB matrix)."""
    assert abs(ppt_two_copy_value_sparse(d) - 1 / (4 * np.sqrt(d - 1))) < 1e-12


def test_theorem3_value_at_D31() -> None:
    """Proof of Theorem 3: at D = 31 the bound D^(-1/3)/4 + 1/(3 sqrt 2) equals 0.31528510... < 1/3."""
    val = 31 ** (-1 / 3) / 4 + 1 / (3 * np.sqrt(2))
    assert abs(val - 0.31528510235695917) < 1e-15
    assert 0.315 < val < 0.316 < 1 / 3


def test_counterexample_chi2_rational() -> None:
    """Appendix A: chi^2 of the binary measurement from the exact conditional marginals (13/70 vs 23/110)."""
    pc, pm = Fraction(13, 70), Fraction(23, 110)
    chi2 = (pc - pm) ** 2 / pm + (pc - pm) ** 2 / (1 - pm)
    assert chi2 == Fraction(108, 32683)
