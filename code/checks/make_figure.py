"""Figure 1 of paper/paper.tex: copies needed to reach advantage 1/3 on the hard pair.

Curves:
  - exact certificate: the smallest N with beta_N >= 1/3, where beta_N is the exact sum of Theorem 8
    (E_m = E K/(D+K), ||W_m||_1 = E sqrt((m-1-L)/(D+L)), K, L ~ Bin(m-1, 1/2)); below it no PPT-BOTH effect,
    hence no single-copy protocol, reaches advantage 1/3.
  - Theorem 3's simplified bound (d-1)^(1/3), plotted for d >= 32.
  - bars: the explicit learn-then-test protocol of Section 6 (retained seeded Monte Carlo rows, seed 20260926):
    measured advantage below 1/3 at N = 2 sqrt(d) and above 1/3 at N = 4 sqrt(d). Not used in any proof.

Usage (from code/): python checks/make_figure.py <rows.json> <out.pdf> <data.json>
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

TARGET = 1 / 3


def _pmf(n: int) -> np.ndarray:
    ln2 = math.log(2)
    return np.array([math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * ln2)
                     for k in range(n + 1)])


def step_terms(m: int, D: int) -> float:
    """E_m + ||W_m||_1 / 2, the m-th term of beta_N."""
    n = m - 1
    k = np.arange(n + 1)
    p = _pmf(n)
    return float(np.dot(p, k / (D + k)) + 0.5 * np.dot(p, np.sqrt((n - k) / (D + k))))


def exact_threshold(d: int) -> int:
    """Smallest N >= 2 with beta_N >= 1/3 (N = 1 gives advantage 0)."""
    D, beta, N = d - 1, 0.0, 1
    while beta < TARGET:
        N += 1
        beta += step_terms(N, D)
    return N


def main(rows_path: str, out_pdf: str, data_path: str) -> None:
    ds = np.unique(np.round(np.logspace(math.log10(32), math.log10(20000), 1200)).astype(int))
    thr = [exact_threshold(int(d)) for d in ds]
    rows = json.load(open(rows_path))["rows"]
    bars = []
    for d in (256, 1024, 4096):
        s = math.isqrt(d)
        lo = next(r for r in rows if r["d"] == d and r["N"] == 2 * s)
        hi = next(r for r in rows if r["d"] == d and r["N"] == 4 * s)
        assert lo["advantage"] + 3 * lo["se"] < TARGET < hi["advantage"] - 3 * hi["se"], (d, lo, hi)
        bars.append({"d": d, "N_low": lo["N"], "adv_low": lo["advantage"], "se_low": lo["se"],
                     "N_high": hi["N"], "adv_high": hi["advantage"], "se_high": hi["se"]})

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    ax.fill_between(ds, 1.0, thr, step="post", color="#d9e6f2", lw=0, zorder=0)
    ax.step(ds, thr, where="post", color="#1f4e79", lw=1.3, zorder=3,
            label="Theorem 8, exact sum (proved)")
    dd = np.logspace(math.log10(32), math.log10(20000), 200)
    ax.plot(dd, (dd - 1) ** (1 / 3), color="black", lw=0.9, ls=(0, (4, 2)), zorder=3,
            label=r"$(d-1)^{1/3}$, Theorem 3 (proved)")
    for i, b in enumerate(bars):
        ax.plot([b["d"], b["d"]], [b["N_low"], b["N_high"]], color="#b5461d", lw=2.2, solid_capstyle="butt",
                zorder=4, label="explicit protocol (simulation)" if i == 0 else None)
        for y in (b["N_low"], b["N_high"]):
            ax.plot([b["d"] / 1.12, b["d"] * 1.12], [y, y], color="#b5461d", lw=1.0, zorder=4)
    ax.text(70, 2.35, "no protocol reaches advantage $1/3$", color="#1f4e79", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(32, 20000)
    ax.set_ylim(2, 400)
    ax.xaxis.set_major_locator(FixedLocator([32, 100, 1000, 10000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["32", "100", "1000", "10000"])
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("copies $N$ for advantage $1/3$")
    ax.legend(loc="upper left", frameon=False, fontsize=7.6, handlelength=2.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_pdf, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    json.dump({"target_advantage": TARGET, "d_grid": [int(x) for x in ds], "exact_threshold": thr,
               "checks": {str(d): exact_threshold(d) for d in (32, 64, 256, 1024, 4096, 20000)},
               "simulation_bars": bars,
               "note": "exact threshold from the exact sum of Theorem 8; bars from the retained seeded rows; "
                       "simulation not used in any proof"}, open(data_path, "w"), indent=1)
    print("thresholds at d=32,64,256,1024,4096,20000:",
          [exact_threshold(d) for d in (32, 64, 256, 1024, 4096, 20000)])
    print("bars:", [(b["d"], b["N_low"], round(b["adv_low"], 4), b["N_high"], round(b["adv_high"], 4)) for b in bars])


if __name__ == "__main__":
    main(*sys.argv[1:4])
