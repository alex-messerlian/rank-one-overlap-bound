"""Cross-check Figure 1's exact thresholds at 40-digit precision (mpmath) against results/figure_data.json."""
import json, sys
from mpmath import mp, mpf, binomial, sqrt
mp.dps = 40
data = json.load(open(sys.argv[1]))
third = mpf(1) / 3
def step(m, D):
    n = m - 1
    tot = mpf(0)
    for k in range(n + 1):
        p = binomial(n, k) / mpf(2) ** n
        tot += p * (mpf(k) / (D + k) + sqrt(mpf(n - k) / (D + k)) / 2)
    return tot
mism, min_margin = [], None
for d, thr in zip(data["d_grid"], data["exact_threshold"]):
    D, beta, N, prev = d - 1, mpf(0), 1, mpf(0)
    while beta < third:
        N += 1; prev = beta; beta += step(N, D)
    if N != thr: mism.append((d, thr, N))
    margin = min(third - prev, beta - third)
    min_margin = margin if min_margin is None or margin < min_margin else min_margin
out = {"grid_points": len(data["d_grid"]), "mismatches": mism, "min_margin_to_one_third": float(min_margin),
       "ok": not mism}
json.dump(out, open(sys.argv[2], "w"), indent=1); print(out)
