# Code and data behind the paper

These scripts check every proved identity, every explicit number and Figure 1 of the paper. They support the proofs
but are not part of them. Run everything from this folder.

**Requirements:** Python 3.12 with the packages in `requirements.txt` (NumPy, SciPy, SymPy, mpmath, Matplotlib,
pytest). They were last run with the versions pinned there.

```bash
python -m pip install -r requirements.txt
```

| Command (from `code/`) | What it checks | Paper | Output | Time |
|---|---|---|---|---|
| `python -m pytest -q tests --deselect "tests/test_hard_pair.py::test_two_copy_adaptive_protocol_attains_ppt_bound[128]"` | 30 tests: moment structure, step norms, the two-copy value, the certified bound, the D = 31 value, the Appendix A counterexample | throughout | — | ~2 s |
| `PYTHONPATH=. python checks/check_identities.py` | The moment structure (Proposition 7) against direct Haar averages | §3 | `results/identities.json` | ~5 s |
| `PYTHONPATH=. python checks/check_ppt_telescoping.py` | Lemmas 10 and 11, Theorem 8 and the bound of Theorem 3, against exact trace norms | §4 | `results/ppt_telescoping.json` | ~5 min |
| `PYTHONPATH=. python checks/check_two_copy.py` | Theorem 4, the value 1/(4√(d−1)), and Remark 13, the value 1/(2d) | §5 | `results/two_copy.json` | ~20 s |
| `python checks/check_counterexample.py` | Appendix A, in exact rational arithmetic | App. A | `results/counterexample.json` | ~1 s |
| `python checks/verify_numbers.py results/verify_numbers.json` | Every explicit number: the D = 31 value, the d = 64 thresholds, the constants of Proposition 14, the Fourier values | §4–6 | `results/verify_numbers.json` | seconds |
| `python checks/check_learn_then_test.py` | The simulated protocol of Section 6 (seed 20260926) | §6, bars in Fig. 1 | `results/learn_then_test.json` | ~16 s |
| `python checks/make_figure.py results/learn_then_test.json ../paper/figures/copies_vs_dimension.pdf results/figure_data.json` | Draws Figure 1 | Fig. 1 | the figure, `results/figure_data.json` | seconds |
| `python checks/crosscheck_figure.py results/figure_data.json results/figure_crosscheck.json` | Recomputes every plotted threshold to 40 digits | Fig. 1 | `results/figure_crosscheck.json` | under a minute |

## Notes

- **The deselected test** builds a dense 2 GiB matrix at d = 128. The same value is checked sparsely at d = 128 and
  d = 1024 in `tests/test_values.py`.
- **Last full run (25 September 2026):** all 30 tests passed, every regenerated result matched the file here apart
  from timing fields, and the regenerated figure was byte-identical to `paper/figures/copies_vs_dimension.pdf`.
- **The simulated protocol** falls back to a fixed direction if the learned vector is exactly zero. With this seed
  that never happens.
