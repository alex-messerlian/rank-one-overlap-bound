# A single-copy lower bound for rank-one nonlinear overlap estimation

[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Paper license: CC BY 4.0](https://img.shields.io/badge/paper%20license-CC%20BY%204.0-lightgrey.svg)](paper/LICENSE.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](code/requirements.txt)

**Alexander Messerlian** · Independent Researcher, Palo Alto, CA, USA ·
ORCID [0009-0003-4933-6832](https://orcid.org/0009-0003-4933-6832)

**Paper:** [`paper/paper.pdf`](paper/paper.pdf)

## Abstract

Randomized single-copy measurements estimate Tr(Oρ²), for ‖O‖∞ ≤ 1, from O(√d) copies of a d-dimensional state at
constant accuracy and confidence; this is optimal for observables with large trace norm. We show that low rank does not
remove the dimension dependence. For a fixed rank-one projector P, any adaptive single-copy protocol that estimates
Tr(Pρ²) to error ε < 1/8 with success probability at least 2/3, for every state, needs more than (d−1)^(1/3) copies
when d ≥ 32. The proof extends partial-transpose telescoping to two ensembles differing only in coherence. For two
copies, we find the largest acceptance-probability difference for this pair exactly: 1/(4√(d−1)).

## Contents

| Folder | What is in it |
|---|---|
| `paper/` | `paper.pdf`, its LaTeX source `paper.tex` (the references are inside it), `figures/`, the Springer Nature template class `sn-jnl.cls`, and the paper's license |
| `code/` | Scripts, tests and saved results that check the identities, numbers and figure in the paper; see [`code/README.md`](code/README.md) |

## Reproducing the checks

The scripts need Python 3.12. From the `code/` folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests --deselect "tests/test_hard_pair.py::test_two_copy_adaptive_protocol_attains_ppt_bound[128]"
```

The deselected case builds a 2 GiB matrix; the same value is checked sparsely elsewhere. [`code/README.md`](code/README.md)
lists every script, what it checks in the paper, and how long it takes.

## Rebuilding the PDF

The source uses Springer Nature's journal template; its class file `sn-jnl.cls` is included in `paper/`. Build with
`pdflatex`, or for example with [Tectonic](https://tectonic-typesetting.github.io):

```bash
cd paper && tectonic paper.tex
```

## Citation

If you use this work, please cite:

```bibtex
@misc{messerlian2026rankone,
  author       = {Alexander Messerlian},
  title        = {A single-copy lower bound for rank-one nonlinear overlap estimation},
  year         = {2026},
  howpublished = {\url{https://github.com/alex-messerlian/rank-one-overlap-bound}},
  note         = {Manuscript}
}
```

GitHub's "Cite this repository" button gives the same reference, from [`CITATION.cff`](CITATION.cff).

## License

- **Code** (everything in `code/`, including the saved results): [MIT License](LICENSE).
- **Paper** (everything in `paper/`, including the figure): [CC BY 4.0](paper/LICENSE.md). The template class
  `sn-jnl.cls` is Springer Nature's and keeps its own license.

## Contact

Alexander Messerlian, alex.messerlian@icloud.com
