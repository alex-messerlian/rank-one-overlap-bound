# A single-copy lower bound for rank-one nonlinear overlap estimation

Alexander Messerlian

**Paper:** [`paper/paper.pdf`](paper/paper.pdf)

Randomized single-copy measurements estimate Tr(Oρ²) from O(√d) copies of a d-dimensional state, and this is optimal
for observables with large trace norm. This paper shows that low rank does not remove the dimension dependence. For a
fixed rank-one projector P, any adaptive single-copy protocol that estimates Tr(Pρ²) to error ε < 1/8 with success
probability at least 2/3, for every state, needs more than (d−1)^(1/3) copies when d ≥ 32. For two copies, the largest
difference in acceptance probability for the hard pair is exactly 1/(4√(d−1)).

## Contents

| Folder | What is in it |
|---|---|
| `paper/` | `paper.pdf`, its LaTeX source `paper.tex` (the references are inside it), `figures/` and the Springer Nature template class `sn-jnl.cls` |
| `code/` | Scripts and saved results that check the identities, numbers and figure in the paper; see [`code/README.md`](code/README.md) |

## Rebuilding the PDF

The source uses Springer Nature's journal template; its class file `sn-jnl.cls` is included in `paper/`. Build with
`pdflatex` or, for example, with [Tectonic](https://tectonic-typesetting.github.io):

```bash
cd paper && tectonic paper.tex
```

## License

- **Code** (everything in `code/`, including the saved results): [MIT License](LICENSE).
- **Paper** (everything in `paper/`, including the figure): [CC BY 4.0](paper/LICENSE.md).

## Contact

Alexander Messerlian, alex.messerlian@icloud.com
