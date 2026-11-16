# gradinv

Code and data for the article *Algebra, closed-form compliance and orientation
averages of cubic and hexagonal strain-gradient elasticity tensors*
(D. González-Monsalve).

The package `gradinv` builds coordinate-free bases, with closed multiplication
tables, of the algebras of sixth-order tensors of Mindlin's strain-gradient
elasticity (form II) for the cubic class m-3m and the hexagonal class 6/mmm. With
them, products, inverses, orientation averages and concentration tensors reduce to
operations on small blocks. The repository also contains every script that writes
the tables, figures and computed numbers of the article and its supplementary
material, the input data with the provenance of every number, and the test suite.

## Installation

Python 3.10 or later.

```
pip install -e ".[test]"        # numpy, scipy, sympy and pytest
pip install matplotlib          # only to redraw the figures
```

## Usage

```python
from gradinv import cubic, averages

a = cubic.a_from_voigt(D_voigt)        # constants a1..a11 (Lazar et al. 2022, eq. 60)
D = cubic.lazar2022_tensor(a)          # 27 x 27 storage of D_ijmkln
alg = cubic.algebra()                  # block algebra of the cubic class (dimension 15)
blocks = alg.moduli(D)                 # {'T1': 3x3, 'T2': 2x2, 'E': 1x1, 'A2': 1x1}
alg.is_positive_definite(blocks)       # block-wise test
S = alg.assemble(alg.inverse(blocks))  # gradient compliance on V

V = averages.voigt(alg, blocks)        # isotropic blocks {'H3', 'H2', 'H1'}
R = averages.reuss(alg, blocks)        # refuses a D that is not positive definite
```

`gradinv.isotropic` and `gradinv.hexagonal` have the same interface
(`hexagonal.extra_modulus(blocks)` is the modulus absent from transverse isotropy).
Every builder takes `exact=True` to work in sympy instead of floating point. The
docstrings of the modules state their conventions: form II, eta_ijk = d_k eps_ij;
D_ijmkln stored as a 27 x 27 array with rows (i, j, m) and columns (k, l, n), so
that the triple contraction is the matrix product.

## Reproducing the article

Every table, figure and computed number of the article is generated; none is
edited by hand.

```
python analysis/make_numbers.py      # paper/paper_numbers.tex (every computed number)
python analysis/make_tables.py       # analysis/tables/*.tex
python analysis/make_index_forms.py  # analysis/tables/index_forms.tex (supplement, S1)
python analysis/audit_map.py         # analysis/tables/audit_map.tex (supplement, S2)
python analysis/make_figures.py      # analysis/figures/*.pdf, *.png
```

The generators read the results stored in `analysis/results/`, written by
`analysis/cubic_results.py`, `analysis/hexagonal_granular.py`,
`analysis/selfconsistent_fcc.py` (the self-consistent scheme against grain size;
long minimizations) and `analysis/fe_case.py` (Solyaev's Fe example; a few
minutes). The tests check the stored results against these scripts.

The journal class is not stored here. To compile the manuscript and the
supplement, pass the folder of Elsevier's CAS LaTeX templates (`els-cas-templates`, with
`cas-sc.cls` and `cas-common.sty`). The class sets the text in STIX and the tables in
Computer Modern sans, so TeX Live needs the `stix` and `cm-super` fonts; without them the
PDF falls back to bitmap fonts, which the build reports:

```
python paper/build.py --cls <template folder> --check     # -> paper/build/
python paper/build.py --cls <template folder> --bundle    # self-contained sources
```

`--bundle` writes the sources of the submitted version to `paper/build/bundle/`
(one file per document, with the tables and the bibliography inlined), checks that
they compile without the repository to the same PDFs, and packs them.

## Input data

`analysis/data/cubic_gradient_constants.json` holds the published elastic and
gradient-elastic constants used as input (Lazar et al. 2022; Po et al. 2019), and
`analysis/data/published_reference_values.json` the published values used only for
comparison. `analysis/data/provenance.csv` gives, for every input number, its source
with table and page, its kind, units, check and rule of use. With the source PDFs
at hand (not distributed here), `python analysis/check_transcriptions.py <folder>`
compares every transcribed digit with the printed tables.

## Tests

```
pytest -m core    # the 42 key checks listed in the supplement (section S2), about 1.5 min
pytest            # fast numerical tests, under half a minute
pytest --all      # everything, symbolic and slow tests included, about 3 min
```

`tests/README.md` explains what the tests compare against and how they are grouped.

## Layout

```
src/gradinv/   the package
analysis/      data, results, and the scripts that write tables, figures and numbers
tests/         test suite
paper/         manuscript and supplement (LaTeX), bibliography, highlights, build script
```

## Licence

The code is released under the MIT licence (`LICENSE`). The data, results, tables
and figures in `analysis/`, and the text of the manuscript in `paper/`, are released
under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).

## Citation

Please cite the article. `CITATION.cff` gives the citation of this software.
