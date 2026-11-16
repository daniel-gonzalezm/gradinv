# Tests

These tests check every result of the paper *Algebra, closed-form compliance and
orientation averages of cubic and hexagonal strain-gradient elasticity tensors*. This
page says what they do, in plain words, and where to start if you want to audit them.

## Where to start: the core tests

Forty-two tests are marked `core`. Together they check every claim of the paper and the
illustrative application, each against something that does not depend on the formula
being checked. Run them with

```
pytest -m core          # about a minute and a half; exact and slow ones included
```

The supplementary material (section S2, "Audit map") lists them claim by claim, with
what each compares against and its level. The same map is the file
`analysis/audit_map.py`; `tests/test_audit_map.py` fails if the map and the `core`
marks ever disagree.

## How to run the rest

```
pytest                  # fast numerical tests (under half a minute)
pytest --exact          # adds the symbolic tests (exact arithmetic with sympy)
pytest --all            # everything, symbolic and slow tests included (a few minutes)
```

## What the comparisons are

Every check has one of three levels:

- **Verification (V).** The result is compared with exact mathematics (computed with
  rational and algebraic numbers, not floating point) or with an independent route of
  our own. The usual independent routes are: brute force (for example, evaluating all
  225 or 1024 products of a multiplication table in components); the plain
  18 × 18 matrix of the tensor, inverted or diagonalised directly instead of block by
  block; and random rotations (Monte Carlo) instead of a closed average.
- **Cross-comparison (CC).** The result is compared with a published one, after
  rebuilding that result from the paper that gives it. When a printed formula turns out
  not to hold, the test checks both the printed form (and that it fails) and the
  corrected form (and that it holds).
- **Consistency (C).** Limit cases (isotropic or separable tensors, an isotropic grain),
  invariances and inequalities checked on random samples.

No test compares with measurements: the inputs are published atomistic constants and,
for the hexagonal class, the granular micromechanics of Pirmoradi et al. (2024).

## The four families

**1. Construction** – the algebras and what follows directly from them: bases,
multiplication tables, invariance, inverses, the hexagonal extra modulus, the changes of
basis to the constants of other authors.
`test_skeleton.py`, `test_cubic.py`, `test_hexagonal.py`,
`test_hexagonal_extra_modulus.py`, `test_odd_inverse.py`, `test_forms.py`,
`test_index_forms.py`, `test_scope.py` (fifth-order coupling and site symmetries, for
the scope stated in sections 2 and 8), `test_package.py`.

**2. Sources** – published results and data: isotropic results of other authors
rebuilt, the transcription of the
atomistic data (including the pair-functional relations that the Cu EAM set obeys),
the provenance of every input number, Kröner's estimate against two published sources.
`test_isotropic_sources.py`, `test_source_data.py`,
`test_provenance.py`, `test_kroner_sources.py`.

**3. Averages and application** – Voigt, Reuss and Hill estimates, block lengths, the
neighbourhood radius, the concentration tensor, the averaged Eshelby tensor of a sphere
(by two independent routes), the self-consistent scheme and its large-grain limit.
`test_averages.py`, `test_main_results.py`, `test_neighborhood.py`,
`test_concentrator.py`, `test_eshelby.py`, `test_eshelby_real_space.py`,
`test_selfconsistent.py`, `test_application.py`, `test_fe_case.py`,
`test_induced_moduli.py` (the moduli induced by the heterogeneity of the classical
stiffness, after Malyi 2021, against the Voigt estimate).

**4. Manuscript chain** – the link between the code and the paper: every computed
number in the text is a LaTeX macro written by a script, every table and figure is
regenerated and compared with the file the paper uses, the numbers behind the claims are
recomputed by a separate route, figures shown with no more digits than the rounding of
the data supports, and the formulas printed in the paper are parsed from its source and
checked against the code.
`test_paper_numbers.py`, `test_paper_formulas.py`, `test_claim_figures.py`,
`test_tables.py`, `test_figures.py`, `test_manuscript.py`, `test_audit_map.py`.

Shared helpers are in `conftest.py`, `helpers.py` and `references/` (published
formulas and values rebuilt for the cross-comparisons; the values themselves are read
from `analysis/data/`).
