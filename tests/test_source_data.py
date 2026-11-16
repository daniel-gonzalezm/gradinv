"""Transcription control of the cubic source data.

Checks only the internal consistency of the transcribed tables; it does not
test any method of the paper.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

DATA = json.loads((Path(__file__).resolve().parents[1] / "analysis" / "data"
                   / "cubic_gradient_constants.json").read_text())
LAZAR = DATA["datasets"]["lazar2022"]
PO = DATA["datasets"]["po2019"]


def a_from_voigt(v):
    """Eq. (60) of Lazar, Agiasofitou & Boehlke (2022)."""
    D11, D12, D13, D22, D23, D24, D25, D33, D35, D1616, D1617 = v
    return np.array([
        2 * D25, D24 / 2, 2 * D35, D1616, 2 * D1617,
        D33 - D35 - D1616 - D1617, D23 - D25 - 2 * D1617,
        D22 - D24 - 2 * D1616, D12 - D24 - 2 * D25, D13 - D25 - 2 * D35,
        D11 - D22 + 2 * (D24 - D12) + 4 * (D1616 - D13 - D23 - D33)
        + 8 * (D25 + D35 + D1617)])


@pytest.mark.core
@pytest.mark.parametrize("mat", ["Al", "Cu", "Fe", "W"])
def test_table2_is_eq60_of_table1(mat):
    # rounding of Table 1 to 5 decimals propagates with weights up to ~46 into a11
    a = a_from_voigt(LAZAR["D_voigt"][mat])
    assert np.allclose(a, LAZAR["a_table2"][mat], atol=3e-4)


@pytest.mark.parametrize("po_key, mat", [("Cu_MEAM", "Cu"), ("Al_MEAM", "Al")])
def test_po2019_meam_equals_lazar2022_table1(po_key, mat):
    assert np.allclose(PO["D_voigt"][po_key], LAZAR["D_voigt"][mat], atol=2e-5)
    assert np.allclose(PO["C"][po_key], LAZAR["C"][mat], atol=1e-4)


def test_po2019_cu_eam_is_a_distinct_dataset():
    assert not np.allclose(PO["D_voigt"]["Cu_EAM"], LAZAR["D_voigt"]["Cu"], atol=1e-2)


# ------------------------------------------------------------------ Lazar and Po (2024)
LP = DATA["datasets"]["lazarpo2024"]
MATS = ["Al", "Cu", "Fe", "W"]


def compliance(C11, C12, C44, delta=None, dd=None):
    """S_ijkl of Lazar and Po (2024), eqs. (97)-(100), as a 3x3x3x3 array (or nested list)."""
    delta = delta or (lambda i, j: 1.0 if i == j else 0.0)
    dd = dd or (lambda *ix: 1.0 if all(x == ix[0] for x in ix) else 0.0)
    S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
    S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12))
    S44 = 1 / (4 * C44)
    R3 = range(3)
    return [[[[S12 * delta(i, j) * delta(k, l) + S44 * (delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k))
               + (S11 - S12 - 2 * S44) * dd(i, j, k, l) for l in R3] for k in R3] for j in R3] for i in R3]


def ell2_eq107(S, D):
    """Eq. (107): ell^2 = S_ijkl D_ijmklm / 18, with D stored 27 x 27 (rows (i,j,m), cols (k,l,n))."""
    R3 = range(3)
    return sum(S[i][j][k][l] * D[9 * i + 3 * j + m, 9 * k + 3 * l + m]
               for i in R3 for j in R3 for k in R3 for l in R3 for m in R3) / 18


def ell2_eq108(C, a):
    C11, C12, C44 = C
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = a
    return (3 / C44 * (a3 + 3 * a4 + a5 + 2 * a6) + 6 / (C11 + 2 * C12) * (a1 + 3 * a2 + a9)
            + 3 * (C11 + C12) / ((C11 + 2 * C12) * (C11 - C12))
            * (2 * a3 + 6 * a4 + 2 * a5 + 4 * a6 + 4 * a7 + 3 * a8 + 4 * a10 + a11))


def ell_eq107(mat):
    from gradinv import cubic
    return np.sqrt(ell2_eq107(compliance(*LP["C"][mat]), cubic.lazar2022_tensor(LP["a_table1"][mat])))


@pytest.mark.parametrize("mat", MATS)
def test_lazarpo2024_table1_repeats_lazar2022(mat):
    assert LP["C"][mat] == LAZAR["C"][mat] and LP["a_table1"][mat] == LAZAR["a_table2"][mat]


@pytest.mark.core
@pytest.mark.parametrize("mat", MATS)
def test_lazarpo2024_compliance_eqs_97_102(mat):
    C11, C12, C44 = LP["C"][mat]
    d = np.eye(3)
    DD = np.zeros((3, 3, 3, 3))
    for s in range(3):
        DD[s, s, s, s] = 1
    C = (C12 * np.einsum("ij,kl->ijkl", d, d) + C44 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
         + (C11 - C12 - 2 * C44) * DD)
    S = np.array(compliance(C11, C12, C44))
    Isym = (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)) / 2
    assert np.allclose(np.einsum("ijmn,mnkl->ijkl", C, S), Isym, atol=1e-13)
    assert np.isclose(np.einsum("ijkl,ijkl", C, S), 6.0)


@pytest.mark.parametrize("mat", ["Al", "Cu"])
def test_ell_eq107_reproduces_table2_for_al_cu(mat):
    assert round(ell_eq107(mat), 4) == LP["ell_table2"][mat]


def test_ell_values_eq107():
    """Recomputed ell (A): Al 1.17737, Cu 0.78523, Fe 0.74288, W 0.85086."""
    got = [round(ell_eq107(m), 5) for m in MATS]
    assert got == [1.17737, 0.78523, 0.74288, 0.85086]


@pytest.mark.parametrize("mat", ["Fe", "W"])
def test_lazar_po_table2_ell_of_fe_and_w_is_a_slip(mat):
    ell = ell_eq107(mat)
    assert round(ell, 4) != LP["ell_table2"][mat]
    assert round(ell / LP["lattice_a_table2"][mat], 4) == LP["ell_over_a_table2"][mat]
    assert round(LP["ell_table2"][mat] / LP["lattice_a_table2"][mat], 4) != LP["ell_over_a_table2"][mat]


@pytest.mark.parametrize("mat", ["Al", "Cu"])
def test_ell_over_a_consistent_for_al_cu(mat):
    assert round(ell_eq107(mat) / LP["lattice_a_table2"][mat], 4) == LP["ell_over_a_table2"][mat]


@pytest.mark.parametrize("mat", MATS)
def test_lazar_po_eq108_is_18_times_eq107(mat):
    assert np.isclose(ell2_eq108(LP["C"][mat], LP["a_table1"][mat]), 18 * ell_eq107(mat) ** 2, rtol=1e-12)


@pytest.mark.exact
def test_lazar_po_eq108_exact_and_separable_limit_eqs_109_110():
    import sympy as sp
    from gradinv import cubic
    C11, C12, C44, ell = sp.symbols("C11 C12 C44 ell", positive=True)
    a = sp.symbols("a1:12", real=True)
    delta = lambda i, j: sp.Integer(1) if i == j else sp.Integer(0)  # noqa: E731
    dd = lambda *ix: sp.Integer(1) if all(x == ix[0] for x in ix) else sp.Integer(0)  # noqa: E731
    S = compliance(C11, C12, C44, delta, dd)
    e107 = ell2_eq107(S, cubic.lazar2022_tensor(a, exact=True))
    assert sp.simplify(ell2_eq108((C11, C12, C44), a) - 18 * e107) == 0
    a_sep = (0, C12 * ell ** 2 / 2, 0, C44 * ell ** 2, 0, 0, 0, (C11 - C12 - 2 * C44) * ell ** 2, 0, 0, 0)  # eq. (109)
    assert sp.simplify(ell2_eq107(S, cubic.lazar2022_tensor(a_sep, exact=True)) - ell ** 2) == 0


# ------------------------------------------------------------------ pair functionals
# Consistency check of the Cu EAM data set (it is a pair functional, P04-P05); not in
# the manuscript. For a pair functional on a centrosymmetric
# simple lattice, Admal, Marian and Po (2017, eqs. 4.5e and 4.7) give
#   D = sum_beta a(R) R^(x6) + sum_beta b(R) G^beta : G^beta,
#   G_PIJM = (d_PI R_J R_M + d_PJ R_I R_M - d_PM R_I R_J) / 2,
# so the cubic D lives in a subspace of dimension 5 of the 11 constants of Lazar et
# al. (2022). The subspace is built here by lattice sums over fcc neighbour shells
# with random shell weights, without assuming its form.
PAIR_RELATIONS = np.array([   # rows c with c . (a1..a11) = 0
    [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],                 # a1 = a5
    [0, 1, 0, -0.25, -0.125, 0, 0, 0, 0, 0, 0],         # a2 = a4/4 + a5/8
    [0, 0, 1, -1, -0.5, 0, 0, 0, 0, 0, 0],              # a3 = a4 + a5/2
    [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0],                 # a6 = a8
    [0, 0, 0, 0, 0, 0, 1, 1, 0, -2, 0],                 # a7 + a8 = 2 a10
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0]])                # a9 = a10


def _fcc_shells(n_shells=4):
    pts = [np.array(p, float) for p in itertools.product(range(-3, 4), repeat=3)
           if sum(p) % 2 == 0 and any(p)]
    radii = sorted({round(float(p @ p), 9) for p in pts})[:n_shells]
    return [[p for p in pts if round(float(p @ p), 9) == r] for r in radii]


def _pair_functional_tensor(weights_a, weights_b):
    d = np.eye(3)
    D = np.zeros((3,) * 6)
    for shell, wa, wb in zip(_fcc_shells(len(weights_a)), weights_a, weights_b):
        for R in shell:
            D += wa * np.einsum("i,j,m,k,l,n->ijmkln", R, R, R, R, R, R)
            G = 0.5 * (np.einsum("pi,j,m->pijm", d, R, R) + np.einsum("pj,i,m->pijm", d, R, R)
                       - np.einsum("pm,i,j->pijm", d, R, R))
            D += wb * np.einsum("pijm,pkln->ijmkln", G, G)
    return D.reshape(27, 27)


def test_pair_functional_subspace_is_the_null_space_of_six_relations():
    from gradinv import cubic
    rng = np.random.default_rng(65)
    alg = cubic.algebra()
    samples = []
    for _ in range(12):
        D = _pair_functional_tensor(rng.normal(size=4), rng.normal(size=4))
        a = cubic.to_lazar2022(alg.moduli(D))
        assert np.allclose(cubic.lazar2022_tensor(a), D, atol=1e-9 * np.abs(D).max())  # cubic
        samples.append(a)
    samples = np.array(samples)
    assert np.linalg.matrix_rank(samples, tol=1e-8 * np.abs(samples).max()) == 5
    assert np.linalg.matrix_rank(PAIR_RELATIONS) == 6
    assert np.allclose(samples @ PAIR_RELATIONS.T, 0, atol=1e-9 * np.abs(samples).max())


def _rounding_half_units(v, sig=5):
    """Half a unit of the fifth significant figure (Po et al. 2019, Table 1)."""
    v = np.asarray(v, float)
    return 0.5 * 10.0 ** (np.floor(np.log10(np.abs(v))) - (sig - 1))


def test_cu_eam_obeys_the_six_relations_within_rounding_and_meam_sets_do_not():
    from gradinv import cubic
    # a = A_60 D (eq. 60 is linear): sensitivity of each relation to every D component
    A60 = np.array([cubic.a_from_voigt(e) for e in np.eye(11)]).T
    v = np.array(PO["D_voigt"]["Cu_EAM"])
    residual = PAIR_RELATIONS @ cubic.a_from_voigt(v)
    bound = np.abs(PAIR_RELATIONS @ A60) @ _rounding_half_units(v)
    assert np.all(np.abs(residual) <= bound), (residual, bound)
    for key, vm in [("Al", LAZAR["D_voigt"]["Al"]), ("Cu", LAZAR["D_voigt"]["Cu"]),
                    ("Fe", LAZAR["D_voigt"]["Fe"]), ("W", LAZAR["D_voigt"]["W"])]:
        am = cubic.a_from_voigt(vm)
        rel = np.linalg.norm(PAIR_RELATIONS @ am) / np.linalg.norm(am)
        assert rel > 0.05, key
