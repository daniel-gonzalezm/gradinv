"""Cubic algebra (class m-3m).

Embeddings, O_h invariance and commutant, multiplication table, change of
constants with Lazar et al. (2022, eq. 40), isotropy conditions (eq. 81),
eigenvalue formulas (eqs. 125, 128-140), Table 5, inverse and
positive definiteness by blocks on the data, separable limit, sign conventions.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from gradinv import cubic, isotropic
from gradinv._tensor import N, R, TRIPLES, index, kd, rotation_27, tensor6
from gradinv.skeleton import identity_V
from helpers import basis_V, commutant_dimension, is_zero

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LZ = DATA["datasets"]["lazar2022"]
SETS = {"Al": LZ["D_voigt"]["Al"], "Cu": LZ["D_voigt"]["Cu"], "Fe": LZ["D_voigt"]["Fe"],
        "W": LZ["D_voigt"]["W"], "Cu_EAM": DATA["datasets"]["po2019"]["D_voigt"]["Cu_EAM"]}

CUB = cubic.algebra()
U = CUB.units()
ISO_U = isotropic.algebra().units()
I = identity_V()
A = sp.symbols("a1:12")


def _emb_list(alg):
    return [(p, a, alg.embedding(p, a)) for p, copies, _d in alg.irreps for a in copies]


# ------------------------------------------------------------------ A embeddings
def test_embeddings_isometric_orthogonal_complete():
    Eall = CUB.embedding_matrix()
    assert Eall.shape == (27, 18)
    assert np.allclose(Eall.T @ Eall, np.eye(18), atol=1e-14)
    assert np.allclose(Eall @ Eall.T, I, atol=1e-14)


@pytest.mark.core
@pytest.mark.exact
def test_embeddings_gram_exact():
    """E_b^T E_c = d_bc Id within an irrep and 0 across irreps: this proves the table exactly."""
    Eall = cubic.algebra(exact=True).embedding_matrix()
    assert is_zero(Eall.T * Eall - sp.eye(18))


def test_copies_lie_in_their_pieces():
    P3, P2 = ISO_U[("H3", "3", "3")], ISO_U[("H2", "2", "2")]
    for p, a in (("T1", "h"), ("T2", "3"), ("A2", "a")):
        assert np.allclose(P3 @ CUB.embedding(p, a), CUB.embedding(p, a), atol=1e-14)
    for p, a in (("T2", "2"), ("E", "e")):
        assert np.allclose(P2 @ CUB.embedding(p, a), CUB.embedding(p, a), atol=1e-14)
    assert np.allclose(U[("T1", "h", "h")] + U[("T2", "3", "3")] + U[("A2", "a", "a")], P3, atol=1e-14)
    assert np.allclose(U[("T2", "2", "2")] + U[("E", "e", "e")], P2, atol=1e-14)
    for a, b in (("s", "s"), ("r", "r"), ("s", "r")):
        assert np.allclose(U[("T1", a, b)], ISO_U[("H1", a, b)], atol=1e-14)
    assert np.allclose(sum(U[(p, a, a)] for p, c, _d in cubic.IRREPS for a in c), I, atol=1e-14)


def test_a2_is_the_kernel_of_the_DD_contraction():
    DD = cubic.structure_tensor()

    def contract(col):
        h = col.reshape(3, 3, 3)
        return np.einsum("ijpq,pqm->ijm", DD, h).ravel()
    assert np.allclose(contract(CUB.embedding("A2", "a")[:, 0]), 0)
    for p, a in (("T1", "h"), ("T2", "3")):
        for c in range(3):
            assert not np.allclose(contract(CUB.embedding(p, a)[:, c]), 0)


# ------------------------------------------------------------------ B invariance, commutant
def test_oh_invariance_numeric():
    for Q in cubic.oh_group():
        Q3 = rotation_27(Q)
        for Uk in U.values():
            assert np.allclose(Q3 @ Uk @ Q3.T, Uk, atol=1e-13)


@pytest.mark.core
@pytest.mark.exact
def test_oh_invariance_exact():
    Ux = cubic.algebra(exact=True).units()
    for Q in cubic.oh_group(exact=True):
        Q3 = rotation_27(Q)
        for Uk in Ux.values():
            assert sp.Matrix(Q3 * Uk * Q3.T) == sp.Matrix(Uk)


def test_commutant_dimension_and_span():
    rot = [Q for Q in cubic.oh_group() if np.linalg.det(Q) > 0]
    assert commutant_dimension(rot) == 15
    assert commutant_dimension(rot, symmetric=True) == 11
    Bv = basis_V()
    Un = np.array([(Bv.T @ Uk @ Bv).ravel() for Uk in U.values()])
    assert np.linalg.matrix_rank(Un) == 15
    assert CUB.dimension == 15 and CUB.symmetric_dimension == 11


# ------------------------------------------------------------------ C table
@pytest.mark.core
def test_multiplication_table_all_225_products():
    for (p, a, b), X in U.items():
        for (q, c, d), Y in U.items():
            target = U[(p, a, d)] if (p == q and b == c) else 0.0
            assert np.allclose(X @ Y, target, atol=1e-14)
    keys = list(U)
    dims = dict((p, d) for p, _c, d in cubic.IRREPS)
    gram = np.array([[np.sum(U[k1] * U[k2]) for k2 in keys] for k1 in keys])
    assert np.allclose(gram, np.diag([dims[k[0]] for k in keys]), atol=1e-13)


@pytest.mark.exact
def test_multiplication_table_exact():
    Ux = cubic.algebra(exact=True).units()
    for (p, a, b), X in Ux.items():
        for (q, c, d), Y in Ux.items():
            target = Ux[(p, a, d)] if (p == q and b == c) else sp.zeros(N, N)
            assert is_zero(X * Y - target), ((p, a, b), (q, c, d))


# ------------------------------------------------------------------ D Lazar et al. (2022)
CLOSED_FORMS = {  # block moduli in Lazar's a1..a11
    "Mhh": 2 * A[3] + 2 * A[4] + R(12, 5) * (A[5] + A[6]) + R(3, 5) * A[7] + R(2, 5) * A[10],
    "Mss": R(10, 3) * (A[0] + A[1] + A[2]) + 2 * A[3] + 2 * A[4] + R(44, 15) * (A[5] + A[6])
    + R(11, 15) * A[7] + 2 * A[8] + 4 * A[9] + R(3, 5) * A[10],
    "Mrr": -R(4, 3) * A[0] + R(8, 3) * A[1] + R(2, 3) * A[2] + 2 * A[3] - A[4] + R(2, 3) * A[5]
    - R(4, 3) * A[6] + R(2, 3) * A[7],
    "Mhs": sp.sqrt(6) * (R(8, 15) * (A[5] + A[6]) + R(2, 15) * A[7] + R(1, 3) * A[8] + R(2, 3) * A[9]
                         + R(1, 5) * A[10]),
    "Mhr": sp.sqrt(30) / 15 * (2 * A[5] - A[6] - A[7] + 2 * A[8] - 2 * A[9]),
    "Msr": sp.sqrt(5) * (R(1, 3) * (A[0] + 4 * A[1] - 2 * A[2]) - R(4, 15) * A[5]
                         + R(2, 15) * (A[6] + A[7]) + R(2, 5) * (A[8] - A[9])),
    "N33": 2 * A[3] + 2 * A[4] + R(4, 3) * (A[5] + A[6]) + R(1, 3) * A[7],
    "N22": 2 * A[3] - A[4] + R(2, 3) * A[5] - R(4, 3) * A[6] + R(2, 3) * A[7],
    "N32": sp.sqrt(2) / 3 * (-2 * A[5] + A[6] + A[7]),
    "mE": 2 * A[3] - A[4], "mA": 2 * A[3] + 2 * A[4]}


@pytest.fixture(scope="module")
def exact_blocks():
    return cubic.algebra(exact=True).moduli(cubic.lazar2022_tensor(A, exact=True))


@pytest.mark.exact
def test_lazar2022_tensor_in_algebra_exact(exact_blocks):
    D = cubic.lazar2022_tensor(A, exact=True)
    Ix = identity_V(exact=True)
    assert is_zero(D - D.T) and is_zero(Ix * D * Ix - D)
    assert is_zero(cubic.algebra(exact=True).assemble(exact_blocks) - D)
    assert all(is_zero(B - B.T) for B in exact_blocks.values())


@pytest.mark.core
@pytest.mark.exact
def test_lazar2022_change_of_constants_exact(exact_blocks):
    v = cubic.blocks_to_vector(exact_blocks)
    for (name, *_), x in zip(cubic.MODULI, v):
        assert is_zero(x - CLOSED_FORMS[name]), name
    J = sp.Matrix(v).jacobian(A)
    assert sp.nsimplify(sp.simplify(J.det())) == -384 * sp.sqrt(2)
    names = sp.symbols([n for n, *_ in cubic.MODULI])
    inv = sp.solve([x - m for x, m in zip(v, names)], A, dict=True)[0]
    Mhh, N33, mA, mE = names[0], names[6], names[10], names[9]
    assert is_zero(inv[A[3]] - (mA / 6 + mE / 3)) and is_zero(inv[A[4]] - (mA - mE) / 3)
    assert is_zero(inv[A[10]] - (R(5, 2) * Mhh - R(9, 2) * N33 + 2 * mA))
    assert is_zero(sp.Matrix(cubic.to_lazar2022(exact_blocks, exact=True)) - sp.Matrix(A))


def test_lazar2022_change_of_constants_numeric():
    rng = np.random.default_rng(2)
    a = rng.normal(size=11)
    blk = cubic.from_lazar2022(a)
    f = sp.lambdify(A, [CLOSED_FORMS[n] for n, *_ in cubic.MODULI])
    assert np.allclose(cubic.blocks_to_vector(blk), f(*a))
    direct = CUB.moduli(cubic.lazar2022_tensor(a))
    assert all(np.allclose(direct[p], blk[p]) for p in blk)
    assert np.allclose(CUB.residual(cubic.lazar2022_tensor(a)), 0, atol=1e-13)
    assert np.allclose(cubic.to_lazar2022(blk), a)


# ------------------------------------------------------------------ E isotropy
@pytest.mark.exact
def test_isotropy_conditions_eq81(exact_blocks):
    B = exact_blocks
    iso = {k: 0 for k in A[5:]}
    Bi = {p: B[p].subs(iso) for p in B}
    m3, m2 = 2 * (A[3] + A[4]), 2 * A[3] - A[4]
    for x in (Bi["T1"][0, 0], Bi["T2"][0, 0], Bi["A2"][0, 0]):
        assert is_zero(x - m3)
    for x in (Bi["T2"][1, 1], Bi["E"][0, 0]):
        assert is_zero(x - m2)
    for x in (Bi["T1"][0, 1], Bi["T1"][0, 2], Bi["T2"][0, 1]):
        assert is_zero(x)
    conds = [B["T1"][0, 1], B["T1"][0, 2], B["T2"][0, 1], B["T1"][0, 0] - B["A2"][0, 0],
             B["T2"][0, 0] - B["A2"][0, 0], B["T2"][1, 1] - B["E"][0, 0]]
    Jc = sp.Matrix(conds).jacobian(A)
    assert is_zero(Jc[:, :5]) and Jc[:, 5:].rank() == 6
    # isotropic limit of the (s, r) sub-block = the isotropic H1 block
    assert is_zero(Bi["T1"][1, 1] - (R(10, 3) * (A[0] + A[1] + A[2]) + 2 * A[3] + 2 * A[4]))
    assert is_zero(Bi["T1"][1, 2] - sp.sqrt(5) / 3 * (A[0] + 4 * A[1] - 2 * A[2]))


# ------------------------------------------------------------------ F eqs. (125), (128)-(140)
@pytest.mark.exact
def test_lazar2022_eigenvalue_formulas(exact_blocks):
    a = A
    s2 = sp.sqrt(2)
    d = {'11': 2 * sum(a[:5]) + 4 * (a[5] + a[6] + a[9]) + a[7] + 2 * a[8] + a[10],
         '12': a[0] + 2 * a[1] + a[8], '13': (a[0] + 2 * a[2] + 2 * a[9]) / s2,
         '22': 2 * (a[1] + a[3]) + a[7], '23': (a[0] + 2 * a[4] + 2 * a[6]) / s2, '24': 2 * a[1],
         '25': a[0] / s2, '33': a[2] + 2 * a[3] + a[4] + 2 * a[5], '35': a[2],
         '1616': 2 * a[3], '1617': a[4]}
    lam1 = d['1616'] - d['1617']
    lam2 = 2 * d['1617'] + d['1616']
    k1 = d['22'] - d['24'] + d['33'] - d['35']
    k3 = (-d['23']**2 + 2 * d['23'] * d['25'] - d['25']**2 + d['22'] * d['33'] - d['24'] * d['33']
          - d['22'] * d['35'] + d['24'] * d['35'])
    k4_tail = (-4 * d['12'] * d['13'] * d['23'] + d['11'] * d['23']**2
               - 4 * d['12'] * d['13'] * d['25'] + 2 * d['11'] * d['23'] * d['25'] + d['11'] * d['25']**2
               + 2 * d['12']**2 * d['33'] - d['11'] * d['22'] * d['33'] - d['11'] * d['24'] * d['33']
               + 2 * d['12']**2 * d['35'] - d['11'] * d['22'] * d['35'] - d['11'] * d['24'] * d['35'])
    Dt = sp.Symbol("Dtilde")  # the stray factor printed in eq. (135)
    k4_printed = 2 * d['13']**2 * d['22'] + Dt * d['13']**2 * d['24'] + k4_tail
    k4_fixed = 2 * d['13']**2 * d['22'] + 2 * d['13']**2 * d['24'] + k4_tail
    k5 = (-2 * d['12']**2 - 2 * d['13']**2 + d['11'] * d['22'] - d['23']**2 + d['11'] * d['24']
          - 2 * d['23'] * d['25'] - d['25']**2 + d['11'] * d['33'] + d['22'] * d['33'] + d['24'] * d['33']
          + d['11'] * d['35'] + d['22'] * d['35'] + d['24'] * d['35'])
    k6 = -(d['11'] + d['22'] + d['24'] + d['33'] + d['35'])
    B = exact_blocks
    T1, T2 = B["T1"], B["T2"]
    minors = sum(T1.extract([p, q], [p, q]).det() for p, q in ((0, 1), (0, 2), (1, 2)))
    assert is_zero(B["E"][0, 0] - lam1) and is_zero(B["A2"][0, 0] - lam2)
    assert is_zero(T2.trace() - k1) and is_zero(T2.det() - k3)
    assert is_zero(-T1.trace() - k6) and is_zero(minors - k5)
    assert is_zero(-T1.det() - k4_fixed)                                  # eq. (125) corrected
    diff = sp.expand(-T1.det() - k4_printed)
    assert is_zero(diff + (Dt - 2) * d['13']**2 * d['24'])                # printed fails unless Dtilde = 2


# ------------------------------------------------------------------ G Table 5
TABLE5 = {  # Lazar et al. (2022), Table 5, p. 128 (rendered); lambda1..lambda7
    'Al': [0.03560, 0.93596, 0.87576, 0.04980, 1.70599, 1.06408, 0.17883],
    'Cu': [0.00005, 0.22441, 0.83490, 0.04664, 1.70391, 0.07599, 0.04664],
    'Fe': [0.06598, 2.36399, 0.65338, -0.48601, 2.47144, 1.03077, 0.77479],
    'W': [0.48426, 4.18264, 1.94819, -0.94259, 3.90703, 3.18388, 0.85833]}


def _table5_match(a, ref, tol=2e-5):
    blk = cubic.from_lazar2022(a)
    t2 = np.linalg.eigvalsh(blk["T2"])
    t1 = np.linalg.eigvalsh(blk["T1"])
    return (abs(blk["E"][0, 0] - ref[0]) < tol and abs(blk["A2"][0, 0] - ref[1]) < tol
            and np.allclose(sorted(t2), sorted(ref[2:4]), atol=tol)
            and np.allclose(sorted(t1), sorted(ref[4:7]), atol=tol))


@pytest.mark.parametrize("mat", ["Al", "Cu", "Fe", "W"])
def test_table5_reproduced(mat):
    ref = list(TABLE5[mat])
    if mat == "Cu":
        ref[6] = 0.62864                                         # misprint of Table 5
    assert _table5_match(LZ["a_table2"][mat], ref)


def test_table5_cu_lambda7_misprint():
    assert not _table5_match(LZ["a_table2"]["Cu"], TABLE5["Cu"])


# ------------------------------------------------------------------ H inverse and conditions
@pytest.mark.core
@pytest.mark.exact
def test_inverse_of_general_element_exact():
    """S = sum of the inverse blocks is the inverse of a general element (15 symbols)."""
    alg = cubic.algebra(exact=True)
    syms = {"T1": sp.Matrix(3, 3, sp.symbols("x0:9")), "T2": sp.Matrix(2, 2, sp.symbols("y0:4")),
            "E": sp.Matrix([[sp.Symbol("e")]]), "A2": sp.Matrix([[sp.Symbol("q")]])}
    Dg, Sg = alg.assemble(syms), alg.assemble(alg.inverse(syms))
    assert is_zero(Sg * Dg - identity_V(exact=True))


# data blocks (eV/A; input by eq. 60)
NOTES_BLOCKS = {
    "Al": [0.61241, 1.38931, 0.94719, 0.00711, -0.52133, 0.40536, 0.79951, 0.12604, -0.23907, 0.0356, 0.93596],
    "Cu": [0.17215, 1.46428, 0.77209, -0.30338, -0.21290, 0.36031, 0.83180, 0.04975, -0.04937, 0.00006, 0.22440],
    "Cu_EAM": [0.45265, 2.16405, 0.23777, -0.36874, 0.02610, -0.05317, 0.78814, 0.18078, -0.03370, 0.08546,
               0.72458],
    "Fe": [1.09196, 2.12667, 1.05838, -0.65172, -0.06708, 0.19633, -0.23761, 0.40497, -0.47046, 0.06598, 2.36398],
    "W": [2.53383, 3.83628, 1.57912, -0.27079, -1.08517, -0.05427, -0.02180, 1.02741, -1.34683, 0.48426, 4.18266]}


@pytest.mark.core
@pytest.mark.parametrize("mat", list(SETS))
def test_data_blocks_inverse_and_positive_definiteness(mat):
    a = cubic.a_from_voigt(SETS[mat])
    D = cubic.lazar2022_tensor(a)
    blk = CUB.moduli(D)
    assert np.allclose(cubic.blocks_to_vector(blk), NOTES_BLOCKS[mat], atol=6e-5)
    Sn = CUB.invert(D)
    assert np.abs(Sn @ D - I).max() < 1e-12
    Bv = basis_V()
    Sref = Bv @ np.linalg.inv(Bv.T @ D @ Bv) @ Bv.T
    assert np.abs(Sn - Sref).max() / np.abs(Sref).max() < 1e-12
    pd_full = np.linalg.eigvalsh(Bv.T @ D @ Bv).min() > 0
    assert CUB.is_positive_definite(blk) == pd_full
    expected_bad = ["T2"] if mat in ("Fe", "W") else []
    assert CUB.non_positive_blocks(blk) == expected_bad
    if mat in ("Fe", "W"):
        assert blk["T2"][0, 0] < 0                                # N33 negative


# ------------------------------------------------------------------ I separable and limits
def _cubic_C(C11, C12, C44, exact):
    f = lambda i, j, k, l: (C12 * kd(i, j) * kd(k, l) + C44 * (kd(i, k) * kd(j, l) + kd(i, l) * kd(j, k))
                            + (C11 - C12 - 2 * C44) * (1 if i == j == k == l else 0))
    return f


@pytest.mark.exact
def test_separable_inverse_exact():
    """C x ell^2 d with C cubic: closed-form inverse = C^-1 x ell^-2 d."""
    vals = (R(7, 4), R(3, 4), R(1, 2))
    C = _cubic_C(*vals, exact=True)
    C9 = sp.Matrix(9, 9, lambda r, c: C(r // 3, r % 3, c // 3, c % 3))
    Iv9 = sp.Matrix(9, 9, lambda r, c: R(1, 2) * (kd(r // 3, c // 3) * kd(r % 3, c % 3)
                                                 + kd(r // 3, c % 3) * kd(r % 3, c // 3)))
    S9 = (C9 + (sp.eye(9) - Iv9)).inv() - (sp.eye(9) - Iv9)
    ell2 = R(3, 2)
    D = tensor6(lambda i, j, m, k, l, n: C9[3 * i + j, 3 * k + l] * ell2 * kd(m, n))
    Sexp = tensor6(lambda i, j, m, k, l, n: S9[3 * i + j, 3 * k + l] / ell2 * kd(m, n))
    alg = cubic.algebra(exact=True)
    assert is_zero(alg.assemble(alg.inverse(alg.moduli(D))) - Sexp)


def test_separable_inverse_numeric():
    C = _cubic_C(1.75, 0.75, 0.5, exact=False)
    C9 = np.array([[C(r // 3, r % 3, c // 3, c % 3) for c in range(9)] for r in range(9)])
    Iv9 = np.array([[0.5 * (kd(r // 3, c // 3) * kd(r % 3, c % 3) + kd(r // 3, c % 3) * kd(r % 3, c // 3))
                     for c in range(9)] for r in range(9)])
    S9 = np.linalg.pinv(C9) @ Iv9
    ell2 = 1.5
    D = np.einsum("ab,mn->ambn", C9 * ell2, np.eye(3)).reshape(27, 27)
    Sexp = np.einsum("ab,mn->ambn", S9 / ell2, np.eye(3)).reshape(27, 27)
    assert np.allclose(CUB.residual(D), 0, atol=1e-13)
    assert np.allclose(CUB.invert(D), Sexp, atol=1e-12)


# ------------------------------------------------------------------ K sign conventions (S3)
def test_sign_conventions_of_the_H3_copies():
    """Blocks of the a8 tensor DD x delta."""
    blk = CUB.moduli(cubic.lazar2022_tensor([0] * 7 + [1, 0, 0, 0]))
    s6, s30, s5 = np.sqrt(6), np.sqrt(30), np.sqrt(5)
    T1 = np.array([[3 / 5, 2 * s6 / 15, -s30 / 15], [2 * s6 / 15, 11 / 15, 2 * s5 / 15],
                   [-s30 / 15, 2 * s5 / 15, 2 / 3]])
    T2 = np.array([[1 / 3, np.sqrt(2) / 3], [np.sqrt(2) / 3, 2 / 3]])
    assert np.allclose(blk["T1"], T1) and np.allclose(blk["T2"], T2)
    assert np.allclose(blk["E"], 0) and np.allclose(blk["A2"], 0)
    assert blk["T1"][0, 1] > 0 and blk["T2"][0, 1] > 0
