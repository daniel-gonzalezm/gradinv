"""Orientation averages in block moduli.

The cubic block moduli from a1..a11 (moduli of the tensors of eq. 40 of Lazar et
al. 2022) are tested in test_cubic.py::test_lazar2022_change_of_constants_*.
The independent route is the 18 x 18 one: the tensor restricted to
an orthonormal basis of V and projected on the five isotropic tensors of
Lazar et al. (2022) through their Gram matrix (no block algebra involved).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.spatial.transform import Rotation

from gradinv import averages as av
from gradinv import cubic, hexagonal, isotropic
from gradinv._tensor import rotation_27
from gradinv.algebra import BlockAlgebra
from helpers import basis_V, is_zero, random_rotation
from references import averages_sources as ref

R = sp.Rational
ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LZ = DATA["datasets"]["lazar2022"]
VOIGT_DATA = {"Al": LZ["D_voigt"]["Al"], "Cu": LZ["D_voigt"]["Cu"], "Fe": LZ["D_voigt"]["Fe"],
              "W": LZ["D_voigt"]["W"], "Cu_EAM": DATA["datasets"]["po2019"]["D_voigt"]["Cu_EAM"]}
A_DATA = {k: cubic.a_from_voigt(v) for k, v in VOIGT_DATA.items()}
PD_SETS = ("Al", "Cu", "Cu_EAM")

CUB, HEX, ISO = cubic.algebra(), hexagonal.algebra(), isotropic.algebra()
BV = basis_V()


def blocks_of(name):
    return cubic.from_lazar2022(A_DATA[name])


def iso_a(iso):
    return np.asarray(isotropic.to_lazar2022(iso), float)


# ------------------------------------------------------------------ 18x18 route
ISO18 = [BV.T @ isotropic.lazar2022_tensor(np.eye(5)[k]) @ BV for k in range(5)]
GRAM = np.array([[np.sum(X * Y) for Y in ISO18] for X in ISO18])


def iso_coeffs(M18):
    """a1..a5 of the Frobenius-orthogonal isotropic projection of an 18x18 matrix."""
    return np.linalg.solve(GRAM, np.array([np.sum(X * M18) for X in ISO18]))


def iso18(c):
    return sum(ci * X for ci, X in zip(c, ISO18))


def route18(D27):
    """(Voigt a, Reuss a, Reuss 18x18) by projection, without the block algebra."""
    D18 = BV.T @ D27 @ BV
    SR = iso18(iso_coeffs(np.linalg.inv(D18)))
    DR = np.linalg.inv(SR)
    return iso_coeffs(D18), iso_coeffs(DR), DR


def rand_pd(rng, n, scale=1.0):
    A = rng.normal(size=(n, n))
    return scale * (A @ A.T + 0.05 * np.eye(n))


def rand_blocks(alg, rng, pd=True, scale=1.0):
    if pd:
        return {p: rand_pd(rng, len(c), scale) for p, c, _d in alg.irreps}
    return {p: rng.normal(size=(len(c), len(c))) for p, c, _d in alg.irreps}


def general_blocks(irreps, prefix, symmetric=False):
    out = {}
    for p, copies, _d in irreps:
        n = len(copies)
        if symmetric:
            M = sp.zeros(n, n)
            for a in range(n):
                for b in range(a, n):
                    M[a, b] = M[b, a] = sp.Symbol(f"{prefix}{p}_{a}{b}")
        else:
            M = sp.Matrix(n, n, lambda a, b: sp.Symbol(f"{prefix}{p}_{a}{b}"))
        out[p] = M
    return out


# ------------------------------------------------------------------ A rule = projection
@pytest.mark.parametrize("alg", [CUB, HEX], ids=["cubic", "hexagonal"])
def test_weighted_rule_is_the_frobenius_projection(alg):
    rng = np.random.default_rng(250)
    X = rand_blocks(alg, rng, pd=False)                 # general, non-symmetric element
    rule = av.isotropic_part(alg, X)
    proj = ISO.moduli(alg.assemble(X))
    assert all(np.allclose(rule[k], proj[k], atol=1e-13) for k in rule)


@pytest.mark.core
@pytest.mark.exact
@pytest.mark.parametrize("cls", [cubic, hexagonal], ids=["cubic, 15 symbols", "hexagonal, 32 symbols"])
def test_weighted_rule_is_the_frobenius_projection_exact(cls):
    alg = cls.algebra(exact=True)
    X = general_blocks(alg.irreps, "x")
    rule = av.isotropic_part(alg, X)
    proj = isotropic.algebra(exact=True).moduli(alg.assemble(X))
    assert all(is_zero(rule[k] - proj[k]) for k in rule)


def test_weights():
    assert av.weights(CUB) == {"H3": {"T1": R(3, 7), "T2": R(3, 7), "A2": R(1, 7)},
                               "H2": {"T2": R(3, 5), "E": R(2, 5)}, "H1": {"T1": 1}}
    assert av.weights(HEX) == {"H3": {"A2": R(1, 7), "B1": R(1, 7), "B2": R(1, 7), "E1": R(2, 7),
                                      "E2": R(2, 7)},
                               "H2": {"A1": R(1, 5), "E1": R(2, 5), "E2": R(2, 5)},
                               "H1": {"A2": R(1, 3), "E1": R(2, 3)}}
    for alg in (CUB, HEX, ISO):
        assert all(sum(w.values()) == 1 for w in av.weights(alg).values())


def test_restriction_entries_and_rotation_invariance():
    rng = np.random.default_rng(251)
    for alg in (CUB, HEX):
        res = av.restriction(alg)
        assert all(set(np.unique(B)) <= {0, 1} for v in res.values() for B in v.values())
        rot = av._compute_restriction(alg.rotated(random_rotation(rng)))
        assert all(np.array_equal(res[k][p], rot[k][p]) for k in res for p in res[k])


def test_embed_isotropic_round_trip():
    rng = np.random.default_rng(252)
    iso = {"H3": np.array([[0.7]]), "H2": np.array([[-0.3]]), "H1": rng.normal(size=(2, 2))}
    for alg in (CUB, HEX, ISO):
        emb = av.embed_isotropic(alg, iso)
        assert np.allclose(alg.assemble(emb), ISO.assemble(iso), atol=1e-14)
        back = av.isotropic_part(alg, emb)
        assert all(np.allclose(back[k], iso[k], atol=1e-14) for k in iso)


# ------------------------------------------------------------------ C Voigt in a_i
@pytest.mark.core
@pytest.mark.exact
def test_voigt_reproduces_lazar2022_eq158_exact():
    a = sp.symbols("a1:12")
    blk = cubic.from_lazar2022(a, exact=True)
    V = av.voigt(cubic.algebra(exact=True), blk)
    aV = isotropic.to_lazar2022(V, exact=True)
    assert all(is_zero(x - y) for x, y in zip(aV, ref.lazar2022_eq158(a)))
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = a
    assert is_zero(V["H3"][0, 0] - (2 * a4 + 2 * a5 + R(8, 5) * (a6 + a7) + R(2, 5) * a8 + R(6, 35) * a11))
    assert is_zero(V["H2"][0, 0] - (2 * a4 - a5 + R(2, 5) * a6 - R(4, 5) * a7 + R(2, 5) * a8))
    assert is_zero(V["H1"] - blk["T1"][1:, 1:])


def test_voigt_closed_cubic_formulas_numeric():
    rng = np.random.default_rng(253)
    a = rng.normal(size=11)
    blk = cubic.from_lazar2022(a)
    V = av.voigt(CUB, blk)
    M, Nn, mE, mA = blk["T1"], blk["T2"], blk["E"][0, 0], blk["A2"][0, 0]
    assert np.isclose(V["H3"][0, 0], (3 * M[0, 0] + 3 * Nn[0, 0] + mA) / 7)
    assert np.isclose(V["H2"][0, 0], (3 * Nn[1, 1] + 2 * mE) / 5)
    assert np.allclose(V["H1"], M[1:, 1:])
    assert np.allclose(iso_a(V), ref.lazar2022_eq158(a))


# ------------------------------------------------------------------ D Reuss closed forms
@pytest.mark.core
@pytest.mark.exact
def test_reuss_cubic_closed_schur_complement_exact():
    X = general_blocks(cubic.IRREPS, "c", symmetric=True)
    alg = cubic.algebra(exact=True)
    gen = av.reuss(alg, X, check_pd=False)
    clo = av.reuss_cubic_closed(X)
    assert all(is_zero(gen[k] - clo[k]) for k in gen)
    x, n22 = sp.Symbol("x", positive=True), sp.Symbol("n22")
    assert is_zero(5 / (3 * n22 + 2 / x) / x - R(5, 2) / (1 + R(3, 2) * x * n22))


def test_reuss_cubic_closed_numeric():
    rng = np.random.default_rng(254)
    for _ in range(20):
        blk = rand_blocks(CUB, rng)
        gen, clo = av.reuss(CUB, blk), av.reuss_cubic_closed(blk)
        assert all(np.allclose(gen[k], clo[k], atol=1e-12) for k in gen)


def test_reuss_hexagonal_written_out():
    rng = np.random.default_rng(255)
    blk = rand_blocks(HEX, rng)
    r = av.reuss(HEX, blk)
    inv = BlockAlgebra.inverse(blk)
    m3 = 7 / (inv["A2"][0, 0] + 2 * inv["E1"][0, 0] + 2 * inv["E2"][0, 0]
              + 1 / blk["B1"][0, 0] + 1 / blk["B2"][0, 0])
    m2 = 5 / (1 / blk["A1"][0, 0] + 2 * inv["E1"][1, 1] + 2 * inv["E2"][1, 1])
    MR = np.linalg.inv(inv["A2"][1:, 1:] / 3 + 2 * inv["E1"][2:, 2:] / 3)
    assert np.isclose(r["H3"][0, 0], m3) and np.isclose(r["H2"][0, 0], m2)
    assert np.allclose(r["H1"], MR)
    v = av.voigt(HEX, blk)
    assert np.isclose(v["H3"][0, 0], (blk["A2"][0, 0] + blk["B1"][0, 0] + blk["B2"][0, 0]
                                      + 2 * blk["E1"][0, 0] + 2 * blk["E2"][0, 0]) / 7)
    assert np.isclose(v["H2"][0, 0], (blk["A1"][0, 0] + 2 * blk["E1"][1, 1] + 2 * blk["E2"][1, 1]) / 5)
    assert np.allclose(v["H1"], blk["A2"][1:, 1:] / 3 + 2 * blk["E1"][2:, 2:] / 3)


# ------------------------------------------------------------------ E order
@pytest.mark.core
def test_reuss_below_voigt_on_random_tensors():
    """Proposition of section 6(b) on random samples (consistency); the proof is checked in
    test_reuss_below_voigt_proof_identity_exact."""
    rng = np.random.default_rng(25)
    worst = np.inf
    for alg in (CUB, HEX):
        for _ in range(500):
            bl = rand_blocks(alg, rng)
            v, r = av.voigt(alg, bl), av.reuss(alg, bl)
            gaps = [v["H3"][0, 0] - r["H3"][0, 0], v["H2"][0, 0] - r["H2"][0, 0],
                    np.linalg.eigvalsh(v["H1"] - r["H1"]).min()]
            worst = min(worst, min(gaps))
    assert worst >= -1e-12


def _bmul(X, Y):
    return {p: (X[p] * Y[p] if isinstance(X[p], sp.MatrixBase) else X[p] @ Y[p]) for p in X}


def _bsub(X, Y):
    return {p: X[p] - Y[p] for p in X}


def _proof_integrand(alg, bl):
    """R = (A^-1 - D^-1) D (A^-1 - D^-1) with A = D^V embedded in the crystal algebra.

    The integrand of eq. (reuss-below-voigt) at Q is (Q*) R (Q*)^T, because A is
    isotropic; its average is the isotropic part of R.
    """
    A = av.embed_isotropic(alg, av.voigt(alg, bl))
    Dm = _bsub(BlockAlgebra.inverse(A), BlockAlgebra.inverse(bl))
    return _bmul(_bmul(Dm, bl), Dm)


def _rational_pd(rng, n):
    A = sp.Matrix(n, n, lambda a, b: R(int(rng.integers(-5, 6)), 7))
    return A * A.T + sp.eye(n) / 5


@pytest.mark.core
@pytest.mark.exact
@pytest.mark.parametrize("cls", [cubic, hexagonal], ids=["cubic", "hexagonal"])
def test_reuss_below_voigt_proof_identity_exact(cls):
    """Proof of D^R <= D^V in section 6(b), in exact arithmetic.

    (D^R)^-1 - (D^V)^-1 = <(A^-1 - D_Q^-1) D_Q (A^-1 - D_Q^-1)>, A = D^V: the identity
    (it uses <D_Q> = A and that A commutes with the rotations), the positive
    semi-definiteness of the average, its trace equal to tr R > 0 for an anisotropic
    D (so equality forces isotropy), and R = 0 for an isotropic D.
    """
    alg = cls.algebra(exact=True)
    dims = {p: d for p, _c, d in alg.irreps}
    rng = np.random.default_rng(66)
    for _ in range(2):
        bl = {p: _rational_pd(rng, len(c)) for p, c, _d in alg.irreps}
        Rbl = _proof_integrand(alg, bl)
        lhs = av.isotropic_part(alg, Rbl)
        rhs = _bsub(av.isotropic_part(alg, BlockAlgebra.inverse(bl)),
                    BlockAlgebra.inverse(av.voigt(alg, bl)))
        assert all(is_zero(lhs[k] - rhs[k]) for k in lhs)
        # positive semi-definite (Sylvester on the 1x1 and 2x2 isotropic blocks)
        assert lhs["H3"][0, 0] >= 0 and lhs["H2"][0, 0] >= 0
        assert lhs["H1"][0, 0] >= 0 and sp.simplify(lhs["H1"].det()) >= 0
        # trace over V of the average = trace of R > 0
        tr_avg = 7 * lhs["H3"][0, 0] + 5 * lhs["H2"][0, 0] + 3 * lhs["H1"].trace()
        tr_R = sum(dims[p] * Rbl[p].trace() for p in Rbl)
        assert is_zero(sp.Matrix([[tr_avg - tr_R]])) and sp.simplify(tr_R) > 0
        # hence D^R <= D^V block by block
        r, v = av.reuss(alg, bl, check_pd=False), av.voigt(alg, bl)
        assert sp.simplify(v["H3"][0, 0] - r["H3"][0, 0]) >= 0
        assert sp.simplify(v["H2"][0, 0] - r["H2"][0, 0]) >= 0
        G = v["H1"] - r["H1"]
        assert sp.simplify(G[0, 0]) >= 0 and sp.simplify(G.det()) >= 0
    # isotropic D: the integrand vanishes, Voigt = Reuss
    iso = {"H3": sp.Matrix([[R(3, 2)]]), "H2": sp.Matrix([[R(1, 3)]]),
           "H1": sp.Matrix([[2, R(1, 2)], [R(1, 2), 1]])}
    bl = av.embed_isotropic(alg, iso)
    assert all(is_zero(B) for B in _proof_integrand(alg, bl).values())


@pytest.mark.parametrize("name", ["Al", "Cu", "Cu_EAM", "hexagonal"])
def test_reuss_below_voigt_proof_identity_by_the_18x18_route(name):
    """The same identity on the 18x18 matrices, with the projection computed from the
    isotropic tensors of Lazar et al. (2022), without the block algebra."""
    if name == "hexagonal":
        D27 = HEX.assemble(rand_blocks(HEX, np.random.default_rng(661)))
    else:
        D27 = CUB.assemble(blocks_of(name))
    D18 = BV.T @ D27 @ BV
    A18 = iso18(iso_coeffs(D18))
    Ai, Di = np.linalg.inv(A18), np.linalg.inv(D18)
    R18 = (Ai - Di) @ D18 @ (Ai - Di)
    avg_R = iso18(iso_coeffs(R18))
    rhs = iso18(iso_coeffs(Di)) - Ai
    scale = np.linalg.norm(rhs)
    assert np.linalg.norm(avg_R - rhs) < 1e-11 * max(1.0, scale)
    assert np.linalg.eigvalsh(avg_R).min() > -1e-12 * scale
    assert abs(np.trace(avg_R) - np.trace(R18)) < 1e-10 * np.trace(R18)
    assert np.trace(R18) > 0


def test_equality_iff_isotropic():
    base = np.array([0.1, 0.3, -0.2, 0.4, 0.2] + [0.0] * 6)
    bl = cubic.from_lazar2022(base)
    v, r = av.voigt(CUB, bl), av.reuss(CUB, bl)
    assert all(np.allclose(v[k], r[k], atol=1e-14) for k in v)
    for k in range(6):                  # each cubic constant alone gives a strict gap
        a = base.copy()
        a[5 + k] = 0.02
        bl = cubic.from_lazar2022(a)
        v, r = av.voigt(CUB, bl), av.reuss(CUB, bl)
        gap = max(v["H3"][0, 0] - r["H3"][0, 0], v["H2"][0, 0] - r["H2"][0, 0],
                  np.linalg.eigvalsh(v["H1"] - r["H1"]).max())
        assert gap > 1e-6, k


# ------------------------------------------------------------------ F data
@pytest.mark.parametrize("name", list(A_DATA))
def test_voigt_block_route_equals_18x18_route(name):
    aV18, _aR18, _ = route18(cubic.lazar2022_tensor(A_DATA[name]))
    aV = iso_a(av.voigt(CUB, blocks_of(name)))
    assert np.abs(aV - aV18).max() < 1e-12
    if name in ref.TABLE6:
        assert np.abs(aV - ref.TABLE6[name]).max() < 1e-5


@pytest.mark.parametrize("name", PD_SETS)
def test_reuss_and_hill_on_data(name):
    aV18, aR18, _ = route18(cubic.lazar2022_tensor(A_DATA[name]))
    bl = blocks_of(name)
    aV, aR, aH = iso_a(av.voigt(CUB, bl)), iso_a(av.reuss(CUB, bl)), iso_a(av.hill(CUB, bl))
    assert np.abs(aR - aR18).max() < 1e-10
    assert np.abs(aH - (aV + aR) / 2).max() < 1e-14
    _aV, _aR, DR18 = route18(cubic.lazar2022_tensor(A_DATA[name]))
    assert np.linalg.eigvalsh(iso18(aV18) - DR18).min() > -1e-12      # D_R <= D_V (18x18 route)


# ------------------------------------------------------------------ G Monte Carlo
@pytest.mark.core
@pytest.mark.slow
@pytest.mark.parametrize("name", ["Al", "Cu", "Cu_EAM", "hexagonal"])
def test_monte_carlo_average_of_rotated_compliances(name):
    """Independent route: <S> by averaging rotated compliances (N = 2000 and 20 000).

    The sampling error scales with the anisotropic part of S, so the criterion is
    |<S>_N - S_R| <= 3 |S - S_R| / sqrt(N) (all three cases give about 1.3/sqrt(N)).
    """
    if name == "hexagonal":
        bl = rand_blocks(HEX, np.random.default_rng(256), scale=0.5)
        alg = HEX
    else:
        bl, alg = blocks_of(name), CUB
    S27 = alg.assemble(BlockAlgebra.inverse(bl))
    SR = ISO.assemble(av.isotropic_part(alg, BlockAlgebra.inverse(bl)))
    rots = Rotation.random(20000, random_state=np.random.default_rng(7)).as_matrix()
    errs = {}
    acc = np.zeros((27, 27))
    for t, Q in enumerate(rots, start=1):
        Q3 = rotation_27(Q)
        acc += Q3 @ S27 @ Q3.T
        if t in (2000, 20000):
            errs[t] = np.linalg.norm(acc / t - SR) / np.linalg.norm(SR)
    aniso = np.linalg.norm(S27 - SR) / np.linalg.norm(SR)
    for t, err in errs.items():
        assert err <= 3 * aniso / np.sqrt(t), (t, err, aniso)
    assert errs[20000] < errs[2000]
    if name in ("Al", "Cu_EAM"):
        assert errs[20000] < 3.5e-3        # Al 3.3e-3, Cu EAM 3.2e-3


# ------------------------------------------------------------------ H Cu MEAM factor
def test_cu_meam_factor():
    bl = blocks_of("Cu")
    r = av.reuss(CUB, bl)
    mE = bl["E"][0, 0]
    assert abs(r["H2"][0, 0] / mE - av.reuss_mE_ratio(bl)) < 1e-10
    small = {k: v.copy() for k, v in bl.items()}
    small["E"] = np.array([[1e-12]])
    assert abs(av.reuss_mE_ratio(small) - 2.5) < 1e-9


# ------------------------------------------------------------------ I limits
@pytest.mark.exact
def test_isotropic_limit_exact():
    b = sp.symbols("b1:6")
    bl = cubic.from_lazar2022(list(b) + [0] * 6, exact=True)
    iso = isotropic.from_lazar2022(b, exact=True)
    alg = cubic.algebra(exact=True)
    v, r = av.voigt(alg, bl), av.reuss(alg, bl, check_pd=False)
    assert all(is_zero(v[k] - iso[k]) and is_zero(r[k] - iso[k]) for k in iso)


@pytest.mark.exact
def test_separable_limit_exact():
    """C (x) l^2 delta with cubic C: Voigt = C_V (x) l^2 delta, Reuss = C_R (x) l^2 delta."""
    C11, C12, C44, l2 = sp.symbols("C11 C12 C44 l2", positive=True)
    asep = [0, C12 * l2 / 2, 0, C44 * l2, 0, 0, 0, (C11 - C12 - 2 * C44) * l2, 0, 0, 0]
    alg = cubic.algebra(exact=True)
    bl = cubic.from_lazar2022(asep, exact=True)
    lamV, muV = (C11 + 4 * C12 - 2 * C44) / 5, (C11 - C12 + 3 * C44) / 5
    muR = 5 * C44 * (C11 - C12) / (4 * C44 + 3 * (C11 - C12))
    lamR = (C11 + 2 * C12) / 3 - 2 * muR / 3
    v, r = av.voigt(alg, bl), av.reuss(alg, bl, check_pd=False)
    tv = isotropic.from_lazar2022([0, lamV * l2 / 2, 0, muV * l2, 0], exact=True)
    tr = isotropic.from_lazar2022([0, lamR * l2 / 2, 0, muR * l2, 0], exact=True)
    assert all(is_zero(v[k] - tv[k]) for k in tv)
    assert all(is_zero(r[k] - tr[k]) for k in tr)


@pytest.mark.parametrize("name", ["Al", "Fe"])
def test_limits_numeric(name):
    """Isotropic (Table 6 of Al, a6..a11 = 0) and separable (eq. 109 of Lazar and Po
    2024, l^2 = 1.3, C of Lazar et al. 2022) limits by the 18x18 and block routes."""
    aiso = np.array(ref.TABLE6["Al"] + [0.0] * 6)
    v, r = av.voigt(CUB, cubic.from_lazar2022(aiso)), av.reuss(CUB, cubic.from_lazar2022(aiso))
    assert np.abs(iso_a(v) - aiso[:5]).max() < 1e-14 and np.abs(iso_a(r) - aiso[:5]).max() < 1e-14
    l2 = 1.3
    C11, C12, C44 = LZ["C"][name]
    asep = np.array([0, C12 * l2 / 2, 0, C44 * l2, 0, 0, 0, (C11 - C12 - 2 * C44) * l2, 0, 0, 0])
    lamV, muV = (C11 + 4 * C12 - 2 * C44) / 5, (C11 - C12 + 3 * C44) / 5
    muR = 5 * C44 * (C11 - C12) / (4 * C44 + 3 * (C11 - C12))
    lamR = (C11 + 2 * C12) / 3 - 2 * muR / 3
    aV18, aR18, _ = route18(cubic.lazar2022_tensor(asep))
    bl = cubic.from_lazar2022(asep)
    for aV, aR in ((aV18, aR18), (iso_a(av.voigt(CUB, bl)), iso_a(av.reuss(CUB, bl)))):
        assert np.abs(aV - [0, lamV * l2 / 2, 0, muV * l2, 0]).max() < 1e-14
        assert np.abs(aR - [0, lamR * l2 / 2, 0, muR * l2, 0]).max() < 1e-14


# ------------------------------------------------------------------ J rejection
@pytest.mark.parametrize("name", ["Fe", "W"])
def test_fe_and_w_rejected_for_reuss_and_hill(name):
    """Fe and W are not positive definite (block T2), so Reuss and Hill are refused.
    Formed anyway by the 18 x 18 route, the formal Reuss tensor is indefinite for Fe
    and, for W, not below the Voigt one: neither is an estimate."""
    bl = blocks_of(name)
    assert BlockAlgebra.non_positive_blocks(bl) == ["T2"]
    with pytest.raises(ValueError, match="positive definite"):
        av.reuss(CUB, bl)
    with pytest.raises(ValueError, match="positive definite"):
        av.hill(CUB, bl)
    av.voigt(CUB, bl)                    # Voigt is always formed
    aV, _aR, DR = route18(cubic.lazar2022_tensor(A_DATA[name]))
    if name == "Fe":
        assert np.linalg.eigvalsh(DR).min() < 0
    else:
        assert np.linalg.eigvalsh(DR).min() > 0
        assert np.linalg.eigvalsh(iso18(aV) - DR).min() < 0


# ------------------------------------------------------------------ K hexagonal
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hexagonal_block_route_equals_18x18_route(seed):
    bl = rand_blocks(HEX, np.random.default_rng(300 + seed), scale=0.5)
    aV18, aR18, _ = route18(HEX.assemble(bl))
    assert np.abs(iso_a(av.voigt(HEX, bl)) - aV18).max() < 1e-10
    assert np.abs(iso_a(av.reuss(HEX, bl)) - aR18).max() < 1e-10


def test_hexagonal_dependence_on_the_extra_modulus_numeric():
    rng = np.random.default_rng(257)
    bl = rand_blocks(HEX, rng)
    sig = 0.5 * (bl["B1"][0, 0] + bl["B2"][0, 0])
    out = []
    for D in (0.3 * sig, -0.3 * sig, 0.0):
        b = {k: v.copy() for k, v in bl.items()}
        b["B1"], b["B2"] = np.array([[sig + D / 2]]), np.array([[sig - D / 2]])
        out.append((av.voigt(HEX, b), av.reuss(HEX, b)))
    (v1, r1), (v2, r2), (v0, r0) = out
    assert all(np.allclose(v1[k], v0[k]) and np.allclose(v2[k], v0[k]) for k in v0)
    assert all(np.allclose(r1[k], r2[k]) for k in r0)
    assert not np.allclose(r1["H3"], r0["H3"])


@pytest.mark.core
@pytest.mark.exact
def test_hexagonal_dependence_on_the_extra_modulus_exact():
    """Voigt depends on m_B1 + m_B2 only; Reuss is even in Delta (section 6)."""
    rng = np.random.default_rng(25)
    sig, Dl = sp.symbols("sigma Delta")
    X = {}
    for p, copies, _d in hexagonal.IRREPS:
        n = len(copies)
        A = sp.Matrix(n, n, lambda a, b: R(int(rng.integers(-5, 6)), 7))
        X[p] = A * A.T + sp.eye(n)
    X["B1"], X["B2"] = sp.Matrix([[sig + Dl / 2]]), sp.Matrix([[sig - Dl / 2]])
    alg = hexagonal.algebra(exact=True)
    v, r = av.voigt(alg, X), av.reuss(alg, X, check_pd=False)
    assert all(is_zero(sp.diff(v[k], Dl)) for k in v)
    flip = {Dl: -Dl}
    assert all(is_zero(r[k] - r[k].subs(flip)) for k in r)
    assert is_zero(1 / (sig + Dl / 2) + 1 / (sig - Dl / 2) - 8 * sig / (4 * sig ** 2 - Dl ** 2))
    Xi = BlockAlgebra.inverse(X)
    explicit = 7 / (Xi["A2"][0, 0] + 2 * Xi["E1"][0, 0] + 2 * Xi["E2"][0, 0] + 1 / X["B1"][0, 0]
                    + 1 / X["B2"][0, 0])
    assert is_zero(r["H3"][0, 0] - explicit)

