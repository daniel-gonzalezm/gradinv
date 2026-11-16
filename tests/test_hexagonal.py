"""Hexagonal algebra (class 6/mmm).

Embeddings, D6h invariance and Mulliken characters, commutants (D6 32/22, D_inf
31/21), multiplication table, coordinate-free construction, the extra modulus
(Delta = D :: H6 / 4 = 4 eta), isotropic limit, Auffray et al. (2013), Monchiet and
Bonnet (2013) TI limit, inverse and positive definiteness, separable C x L,
sign conventions.
"""
import itertools

import numpy as np
import pytest
import sympy as sp

from gradinv import hexagonal, isotropic
from gradinv._tensor import N, R, TRIPLES, index, kd, rotation_27, tensor6, to_float
from gradinv.hexagonal import AUFFRAY2013_PARAMS as P_
from gradinv.index_forms import MB2013_GROUPS, monchiet_bonnet_2013_tensors
from gradinv.skeleton import identity_V
from helpers import basis_V, commutant_dimension, is_zero, rot_z

HEX = hexagonal.algebra()
U = HEX.units()
I = identity_V()
ISO_U = isotropic.algebra().units()
G6 = hexagonal.d6_group()


@pytest.fixture(scope="module")
def hexact():
    return hexagonal.algebra(exact=True)


@pytest.fixture(scope="module")
def mb2013():
    return monchiet_bonnet_2013_tensors()


def _axial_generator(exact):
    W = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    if exact:
        Wn, I3, kp = sp.Matrix(W), sp.eye(3), sp.kronecker_product
    else:
        Wn, I3, kp = np.array(W, float), np.eye(3), np.kron
    return kp(kp(Wn, I3), I3) + kp(kp(I3, Wn), I3) + kp(kp(I3, I3), Wn)


J27 = _axial_generator(False)


def commutes_axial(X):
    return np.allclose(J27 @ X, X @ J27, atol=1e-12)


# ------------------------------------------------------------------ A embeddings
def test_embeddings_gram_is_identity():
    Eall = HEX.embedding_matrix()
    assert np.allclose(Eall.T @ Eall, np.eye(18), atol=1e-14)
    assert np.allclose(I @ Eall, Eall, atol=1e-14)


@pytest.mark.core
@pytest.mark.exact
def test_embeddings_gram_exact(hexact):
    """Gram of the 18 columns = Id exactly: isometric, orthogonal, complete, hence the table."""
    Eall = hexact.embedding_matrix()
    assert is_zero(Eall.T * Eall - sp.eye(18))
    assert is_zero(identity_V(exact=True) * Eall - Eall)


def test_copies_lie_in_their_pieces():
    piece = {"3": ISO_U[("H3", "3", "3")], "2": ISO_U[("H2", "2", "2")],
             "s": ISO_U[("H1", "s", "s")], "r": ISO_U[("H1", "r", "r")]}
    for p, copies, _d in hexagonal.IRREPS:
        for a in copies:
            E = HEX.embedding(p, a)
            assert np.allclose(piece[a] @ E, E, atol=1e-14), (p, a)
    diag = lambda c: sum(U[(p, c, c)] for p, copies, _d in hexagonal.IRREPS if c in copies)
    assert np.allclose(diag("3"), piece["3"], atol=1e-14) and np.allclose(diag("2"), piece["2"], atol=1e-14)
    for a in "sr":
        for b in "sr":
            assert np.allclose(U[("A2", a, b)] + U[("E1", a, b)], ISO_U[("H1", a, b)], atol=1e-14)
    assert np.allclose(sum(U[(p, a, a)] for p, c, _d in hexagonal.IRREPS for a in c), I, atol=1e-14)


@pytest.mark.exact
def test_structure_tensors_exact():
    h, k, _H6 = hexagonal.structure_tensors(exact=True)
    z = [1, sp.I, 0]
    for t in TRIPLES:
        w = z[t[0]] * z[t[1]] * z[t[2]]
        assert h[t] == sp.re(w) and k[t] == sp.im(w)
    assert sum(v**2 for v in h.values()) == 4 and sum(v**2 for v in k.values()) == 4
    assert all(sum(T[(p, p, m)] for p in range(3)) == 0 for T in (h, k) for m in range(3))


# ------------------------------------------------------------------ B invariance, characters, commutants
MULLIKEN = {"A1": (1, 1, 1, 1, 1, 1), "A2": (1, 1, 1, 1, -1, -1), "B1": (1, -1, 1, -1, 1, -1),
            "B2": (1, -1, 1, -1, -1, 1), "E1": (2, 1, -1, -2, 0, 0), "E2": (2, -1, -1, 2, 0, 0)}
CLASSES = (0, 1, 2, 3, 6, 9)   # E, C6, C3, C2, C2' (about e1), C2'' (about e2)


def test_d6h_invariance_and_characters():
    for gi, Q in enumerate(G6):
        RV = rotation_27(Q)
        for p, copies, _d in hexagonal.IRREPS:
            reps = []
            for a in copies:
                Ea = HEX.embedding(p, a)
                rp = Ea.T @ RV @ Ea
                assert np.allclose(RV @ Ea, Ea @ rp, atol=1e-13)
                reps.append(rp)
            assert all(np.allclose(r, reps[0], atol=1e-13) for r in reps)
            if gi in CLASSES:
                assert np.isclose(np.trace(reps[0]), MULLIKEN[p][CLASSES.index(gi)])
    assert np.allclose(rotation_27(-np.eye(3)), -np.eye(27))


@pytest.mark.core
@pytest.mark.exact
def test_d6_invariance_exact(hexact):
    """Every copy is invariant and carries the same (exact) matrices as the other copies."""
    Ux = hexact.units()
    for Q in hexagonal.d6_group(exact=True):
        RV = rotation_27(Q)
        for Uk in Ux.values():
            assert is_zero(RV * Uk * RV.T - Uk)
        for p, copies, _d in hexagonal.IRREPS:
            reps = []
            for a in copies:
                Ea = hexact.embedding(p, a)
                rp = sp.simplify(Ea.T * RV * Ea)
                assert is_zero(RV * Ea - Ea * rp)
                reps.append(rp)
            assert all(is_zero(r - reps[0]) for r in reps)


def test_only_B1_B2_move_under_rotations_about_n():
    moved = sorted(k for k, X in U.items() if not commutes_axial(X))
    assert moved == [("B1", "3", "3"), ("B2", "3", "3")]
    assert commutes_axial(U[("B1", "3", "3")] + U[("B2", "3", "3")])


def test_commutant_dimensions_and_basis():
    g6 = [G6[1], G6[6]]
    ginf = [rot_z(0.7319), G6[6]]
    dims = (commutant_dimension(g6), commutant_dimension(g6, True), commutant_dimension(ginf),
            commutant_dimension(ginf, True))
    assert dims == (32, 22, 31, 21)
    keys = list(U)
    gram = np.array([[np.sum(U[a] * U[b]) for b in keys] for a in keys])
    dims_p = dict((p, d) for p, _c, d in hexagonal.IRREPS)
    assert np.allclose(gram, np.diag([dims_p[k[0]] for k in keys]), atol=1e-13)
    assert HEX.dimension == 32 and HEX.symmetric_dimension == 22


# ------------------------------------------------------------------ C table
@pytest.mark.core
def test_multiplication_table_all_1024_products():
    err = 0.0
    for (p, a, b), X in U.items():
        for (q, c, d), Y in U.items():
            exp = U[(p, a, d)] if (p == q and b == c) else 0.0
            err = max(err, np.abs(X @ Y - exp).max())
    assert err < 1e-13


# ------------------------------------------------------------------ D coordinate-free construction
@pytest.mark.core
@pytest.mark.exact
def test_coordinate_free_construction_exact(hexact):
    qw, qx, qy, qz = 1, 2, 3, 4
    Rq = sp.Matrix([[qw**2 + qx**2 - qy**2 - qz**2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                    [2 * (qx * qy + qw * qz), qw**2 - qx**2 + qy**2 - qz**2, 2 * (qy * qz - qw * qx)],
                    [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw**2 - qx**2 - qy**2 + qz**2]]) / 30
    assert is_zero(Rq.T * Rq - sp.eye(3)) and Rq.det() == 1
    algR = hexagonal.algebra_from_axes(list(Rq[:, 2]), list(Rq[:, 0]), exact=True)
    RR = rotation_27(Rq)
    UR, Ux = algR.units(), hexact.units()
    for k in Ux:
        assert is_zero(UR[k] - RR * Ux[k] * RR.T), k


def test_rotated_algebra_numeric():
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    algF = hexagonal.algebra_from_axes(Q[:, 2], Q[:, 0], exact=False)
    algR = HEX.rotated(Q)
    for k, X in algR.units().items():
        assert np.allclose(algF.units()[k], X, atol=1e-13)


# ------------------------------------------------------------------ E extra modulus
def test_extra_modulus_structure_numeric():
    h, k, H6 = hexagonal.structure_tensors()
    assert np.allclose(U[("B1", "3", "3")], np.outer(h.ravel(), h.ravel()) / 4)
    assert np.allclose(U[("B2", "3", "3")], np.outer(k.ravel(), k.ravel()) / 4)
    assert np.allclose(U[("B1", "3", "3")] - U[("B2", "3", "3")], H6 / 4)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(27, 27))
    D = I @ (X + X.T) @ I
    assert np.isclose(hexagonal.extra_modulus(HEX.moduli(D)), np.sum(D * H6) / 4)
    R30 = rotation_27(rot_z(np.pi / 6))
    assert np.allclose(R30 @ U[("B1", "3", "3")] @ R30.T, U[("B2", "3", "3")])
    assert np.allclose(R30 @ U[("B2", "3", "3")] @ R30.T, U[("B1", "3", "3")])


@pytest.mark.exact
def test_extra_modulus_exact(hexact):
    Ux = hexact.units()
    h, k, H6 = hexagonal.structure_tensors(exact=True)
    hh = sp.Matrix(N, 1, lambda r, _c: h[TRIPLES[r]])
    kk = sp.Matrix(N, 1, lambda r, _c: k[TRIPLES[r]])
    assert is_zero(Ux[("B1", "3", "3")] - hh * hh.T / 4) and is_zero(Ux[("B2", "3", "3")] - kk * kk.T / 4)
    assert is_zero(Ux[("B1", "3", "3")] - Ux[("B2", "3", "3")] - H6 / 4)
    idx6 = list(itertools.product(range(3), repeat=6))
    H6t = {t: H6[index(*t[:3]), index(*t[3:])] for t in idx6}
    assert all(H6t[t] == H6t[t[:q] + (t[q + 1], t[q]) + t[q + 2:]] for t in idx6 for q in range(5))
    assert all(sum(H6t[(p, p) + t[2:]] for p in range(3)) == 0 for t in idx6)
    Ds = sp.Matrix(N, N, lambda r, c: sp.Symbol(f"D{min(r, c)}_{max(r, c)}"))
    Ix = identity_V(exact=True)
    Ds = Ix * Ds * Ix
    lhs = hexagonal.extra_modulus(hexact.moduli(Ds))
    assert sp.expand(lhs - sum(Ds[t] * H6[t] for t in range(N * N)) / 4) == 0
    R30 = rotation_27(sp.Matrix([[sp.sqrt(3) / 2, -R(1, 2), 0], [R(1, 2), sp.sqrt(3) / 2, 0], [0, 0, 1]]))
    assert is_zero(R30 * Ux[("B1", "3", "3")] * R30.T - Ux[("B2", "3", "3")])


# ------------------------------------------------------------------ F isotropic limit
@pytest.mark.exact
def test_isotropic_limit_exact(hexact):
    a = sp.symbols("a1:6")
    B = hexact.moduli(isotropic.lazar2022_tensor(a, exact=True))
    m3, m2 = 2 * (a[3] + a[4]), 2 * a[3] - a[4]
    M = isotropic.from_lazar2022(a, exact=True)["H1"]
    exp = {"A1": sp.Matrix([[m2]]), "B1": sp.Matrix([[m3]]), "B2": sp.Matrix([[m3]]),
           "A2": sp.diag(m3, M), "E1": sp.diag(m3, m2, M), "E2": sp.diag(m3, m2)}
    assert all(is_zero(B[p] - exp[p]) for p in exp)
    BI = hexact.moduli(identity_V(exact=True))
    assert all(is_zero(BI[p] - sp.eye(BI[p].shape[0])) for p in BI)


def test_isotropic_limit_numeric():
    a = np.random.default_rng(3).normal(size=5)
    B = HEX.moduli(isotropic.lazar2022_tensor(a))
    iso = isotropic.from_lazar2022(a)
    m3, m2, M = iso["H3"][0, 0], iso["H2"][0, 0], iso["H1"]
    from scipy.linalg import block_diag
    assert np.allclose(B["A2"], block_diag([[m3]], M)) and np.allclose(B["E1"], block_diag(np.diag([m3, m2]), M))
    assert np.allclose(B["E2"], np.diag([m3, m2])) and np.isclose(B["A1"][0, 0], m2)
    assert np.isclose(B["B1"][0, 0], m3) and np.isclose(B["B2"][0, 0], m3)


# ------------------------------------------------------------------ G Auffray et al. (2013)
@pytest.mark.core
@pytest.mark.exact
def test_auffray2013_exact(hexact):
    Ix = identity_V(exact=True)
    Bas = hexagonal.auffray2013_basis()
    assert is_zero(Bas.T * Bas - sp.eye(18))
    Dau = hexagonal.auffray2013_tensor(P_, exact=True)
    assert is_zero(Ix * Dau * Ix - Dau) and is_zero(Dau - Dau.T)
    G = hexagonal.d6_group(exact=True)
    for Q in (G[1], G[6]):
        RQ = rotation_27(Q)
        assert is_zero(RQ * Dau * RQ.T - Dau)
    eta = P_[11]
    J = _axial_generator(True)
    Dau0 = Dau.subs(eta, 0)
    assert is_zero(J * Dau0 - Dau0 * J)                       # A_O(2): transversely isotropic
    Ac = Bas * sp.diag(hexagonal.auffray2013_blocks()[1] / eta, sp.zeros(13, 13)) * Bas.T
    h, _k, _H6 = hexagonal.structure_tensors(exact=True)
    hh = sp.Matrix(N, 1, lambda r, _c: h[TRIPLES[r]])
    assert is_zero(Ac - hh * hh.T)                            # their Ac = h x h
    B = hexact.moduli(Dau)
    assert is_zero(hexagonal.extra_modulus(B) - 4 * eta)     # Delta = 4 eta
    p = dict(zip([str(s) for s in P_], P_))
    s2 = sp.sqrt(2)
    assert is_zero(B["A1"][0, 0] - (p["j22"] - p["j23"]))
    assert is_zero(B["B2"][0, 0] - (p["a11"] - p["a12"] - s2 * p["a13"]))
    Gx = sp.Matrix([[p["j11"] + 4 * p["j12"] + 2 * p["j22"] + 2 * p["j23"],
                     s2 * (-p["j11"] - p["j12"] + p["j22"] + p["j23"])],
                    [s2 * (-p["j11"] - p["j12"] + p["j22"] + p["j23"]),
                     2 * p["j11"] - 4 * p["j12"] + p["j22"] + p["j23"]]]) / 3
    assert is_zero(B["E2"] - Gx)
    Jm = hexagonal._auffray_map_exact()
    assert sp.simplify(Jm.det()) == 2048 * sp.sqrt(2)
    back = hexagonal.to_auffray2013(B, exact=True)
    assert is_zero(sp.Matrix(back) - sp.Matrix(P_))


def test_auffray2013_numeric_round_trip():
    rng = np.random.default_rng(11)
    params = rng.normal(size=22)
    blk = hexagonal.from_auffray2013(params)
    direct = HEX.moduli(hexagonal.auffray2013_tensor(params))
    assert all(np.allclose(direct[p], blk[p]) for p in blk)
    assert np.isclose(hexagonal.extra_modulus(blk), 4 * params[11])
    assert np.allclose(hexagonal.to_auffray2013(blk), params)


# ------------------------------------------------------------------ H Monchiet and Bonnet (2013)
def test_monchiet_bonnet_2013_tables_and_identity(mb2013):
    T = mb2013
    Tf = {m: to_float(T[m]) for m in T}
    for m, X in Tf.items():
        assert np.allclose(I @ X @ I, X, atol=1e-14), m
        assert commutes_axial(X), m
        for Q in (G6[6], -np.eye(3)):
            RQ = rotation_27(Q)
            assert np.allclose(RQ @ X @ RQ.T, X, atol=1e-13)
    glist = list(MB2013_GROUPS.values())
    for gi, g1 in enumerate(glist):
        for gj, g2 in enumerate(glist):
            for ra, row in enumerate(g1):
                for cb, x in enumerate(row):
                    for rc, row2 in enumerate(g2):
                        for cd, y in enumerate(row2):
                            exp = Tf[g1[ra][cd]] if (gi == gj and cb == rc) else 0.0
                            assert np.allclose(Tf[x] @ Tf[y], exp, atol=1e-13), (x, y)
    idem = [1, 2, 3, 6, 7, 11, 15, 16, 21, 26, 31]
    assert np.allclose(sum(Tf[m] for m in idem), I, atol=1e-14)
    assert np.allclose(Tf[1], U[("B1", "3", "3")] + U[("B2", "3", "3")], atol=1e-14)
    assert np.allclose(Tf[2], U[("A1", "2", "2")], atol=1e-14)
    TI = [U[k].ravel() for k in U if k[0] not in ("B1", "B2")]
    TI.append((U[("B1", "3", "3")] + U[("B2", "3", "3")]).ravel())
    TT = np.array([Tf[m].ravel() for m in Tf])
    rk = (np.linalg.matrix_rank(TT), np.linalg.matrix_rank(np.array(TI)),
          np.linalg.matrix_rank(np.vstack([TT, TI])))
    assert rk == (31, 31, 31)


@pytest.mark.core
@pytest.mark.exact
def test_monchiet_bonnet_2013_exact(mb2013, hexact):
    """Symmetries and invariance exactly; in each block their units are matrix units in
    a fixed, non-orthonormal copy basis P (our block = P A_MB P^-1)."""
    T = mb2013
    Ix = identity_V(exact=True)
    J = _axial_generator(True)
    Ux = hexact.units()
    for m, X in T.items():
        assert is_zero(Ix * X * Ix - X), m
        assert is_zero(J * X - X * J), m
    assert is_zero(T[1] - Ux[("B1", "3", "3")] - Ux[("B2", "3", "3")])
    assert is_zero(T[2] - Ux[("A1", "2", "2")])
    PtP = {}
    for g, (p, cp, d) in {"E2": ("E2", ("3", "2"), 2), "A2": ("A2", ("3", "s", "r"), 1),
                          "E1": ("E1", ("3", "2", "s", "r"), 2)}.items():
        M_ = MB2013_GROUPS[g]
        kk = len(M_)
        C = {(i, j): sp.Matrix(kk, kk, lambda c, e: sp.nsimplify(sp.simplify(
            sum(T[M_[i][j]][t] * Ux[(p, cp[c], cp[e])][t] for t in range(N * N)) / d)))
             for i in range(kk) for j in range(kk)}
        j0 = max(range(kk), key=lambda c: sum(abs(x) for x in C[(0, 0)][:, c]))
        Pm = sp.Matrix.hstack(*[C[(i, 0)][:, j0] for i in range(kk)])
        Pm = sp.simplify(Pm / sp.sqrt((Pm[:, 0].T * Pm[:, 0])[0]))
        Pinv = sp.simplify(Pm.inv())
        for i in range(kk):
            for j in range(kk):
                assert is_zero(C[(i, j)] - Pm[:, i] * Pinv[j, :]), (g, i, j)
        PtP[g] = sp.simplify(Pm.T * Pm)
    # P^T P proportional to diag(1, 1/2), diag(1, 1/2, 1/2) and
    # [[1, 2/3], [2/3, 4/3]] (+) diag(2/3, 4/3)
    assert is_zero(PtP["E2"] / PtP["E2"][0, 0] - sp.diag(1, R(1, 2)))
    assert is_zero(PtP["A2"] / PtP["A2"][0, 0] - sp.diag(1, R(1, 2), R(1, 2)))
    E1 = PtP["E1"] / PtP["E1"][0, 0]
    assert is_zero(E1 - sp.diag(sp.Matrix([[1, R(2, 3)], [R(2, 3), R(4, 3)]]), sp.diag(R(2, 3), R(4, 3))))


# ------------------------------------------------------------------ I inverse, positive definiteness
@pytest.mark.core
@pytest.mark.exact
def test_inverse_of_general_element_exact(hexact):
    rng = np.random.default_rng(20261022)
    rat = lambda n: sp.Matrix(n, n, lambda *_: R(int(rng.integers(-9, 10)), int(rng.integers(1, 6))))
    Bg = {p: rat(len(c)) + 3 * sp.eye(len(c)) for p, c, _d in hexagonal.IRREPS}
    Dg, Sg = hexact.assemble(Bg), hexact.assemble(hexact.inverse(Bg))
    Ix = identity_V(exact=True)
    assert is_zero(Dg * Sg - Ix) and is_zero(Sg * Dg - Ix)
    Bm = hexact.moduli(Dg)
    assert all(is_zero(Bm[p] - Bg[p]) for p in Bg)


# ------------------------------------------------------------------ J separable C x L
def _ti_C(c11, c12, c13, c33, c44):
    V = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    c66 = (c11 - c12) / 2
    Cv = [[c11, c12, c13, 0, 0, 0], [c12, c11, c13, 0, 0, 0], [c13, c13, c33, 0, 0, 0],
          [0, 0, 0, c44, 0, 0], [0, 0, 0, 0, c44, 0], [0, 0, 0, 0, 0, c66]]
    return lambda i, j, k, l: Cv[V[(i, j)]][V[(k, l)]]


@pytest.mark.exact
def test_separable_extra_modulus_vanishes_exact(hexact):
    c = sp.symbols("c11 c12 c13 c33 c44")
    l1, l2 = sp.symbols("l1 l2")
    C = _ti_C(*c)
    L = sp.diag(l2, l2, l1)
    D = tensor6(lambda i, j, m, k, l, n: C(i, j, k, l) * L[m, n])
    assert sp.simplify(hexagonal.extra_modulus(hexact.moduli(D))) == 0


def test_separable_inverse_numeric():
    vals = (3.1, 1.2, 0.9, 2.7, 0.8)
    l1, l2 = 0.7, 1.3
    C = _ti_C(*vals)
    C9 = np.array([[C(r // 3, r % 3, c // 3, c % 3) for c in range(9)] for r in range(9)])
    I4 = np.array([[0.5 * (kd(r // 3, c // 3) * kd(r % 3, c % 3) + kd(r // 3, c % 3) * kd(r % 3, c // 3))
                    for c in range(9)] for r in range(9)])
    L = np.diag([l2, l2, l1])
    D = np.einsum("ab,mn->ambn", C9, L).reshape(27, 27)
    blk = HEX.moduli(D)
    assert abs(hexagonal.extra_modulus(blk)) < 1e-14
    Sexp = np.einsum("ab,mn->ambn", np.linalg.pinv(C9) @ I4, np.linalg.inv(L)).reshape(27, 27)
    assert np.abs(HEX.invert(D) - Sexp).max() < 1e-12


# ------------------------------------------------------------------ K sign conventions
def test_sign_conventions_reference_tensors():
    """N x delta has A_3s = sqrt6/5 and F_32 = -2 sqrt5/15; I x pi has G_32 = +sqrt2/3."""
    n = [0, 0, 1]
    pi = lambda i, j: kd(i, j) - n[i] * n[j]
    Nd = tensor6(lambda i, j, m, k, l, q: n[i] * n[j] * n[k] * n[l] * kd(m, q), exact=False)
    Ipi = tensor6(lambda i, j, m, k, l, q: 0.5 * (kd(i, k) * kd(j, l) + kd(i, l) * kd(j, k)) * pi(m, q),
                  exact=False)
    Inn = tensor6(lambda i, j, m, k, l, q: 0.5 * (kd(i, k) * kd(j, l) + kd(i, l) * kd(j, k)) * n[m] * n[q],
                  exact=False)
    b1, b2, b3 = HEX.moduli(Nd), HEX.moduli(Ipi), HEX.moduli(Inn)
    assert np.isclose(b1["A2"][0, 1], np.sqrt(6) / 5) and np.isclose(b1["E1"][0, 1], -2 * np.sqrt(5) / 15)
    assert np.allclose(b1["E2"], 0)
    assert np.isclose(b2["E2"][0, 1], np.sqrt(2) / 3) and np.isclose(b3["E2"][0, 1], -np.sqrt(2) / 3)
