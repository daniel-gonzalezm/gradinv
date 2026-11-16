"""GL(3) skeleton and isotropic algebra.

Fast tests are numerical; tests marked `exact` repeat the check in sympy.
"""
import itertools

import numpy as np
import pytest
import sympy as sp

from gradinv import isotropic
from gradinv._tensor import R, TRIPLES, index, kd, rotation_27, tensor6
from gradinv.skeleton import embeddings, h2_basis, identity_V, sym_projector
from helpers import basis_V, is_zero, random_rotation

E = embeddings()
I = identity_V()
S = sym_projector()
ISO = isotropic.algebra()
U = ISO.units()
P3, P2 = U[("H3", "3", "3")], U[("H2", "2", "2")]
Pss, Psr, Prs, Prr = (U[("H1", a, b)] for a, b in (("s", "s"), ("s", "r"), ("r", "s"), ("r", "r")))


# ------------------------------------------------------------------ A skeleton
def test_embeddings_isometric_orthogonal_complete():
    Eall = np.hstack([E[k] for k in ("3", "s", "2", "r")])
    assert np.allclose(Eall.T @ Eall, np.eye(18), atol=1e-14)
    assert np.allclose(Eall @ Eall.T, I, atol=1e-14)


def test_skeleton_projectors():
    assert np.allclose(P3 + P2 + Pss + Prr, I, atol=1e-14)
    assert np.allclose(S, P3 + Pss, atol=1e-14)
    assert [np.linalg.matrix_rank(M) for M in (P3, P2, Pss, Prr)] == [7, 5, 3, 3]
    assert np.allclose(I @ I, I) and np.linalg.matrix_rank(I) == 18


@pytest.mark.exact
def test_skeleton_exact():
    Ex = embeddings(exact=True)
    for k in ("s", "r", "2", "3"):
        assert is_zero(Ex[k].T * Ex[k] - sp.eye(Ex[k].shape[1]))
    for a, b in itertools.combinations(("s", "r", "2", "3"), 2):
        assert is_zero(Ex[a].T * Ex[b])
    Ux = isotropic.algebra(exact=True).units()
    Ix, Sx = identity_V(exact=True), sym_projector(exact=True)
    P3x, P2x = Ux[("H3", "3", "3")], Ux[("H2", "2", "2")]
    assert is_zero(P3x - (Sx - Ux[("H1", "s", "s")]))
    assert is_zero(P2x - (Ix - Sx - Ux[("H1", "r", "r")]))
    assert is_zero(P3x + P2x + Ux[("H1", "s", "s")] + Ux[("H1", "r", "r")] - Ix)
    assert is_zero(Ix * Ix - Ix) and Ix.rank() == 18


# ------------------------------------------------------------------ B equivariance
def _h2_coords(X):
    return np.array([float(np.sum(np.array(B, dtype=float) * X)) for B in h2_basis()])


def test_equivariance():
    rng = np.random.default_rng(20261022)
    for _ in range(3):
        Q = random_rotation(rng)
        Q3 = rotation_27(Q)
        v = rng.normal(size=3)
        assert np.allclose(Q3 @ E["s"] @ v, E["s"] @ (Q @ v), atol=1e-13)
        assert np.allclose(Q3 @ E["r"] @ v, E["r"] @ (Q @ v), atol=1e-13)
        X = sum(c * np.array(B, dtype=float) for c, B in zip(rng.normal(size=5), h2_basis()))
        assert np.allclose(Q3 @ E["2"] @ _h2_coords(X), E["2"] @ _h2_coords(Q @ X @ Q.T), atol=1e-13)
        assert np.allclose(Q3 @ P3, P3 @ Q3, atol=1e-13)


def test_inversion_is_minus_identity_on_V():
    assert np.allclose(rotation_27(-np.eye(3)) @ I, -I, atol=1e-15)


# ------------------------------------------------------------------ C physical meaning
@pytest.mark.exact
def test_physical_meaning_of_coordinates():
    Ex = embeddings(exact=True)
    x = sp.symbols("x0:3")
    cq = sp.symbols("c0:18")
    monos = [x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2]]
    u = [sum(cq[6 * i + p] * monos[p] for p in range(6)) for i in range(3)]
    eps = [[(sp.diff(u[i], x[j]) + sp.diff(u[j], x[i])) / 2 for j in range(3)] for i in range(3)]
    eta = sp.Matrix([sp.diff(eps[i][j], x[m]) for (i, j, m) in TRIPLES])
    divu = sum(sp.diff(u[i], x[i]) for i in range(3))
    lap = [sum(sp.diff(u[i], x[k], 2) for k in range(3)) for i in range(3)]

    def curl(f):
        return [sp.diff(f[2], x[1]) - sp.diff(f[1], x[2]), sp.diff(f[0], x[2]) - sp.diff(f[2], x[0]),
                sp.diff(f[1], x[0]) - sp.diff(f[0], x[1])]
    omega = [c / 2 for c in curl(u)]
    gw = sp.Matrix(3, 3, lambda i, j: sp.diff(omega[i], x[j]))
    chi = (gw + gw.T) / 2
    vs = sp.Matrix([(lap[i] + 2 * sp.diff(divu, x[i])) / sp.sqrt(15) for i in range(3)])
    assert is_zero(Ex["s"].T * eta - vs)
    assert is_zero(Ex["r"].T * eta - sp.Matrix(curl(omega)) / sp.sqrt(3))
    chi_c = sp.Matrix([sum(chi[a, b] * B[a, b] for a in range(3) for b in range(3)) for B in h2_basis()])
    assert is_zero(Ex["2"].T * eta - sp.sqrt(R(2, 3)) * chi_c)
    assert sp.simplify(gw.trace()) == 0
    sym3 = sp.Matrix([sum(sp.diff(u[a], x[b], x[c]) for (a, b, c) in set(itertools.permutations((i, j, m))))
                      / len(set(itertools.permutations((i, j, m)))) for (i, j, m) in TRIPLES])
    assert is_zero(sym_projector(exact=True) * eta - sym3)


def test_dilatation_trace_sign_convention():
    tr_s = [sum(E["s"][index(p, p, m), 0] for p in range(3)) for m in range(3)]
    tr_r = [sum(E["r"][index(p, p, m), 0] for p in range(3)) for m in range(3)]
    assert np.allclose(tr_s, [np.sqrt(5 / 3), 0, 0]) and np.allclose(tr_r, [2 / np.sqrt(3), 0, 0])


# ------------------------------------------------------------------ D table
def _table_ok(Uk, tol):
    names = {"P3": ("H3", "3", "3"), "P2": ("H2", "2", "2"), "Pss": ("H1", "s", "s"),
             "Qsr": ("H1", "s", "r"), "Qrs": ("H1", "r", "s"), "Prr": ("H1", "r", "r")}
    for A, B in itertools.product(names, names):
        p, a, b = names[A]
        q, c, d = names[B]
        prod = Uk[names[A]] @ Uk[names[B]] if tol else Uk[names[A]] * Uk[names[B]]
        target = Uk[(p, a, d)] if (p == q and b == c) else 0 * prod
        if tol:
            if not np.allclose(prod, target, atol=tol):
                return False
        elif not is_zero(prod - target):
            return False
    return True


def test_isotropic_table_and_gram():
    assert _table_ok(U, 1e-14)
    keys = list(U)
    gram = np.array([[np.sum(U[a] * U[b]) for b in keys] for a in keys])
    assert np.allclose(gram, np.diag([7, 5, 3, 3, 3, 3]), atol=1e-12)


@pytest.mark.exact
def test_isotropic_table_exact():
    assert _table_ok(isotropic.algebra(exact=True).units(), None)


# ------------------------------------------------------------------ H inverse
def test_inverse_and_positive_definiteness_random():
    rng = np.random.default_rng(20261022)
    Bv = basis_V()
    agree = 0
    for _ in range(400):
        a = rng.normal(size=5)
        D = isotropic.lazar2022_tensor(a)
        blk = ISO.moduli(D)
        assert np.allclose(ISO.assemble(blk), D, atol=1e-12)
        D18 = Bv.T @ D @ Bv
        ev = np.linalg.eigvalsh(D18)
        agree += (ev.min() > 0) == ISO.is_positive_definite(blk)
        assert np.allclose(np.sort(ev), ISO.spectrum(blk), atol=1e-12)
        if ISO.is_invertible(blk, tol=1e-8):
            Sinv = ISO.invert(D)
            assert np.allclose(Sinv @ D, I, atol=1e-9 * max(1, np.abs(Sinv).max()))
    assert agree == 400


@pytest.mark.exact
def test_inverse_exact():
    a = sp.symbols("a1:6")
    alg = isotropic.algebra(exact=True)
    D = isotropic.lazar2022_tensor(a, exact=True)
    Sinv = alg.assemble(alg.inverse(alg.moduli(D)))
    Ix = identity_V(exact=True)
    assert is_zero(Sinv * D - Ix) and is_zero(D * Sinv - Ix)


# ------------------------------------------------------------------ J limits
def test_identity_limit():
    blk = ISO.moduli(I)
    assert np.allclose(blk["H3"], 1) and np.allclose(blk["H2"], 1) and np.allclose(blk["H1"], np.eye(2))
    assert np.allclose(isotropic.to_lazar2022(blk), [0, 0, 0, 0.5, 0])


@pytest.mark.exact
def test_helmholtz_separable_limit():
    """Lazar (2016), eq. (86): a1 = a3 = a5 = 0 gives ell^2 C x delta."""
    lam, mu, ell = sp.symbols("lambda mu ell", positive=True)
    c1, c2 = 2 * mu, 3 * lam + 2 * mu
    a = [0, ell**2 * (c2 - c1) / 6, 0, ell**2 * c1 / 2, 0]
    D = isotropic.lazar2022_tensor(a, exact=True)
    Dsep = tensor6(lambda i, j, m, k, l, n: ell**2 * (lam * kd(i, j) * kd(k, l) + mu * (
        kd(i, k) * kd(j, l) + kd(i, l) * kd(j, k))) * kd(m, n))
    assert is_zero(D - Dsep)
    blk = isotropic.algebra(exact=True).moduli(D)
    assert sp.simplify(blk["H3"][0, 0] - 2 * mu * ell**2) == 0
    assert sp.simplify(blk["H2"][0, 0] - 2 * mu * ell**2) == 0
    M = blk["H1"]
    assert sp.simplify(M.trace() - (3 * lam + 4 * mu) * ell**2) == 0
    assert sp.simplify(M.det() - 2 * mu * (3 * lam + 2 * mu) * ell**4) == 0
    num = M.subs({lam: 1, mu: 1, ell: 1})
    assert num == sp.Matrix([[R(11, 3), 2 * sp.sqrt(5) / 3], [2 * sp.sqrt(5) / 3, R(10, 3)]])


def test_helmholtz_separable_numeric():
    lam, mu, ell = 1.3, 0.7, 1.1
    D = isotropic.lazar2022_tensor([0, ell**2 * 3 * lam / 6, 0, ell**2 * mu, 0])
    blk = ISO.moduli(D)
    assert np.isclose(blk["H3"][0, 0], 2 * mu * ell**2) and np.isclose(blk["H2"][0, 0], 2 * mu * ell**2)
    assert np.allclose(np.linalg.eigvalsh(blk["H1"]), sorted([2 * mu * ell**2, (3 * lam + 2 * mu) * ell**2]))
