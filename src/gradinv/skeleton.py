"""GL(3) skeleton of the strain-gradient space V.

Splitting (GL(3) chain):
    V = S (totally symmetric, dim 10) + R (remainder, dim 8)
    S = H3 + H1^s,   R = H2 + H1^r
Model spaces and isometric embeddings (each E is 27 x dim with E^T E = Id):
    H3   : totally symmetric traceless 3-tensors; E_3 = inclusion
           (any orthonormal basis; adapted bases are built by the crystal classes)
    H1^s : vectors v;  (E_s v)_ijm = (v_i d_jm + v_j d_im + v_m d_ij)/sqrt(15)
    H1^r : vectors w;  (E_r w)_ijm = (2 d_ij w_m - d_im w_j - d_jm w_i)/sqrt(12)
    H2   : symmetric traceless 2-tensors X;
           (E_2 X)_ijm = -(e_jml X_li + e_iml X_lj)/sqrt(6)
Sign conventions (resolve S3 for these copies):
    * both H1 copies are oriented so that the dilatation-gradient trace
      eta_ppm of the embedded tensor is a positive multiple of the vector;
    * H2 is oriented so that its coordinate is a positive multiple of the
      symmetric curvature chi = sym(grad omega), omega = curl(u)/2.
Physical meaning of the coordinates (tests/test_skeleton.py):
    E_s^T eta = (lap u + 2 grad div u)/sqrt(15)
    E_r^T eta = curl(omega)/sqrt(3)
    E_2^T eta = sqrt(2/3) chi
    E_3^T eta = coordinates of the totally symmetric traceless part of grad grad u
"""
import itertools
from functools import lru_cache

import numpy as np
import sympy as sp

from ._tensor import N, R, TRIPLES, index, kd, tensor6, to_float

__all__ = ["identity_V", "sym_projector", "embeddings", "h1_pair", "THETA_LAZAR",
           "vector_basis", "h2_basis", "embed_vector_s", "embed_vector_r", "embed_h2"]

THETA_LAZAR = sp.acos(sp.sqrt(5) / 3)  # 41.81 deg; cos = sqrt5/3, sin = 2/3


def _freeze(M, exact):
    if exact:
        return sp.ImmutableMatrix(M)
    A = to_float(M)
    A.setflags(write=False)
    return A


# ---------------------------------------------------------------- identities
@lru_cache(maxsize=None)
def _identity_exact():
    return sp.ImmutableMatrix(tensor6(
        lambda i, j, m, k, l, n: R(1, 2) * (kd(i, k) * kd(j, l) + kd(i, l) * kd(j, k)) * kd(m, n)))


@lru_cache(maxsize=None)
def _sym_exact():
    M = sp.zeros(N, N)
    for (i, j, m) in TRIPLES:
        for p in itertools.permutations((i, j, m)):
            M[index(i, j, m), index(*p)] += R(1, 6)
    return sp.ImmutableMatrix(M)


@lru_cache(maxsize=None)
def _float(name):
    return _freeze({"I": _identity_exact, "S": _sym_exact}[name](), exact=False)


def identity_V(exact=False):
    """I_ijmkln = (d_ik d_jl + d_il d_jk) d_mn / 2, the identity (projector) on V."""
    return _identity_exact() if exact else _float("I")


def sym_projector(exact=False):
    """Total symmetrizer (S eta)_ijm = eta_(ijm): orthogonal projector onto S."""
    return _sym_exact() if exact else _float("S")


# ---------------------------------------------------------------- model bases
def vector_basis():
    return [sp.Matrix([kd(a, b) for b in range(3)]) for a in range(3)]


def h2_basis():
    """Orthonormal (Frobenius) basis of symmetric traceless 3 x 3 matrices.

    The first three are off-diagonal, the last two diagonal (used by the cubic class).
    """
    def e(a, b):
        M = sp.zeros(3, 3)
        M[a, b] = 1
        return M
    s2 = sp.sqrt(2)
    return [(e(1, 2) + e(2, 1)) / s2, (e(0, 2) + e(2, 0)) / s2, (e(0, 1) + e(1, 0)) / s2,
            (e(0, 0) - e(1, 1)) / s2, (e(0, 0) + e(1, 1) - 2 * e(2, 2)) / sp.sqrt(6)]


# ---------------------------------------------------------------- embeddings
def embed_vector_s(v):
    """(E_s v)_ijm = (v_i d_jm + v_j d_im + v_m d_ij)/sqrt(15), as a 27 x 1 sympy column."""
    out = sp.zeros(N, 1)
    for (i, j, m) in TRIPLES:
        out[index(i, j, m)] = (v[i] * kd(j, m) + v[j] * kd(i, m) + v[m] * kd(i, j)) / sp.sqrt(15)
    return out


def embed_vector_r(w):
    """(E_r w)_ijm = (2 d_ij w_m - d_im w_j - d_jm w_i)/sqrt(12)."""
    out = sp.zeros(N, 1)
    for (i, j, m) in TRIPLES:
        out[index(i, j, m)] = (2 * kd(i, j) * w[m] - kd(i, m) * w[j] - kd(j, m) * w[i]) / sp.sqrt(12)
    return out


def embed_h2(X):
    """(E_2 X)_ijm = -(e_jml X_li + e_iml X_lj)/sqrt(6), X symmetric traceless."""
    out = sp.zeros(N, 1)
    for (i, j, m) in TRIPLES:
        s = 0
        for l in range(3):
            s += sp.LeviCivita(j, m, l) * X[l, i] + sp.LeviCivita(i, m, l) * X[l, j]
        out[index(i, j, m)] = -s / sp.sqrt(6)
    return out


@lru_cache(maxsize=None)
def _embeddings_exact():
    Es = sp.Matrix.hstack(*[embed_vector_s(v) for v in vector_basis()])
    Er = sp.Matrix.hstack(*[embed_vector_r(v) for v in vector_basis()])
    E2 = sp.Matrix.hstack(*[embed_h2(X) for X in h2_basis()])
    P3 = _sym_exact() - Es * Es.T
    cols = sp.GramSchmidt(P3.columnspace(), orthonormal=True)
    E3 = sp.Matrix.hstack(*[sp.simplify(c) for c in cols])
    return {k: sp.ImmutableMatrix(v) for k, v in (("s", Es), ("r", Er), ("2", E2), ("3", E3))}


@lru_cache(maxsize=None)
def _embeddings_float():
    return {k: _freeze(v, exact=False) for k, v in _embeddings_exact().items()}


def embeddings(exact=False):
    """Isometric embeddings {'s': E_s (27x3), 'r': E_r (27x3), '2': E_2 (27x5), '3': E_3 (27x7)}."""
    return dict(_embeddings_exact() if exact else _embeddings_float())


def h1_pair(phi=0, exact=False):
    """Rotated pair of H1 copies, (E_a, E_b) = (c E_s + s E_r, -s E_s + c E_r).

    phi = 0 gives the GL(3) pair (convention of the project). phi = THETA_LAZAR
    gives (E_(4), -E_(3)): Lazar's (2016) dilatation-gradient piece (4) and his
    shear-divergence piece (3) with the opposite orientation (his pieces, as
    printed in eqs. 53-54, differ from the GL(3) pair by a reflection).
    """
    E = embeddings(exact)
    if exact:
        c, s = sp.cos(phi), sp.sin(phi)
    else:
        c, s = np.cos(float(phi)), np.sin(float(phi))
    return c * E["s"] + s * E["r"], -s * E["s"] + c * E["r"]
