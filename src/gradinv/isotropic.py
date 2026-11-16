"""Isotropic algebra on V (dimension 6; 5 symmetric elements).

Blocks: 'H3' (1 x 1, modulus m3), 'H2' (1 x 1, m2) and 'H1' (2 x 2, M, copies
(s, r) of the GL(3) chain). Units:
    P3 = S - Phi_ss,  P2 = I - S - Phi_rr,  Phi_ab = E_a E_b^T  (a, b in {s, r}),
so  D = m3 P3 + m2 P2 + sum_ab M_ab Phi_ab  and, for any D, these moduli give
its orthogonal projection on the isotropic tensors (the uniform orientation
average). Inverse: P3/m3 + P2/m2 + sum (M^-1)_ab Phi_ab.

Change of constants with Lazar, Agiasofitou and Boehlke (2022), eq. (40),
isotropic part a1..a5 (exact, determinant -120 sqrt5):
    m3 = 2(a4 + a5),  m2 = 2 a4 - a5,
    Mss = 10/3 (a1 + a2 + a3) + 2 a4 + 2 a5,
    Msr = Mrs = sqrt5/3 (a1 + 4 a2 - 2 a3),
    Mrr = -4/3 a1 + 8/3 a2 + 2/3 a3 + 2 a4 - a5.
"""
from functools import lru_cache

import numpy as np
import sympy as sp

from ._tensor import R, kd, tensor6, to_float
from .algebra import BlockAlgebra
from .skeleton import embeddings

__all__ = ["IRREPS", "MODULI", "algebra", "lazar2022_tensor", "from_lazar2022",
           "to_lazar2022", "blocks_to_vector", "vector_to_blocks", "isotropic_blocks"]

IRREPS = (("H3", ("3",), 7), ("H2", ("2",), 5), ("H1", ("s", "r"), 3))

# symmetric moduli: (name, block, row, column)
MODULI = (("m3", "H3", 0, 0), ("m2", "H2", 0, 0), ("Mss", "H1", 0, 0),
          ("Msr", "H1", 0, 1), ("Mrr", "H1", 1, 1))


@lru_cache(maxsize=None)
def _algebra(exact):
    E = embeddings(exact)
    emb = {"H3": {"3": E["3"]}, "H2": {"2": E["2"]}, "H1": {"s": E["s"], "r": E["r"]}}
    return BlockAlgebra("isotropic", IRREPS, emb, exact)


def algebra(exact=False):
    """The isotropic BlockAlgebra (float arrays by default, sympy with exact=True)."""
    return _algebra(bool(exact))


def isotropic_blocks(m3, m2, M):
    """Blocks of the isotropic element m3 P3 + m2 P2 + sum M_ab Phi_ab."""
    if isinstance(M, sp.MatrixBase) or any(isinstance(x, sp.Basic) for x in (m3, m2)):
        return {"H3": sp.Matrix([[m3]]), "H2": sp.Matrix([[m2]]), "H1": sp.Matrix(M)}
    return {"H3": np.array([[m3]], float), "H2": np.array([[m2]], float),
            "H1": np.array(M, dtype=float)}


def blocks_to_vector(blocks):
    """(m3, m2, Mss, Msr, Mrr) of a symmetric element."""
    return [blocks[b][i, j] for _n, b, i, j in MODULI]


def vector_to_blocks(v, exact=False):
    m3, m2, mss, msr, mrr = v
    return isotropic_blocks(m3, m2, (sp.Matrix if exact else np.array)([[mss, msr], [msr, mrr]]))


# ---------------------------------------------------------------- Lazar et al. (2022)
def _lazar_iso(a1, a2, a3, a4, a5):
    def f(i, j, m, k, l, n):
        return (R(1, 2) * a1 * (kd(i, j) * kd(k, m) * kd(l, n) + kd(i, j) * kd(k, n) * kd(l, m)
                                + kd(k, l) * kd(i, m) * kd(j, n) + kd(k, l) * kd(i, n) * kd(j, m))
                + 2 * a2 * kd(i, j) * kd(k, l) * kd(m, n)
                + R(1, 2) * a3 * (kd(j, k) * kd(i, m) * kd(l, n) + kd(i, k) * kd(j, m) * kd(l, n)
                                  + kd(i, l) * kd(j, m) * kd(k, n) + kd(j, l) * kd(i, m) * kd(k, n))
                + a4 * (kd(i, l) * kd(j, k) * kd(m, n) + kd(i, k) * kd(j, l) * kd(m, n))
                + R(1, 2) * a5 * (kd(j, k) * kd(i, n) * kd(l, m) + kd(i, k) * kd(j, n) * kd(l, m)
                                  + kd(j, l) * kd(k, m) * kd(i, n) + kd(i, l) * kd(k, m) * kd(j, n)))
    return f


def lazar2022_tensor(a, exact=False):
    """Isotropic tensor of Lazar et al. (2022), eq. (40), with constants a = (a1..a5)."""
    a = list(a)
    if exact:
        return tensor6(_lazar_iso(*[sp.sympify(x) for x in a]), exact=True)
    return np.tensordot(np.asarray(a, dtype=float), _lazar_basis_float(), axes=1)


@lru_cache(maxsize=None)
def _lazar_basis_float():
    """The five tensors L1..L5 of eq. (40) (a_k = delta_kj), as a 5 x 27 x 27 array."""
    out = []
    for k in range(5):
        e = [0.0] * 5
        e[k] = 1.0
        out.append(tensor6(_lazar_iso(*e), exact=False))
    arr = np.array(out)
    arr.setflags(write=False)
    return arr


@lru_cache(maxsize=None)
def _lazar_map_exact():
    """5 x 5 matrix A with (m3, m2, Mss, Msr, Mrr) = A a (exact)."""
    alg = algebra(exact=True)
    cols = []
    for k in range(5):
        e = [0] * 5
        e[k] = 1
        cols.append(blocks_to_vector(alg.moduli(lazar2022_tensor(e, exact=True))))
    return sp.ImmutableMatrix(sp.Matrix(cols).T)


@lru_cache(maxsize=None)
def _lazar_map_float():
    A = to_float(_lazar_map_exact())
    return A, np.linalg.inv(A)


def from_lazar2022(a, exact=False):
    """Blocks of the isotropic tensor with constants a1..a5 of Lazar et al. (2022)."""
    if exact:
        return vector_to_blocks(list(_lazar_map_exact() * sp.Matrix(list(a))), exact=True)
    return vector_to_blocks(_lazar_map_float()[0] @ np.asarray(a, float))


def to_lazar2022(blocks, exact=False):
    """Constants a1..a5 of Lazar et al. (2022) of a symmetric isotropic element."""
    v = blocks_to_vector(blocks)
    if exact:
        return list(sp.simplify(_lazar_map_exact().inv() * sp.Matrix(v)))
    return _lazar_map_float()[1] @ np.asarray(v, float)
