"""Cubic algebra on V (classes m-3m / [O]; dimension 15, 11 symmetric).

Structure tensor: DD_ijkl = sum_s e^s_i e^s_j e^s_k e^s_l (e^s: cubic axes).
Branchings SO(3) -> O (multiplicity free for l <= 3):
    H3 -> A2 + T1 + T2,   H2 -> E + T2,   H1 -> T1
so V = 3 T1 + 2 T2 + E + A2 and the algebra is M3 + M2 + R + R (9 + 4 + 1 + 1).

Isometric embeddings (compositional form; model spaces: T1 vectors, T2
off-diagonal symmetric matrices, E diagonal traceless matrices, A2 a line):
    T1, copy h (H3):  E_h v = sqrt(5/2) [DD_ijml v_l - (v_i d_jm + v_j d_im + v_m d_ij)/5]
                      (harmonic part of DD.v, oriented along +DD.v)
    T1, copy s / r :  E_s v, E_r v (skeleton)
    T2, copy 3 (H3):  E_t X = sqrt3 sym_(ijm)[e_ilp X_pq DD_lqjm], oriented so that
                      the a8 tensor DD x delta has a positive coupling N_32 = sqrt2/3
    T2, copy 2 (H2):  E_2 X, X off-diagonal
    E        (H2):    E_2 X, X diagonal traceless
    A2       (H3):    sym(e1 e2 e3), normalised; the kernel in H3 of h -> DD_ijpq h_pqm
Order of the copies: decreasing l, S before R:  T1 = (h, s, r),  T2 = (3, 2).

Block names of the paper: T1 = M (3 x 3), T2 = N (2 x 2), E = m_E, A2 = m_A.
Inverse: sum (M^-1)_ab U^T1_ab + sum (N^-1)_ab U^T2_ab + U^E/m_E + U^A2/m_A;
invertible iff det M det N m_E m_A != 0; positive definite iff M, N > 0 and
m_E, m_A > 0. Spectrum: eig(M) x3, eig(N) x3, m_E x2, m_A x1.

Change of constants with Lazar et al. (2022), eq. (40), a1..a11: exact and
invertible (determinant -384 sqrt2); see `from_lazar2022`, `to_lazar2022`.
"""
import itertools
from functools import lru_cache

import numpy as np
import sympy as sp

from ._tensor import N, R, TRIPLES, index, kd, tensor6, to_float
from .algebra import BlockAlgebra
from .isotropic import _lazar_iso
from .skeleton import embeddings as skeleton_embeddings

__all__ = ["IRREPS", "MODULI", "structure_tensor", "algebra", "lazar2022_tensor",
           "a_from_voigt", "from_lazar2022", "to_lazar2022", "blocks_to_vector",
           "vector_to_blocks", "oh_group"]

IRREPS = (("T1", ("h", "s", "r"), 3), ("T2", ("3", "2"), 3), ("E", ("e",), 2), ("A2", ("a",), 1))

MODULI = (("Mhh", "T1", 0, 0), ("Mss", "T1", 1, 1), ("Mrr", "T1", 2, 2),
          ("Mhs", "T1", 0, 1), ("Mhr", "T1", 0, 2), ("Msr", "T1", 1, 2),
          ("N33", "T2", 0, 0), ("N22", "T2", 1, 1), ("N32", "T2", 0, 1),
          ("mE", "E", 0, 0), ("mA", "A2", 0, 0))


def _DD(i, j, k, l):
    return 1 if i == j == k == l else 0


def structure_tensor():
    """DD_ijkl = sum_s e^s_i e^s_j e^s_k e^s_l as a 3 x 3 x 3 x 3 array."""
    out = np.zeros((3, 3, 3, 3))
    for s in range(3):
        out[s, s, s, s] = 1.0
    return out


# ---------------------------------------------------------------- embeddings
def _embed_T1_h(v):
    out = sp.zeros(N, 1)
    for (i, j, m) in TRIPLES:
        dv = sum(_DD(i, j, m, l) * v[l] for l in range(3))
        tr = (v[i] * kd(j, m) + v[j] * kd(i, m) + v[m] * kd(i, j)) / 5
        out[index(i, j, m)] = sp.sqrt(R(5, 2)) * (dv - tr)
    return out


def _embed_T2_3(X):
    raw = {}
    for (i, j, m) in TRIPLES:
        raw[(i, j, m)] = sum(sp.LeviCivita(i, l, p) * X[p, q] * _DD(l, q, j, m)
                             for p in range(3) for q in range(3) for l in range(3))
    out = sp.zeros(N, 1)
    for (i, j, m) in TRIPLES:
        perms = list(itertools.permutations((i, j, m)))
        out[index(i, j, m)] = sp.sqrt(3) * sum(raw[p] for p in perms) / len(perms)
    return out


def _a2_vector():
    out = sp.zeros(N, 1)
    for p in itertools.permutations((0, 1, 2)):
        out[index(*p)] = 1 / sp.sqrt(6)
    return out


@lru_cache(maxsize=None)
def _algebra(exact):
    from .skeleton import h2_basis, vector_basis
    E = skeleton_embeddings(exact=True)
    hb = h2_basis()
    emb = {"T1": {"h": sp.Matrix.hstack(*[_embed_T1_h(v) for v in vector_basis()]),
                  "s": E["s"], "r": E["r"]},
           "T2": {"3": sp.Matrix.hstack(*[_embed_T2_3(X) for X in hb[:3]]),
                  "2": sp.Matrix(E["2"])[:, :3]},
           "E": {"e": sp.Matrix(E["2"])[:, 3:]},
           "A2": {"a": _a2_vector()}}
    if exact:
        emb = {p: {a: sp.ImmutableMatrix(M) for a, M in cs.items()} for p, cs in emb.items()}
    else:
        emb = {p: {a: to_float(M) for a, M in cs.items()} for p, cs in emb.items()}
    return BlockAlgebra("cubic", IRREPS, emb, exact)


def algebra(exact=False, rotation=None):
    """The cubic BlockAlgebra with cubic axes along e1, e2, e3 (or rotated by Q)."""
    alg = _algebra(bool(exact))
    return alg if rotation is None else alg.rotated(rotation)


def blocks_to_vector(blocks):
    """The 11 symmetric moduli (Mhh, Mss, Mrr, Mhs, Mhr, Msr, N33, N22, N32, mE, mA)."""
    return [blocks[b][i, j] for _n, b, i, j in MODULI]


def vector_to_blocks(v, exact=False):
    Mhh, Mss, Mrr, Mhs, Mhr, Msr, N33, N22, N32, mE, mA = v
    mk = sp.Matrix if exact else (lambda x: np.array(x, dtype=float))
    return {"T1": mk([[Mhh, Mhs, Mhr], [Mhs, Mss, Msr], [Mhr, Msr, Mrr]]),
            "T2": mk([[N33, N32], [N32, N22]]), "E": mk([[mE]]), "A2": mk([[mA]])}


# ---------------------------------------------------------------- Lazar et al. (2022)
def _lazar_cubic_extra(a6, a7, a8, a9, a10, a11):
    def dd(*idx):
        return 1 if len(set(idx)) == 1 else 0

    def f(i, j, m, k, l, n):
        return (a6 * (kd(i, k) * dd(j, l, m, n) + kd(i, l) * dd(j, k, m, n)
                      + kd(j, k) * dd(i, l, m, n) + kd(j, l) * dd(i, k, m, n))
                + a7 * (kd(k, m) * dd(i, j, l, n) + kd(l, m) * dd(i, j, k, n)
                        + kd(i, n) * dd(j, k, l, m) + kd(j, n) * dd(i, k, l, m))
                + a8 * kd(m, n) * dd(i, j, k, l)
                + a9 * (kd(i, j) * dd(k, l, m, n) + kd(k, l) * dd(i, j, m, n))
                + a10 * (kd(i, m) * dd(j, k, l, n) + kd(j, m) * dd(i, k, l, n)
                         + kd(k, n) * dd(i, j, l, m) + kd(l, n) * dd(i, j, k, m))
                + a11 * dd(i, j, k, l, m, n))
    return f


def lazar2022_tensor(a, exact=False):
    """Cubic tensor of Lazar, Agiasofitou and Boehlke (2022), eq. (40), a = (a1..a11)."""
    if not exact:
        return np.tensordot(np.asarray(list(a), dtype=float), _lazar_basis_float(), axes=1)
    a = [sp.sympify(x) for x in a]
    fi, fc = _lazar_iso(*a[:5]), _lazar_cubic_extra(*a[5:])
    return tensor6(lambda *t: fi(*t) + fc(*t), exact=True)


@lru_cache(maxsize=None)
def _lazar_basis_float():
    """The eleven tensors L1..L11 of eq. (40), as an 11 x 27 x 27 array."""
    out = []
    for k in range(11):
        e = [0.0] * 11
        e[k] = 1.0
        fi, fc = _lazar_iso(*e[:5]), _lazar_cubic_extra(*e[5:])
        out.append(tensor6(lambda *t: fi(*t) + fc(*t), exact=False))
    arr = np.array(out)
    arr.setflags(write=False)
    return arr


def a_from_voigt(v):
    """a1..a11 from the 11 Voigt components (D11, D12, D13, D22, D23, D24, D25, D33,
    D35, D1616, D1617) of Lazar et al. (2022), Table 1, by their eq. (60)."""
    D11, D12, D13, D22, D23, D24, D25, D33, D35, D1616, D1617 = v
    return np.array([
        2 * D25, D24 / 2, 2 * D35, D1616, 2 * D1617,
        D33 - D35 - D1616 - D1617, D23 - D25 - 2 * D1617,
        D22 - D24 - 2 * D1616, D12 - D24 - 2 * D25, D13 - D25 - 2 * D35,
        D11 - D22 + 2 * (D24 - D12) + 4 * (D1616 - D13 - D23 - D33)
        + 8 * (D25 + D35 + D1617)], dtype=float)


@lru_cache(maxsize=None)
def _lazar_map_exact():
    """11 x 11 matrix A with (Mhh, ..., mA) = A a (exact)."""
    alg = algebra(exact=True)
    cols = []
    for k in range(11):
        e = [0] * 11
        e[k] = 1
        cols.append(blocks_to_vector(alg.moduli(lazar2022_tensor(e, exact=True))))
    return sp.ImmutableMatrix(sp.Matrix(cols).T)


@lru_cache(maxsize=None)
def _lazar_map_float():
    A = to_float(_lazar_map_exact())
    return A, np.linalg.inv(A)


def from_lazar2022(a, exact=False):
    """Blocks of the cubic tensor with constants a1..a11 of Lazar et al. (2022)."""
    if exact:
        return vector_to_blocks(list(_lazar_map_exact() * sp.Matrix(list(a))), exact=True)
    return vector_to_blocks(_lazar_map_float()[0] @ np.asarray(a, float))


def to_lazar2022(blocks, exact=False):
    """Constants a1..a11 of Lazar et al. (2022) of a symmetric cubic element."""
    v = blocks_to_vector(blocks)
    if exact:
        return list(sp.simplify(_lazar_map_exact().inv() * sp.Matrix(v)))
    return _lazar_map_float()[1] @ np.asarray(v, float)


# ---------------------------------------------------------------- group
def oh_group(exact=False):
    """The 48 signed permutation matrices of O_h (the 24 with det +1 form O)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            Q = sp.zeros(3, 3)
            for r_, c_ in enumerate(perm):
                Q[r_, c_] = signs[r_]
            mats.append(Q if exact else to_float(Q))
    return mats
