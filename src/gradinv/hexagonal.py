"""Hexagonal algebra on V (classes 6/mmm / [D6]; dimension 32, 22 symmetric).

Structure tensors: the hexagonal axis n and
    h = Re (a + i n x a)^{x3},   with a along an a-axis (C2' family through a),
    k = sym(n x h) = Im (a + i n x a)^{x3}   (h rotated by -30 deg about n),
    H6 = Re (a + i n x a)^{x6}  (sixth-order structure tensor, a harmonic of degree 6),
where (n x T)_ijm = e_ipq n_p T_qjm.  |h|^2 = |k|^2 = 4.

Branchings SO(3) -> D6 (Mulliken labels, multiplicity free for l <= 3):
    H3 -> A2 + B1 + B2 + E1 + E2,   H2 -> A1 + E1 + E2,   H1 -> A2 + E1
so V = A1 + 3 A2 + B1 + B2 + 4 E1 + 2 E2 and the algebra is
R + M3 + R + R + M4 + M2  (1 + 9 + 1 + 1 + 16 + 4 = 32; symmetric 22).
Under D_inf (transverse isotropy) B1 and B2 merge: 31 (21 symmetric).

Isometric embeddings (H(.) = harmonic part of the totally symmetric part):
    A1, copy 2 (H2):  E_2 X0,  X0 = sqrt(3/2) (n n - d/3)
    A2, copy 3 (H3):  sqrt(5/2) H(n n n);   copies s, r: E_s n, E_r n
    B1 (H3): h/2;     B2 (H3): k/2
    E1, copy 3 (H3):  (sqrt15/2) H(n n v);  copy 2 (H2): E_2[sqrt2 sym(n x (n x v))];
        copies s, r:  E_s v, E_r v          (v orthogonal to n)
    E2, copy 3 (H3):  sqrt3 sym(n x (n x X));  copy 2 (H2): E_2 X  (X n = 0, traceless)
Order of the copies: A2 = (3, s, r), E1 = (3, 2, s, r), E2 = (3, 2).
Block names of the paper: A2 = A (3 x 3), E1 = F (4 x 4), E2 = G (2 x 2),
m_A1, m_B1, m_B2.

The extra modulus: U_B1 = h x h/4, U_B2 = k x k/4, U_B1 - U_B2 = H6/4, so
    Delta = m_B1 - m_B2 = D :: H6 / 4 = 4 eta   (eta of Auffray et al. 2013, same a-axis).
A rotation by 30 deg about n swaps B1 and B2.
"""
import itertools
from functools import lru_cache

import numpy as np
import sympy as sp

from ._tensor import N, R, TRIPLES, index, kd, to_float
from .algebra import BlockAlgebra
from .skeleton import embed_h2, embed_vector_r, embed_vector_s

__all__ = ["IRREPS", "MODULI", "structure_tensors", "algebra", "algebra_from_axes",
           "extra_modulus", "blocks_to_vector", "vector_to_blocks", "d6_group",
           "AUFFRAY2013_PARAMS", "auffray2013_tensor", "from_auffray2013", "to_auffray2013"]

IRREPS = (("A1", ("2",), 1), ("A2", ("3", "s", "r"), 1), ("B1", ("3",), 1), ("B2", ("3",), 1),
          ("E1", ("3", "2", "s", "r"), 2), ("E2", ("3", "2"), 2))


def _moduli_names():
    out = []
    for p, copies, _d in IRREPS:
        if len(copies) == 1:
            out.append((f"m{p}", p, 0, 0))
            continue
        for i in range(len(copies)):
            for j in range(i, len(copies)):
                out.append((f"{p}_{copies[i]}{copies[j]}", p, i, j))
    return tuple(out)


# symmetric moduli: mA1, A2_33, A2_3s, ..., mB1, mB2, E1_33, ..., E2_22 (22)
MODULI = _moduli_names()


# ---------------------------------------------------------------- tensor helpers (exact)
def _lc(i, j, k):
    return sp.LeviCivita(i, j, k)


def _complex_axis(n, a):
    n = [sp.sympify(x) for x in n]
    a = [sp.sympify(x) for x in a]
    b = [sum(_lc(i, j, k) * n[j] * a[k] for j in range(3) for k in range(3)) for i in range(3)]
    return n, a, b, [a[i] + sp.I * b[i] for i in range(3)]


def _h_tensor(z):
    return {t: sp.expand(sp.re(sp.expand(z[t[0]] * z[t[1]] * z[t[2]]))) for t in TRIPLES}


def _sym3(T):
    return {(i, j, m): sum(T[p] for p in itertools.permutations((i, j, m))) / 6
            for (i, j, m) in TRIPLES}


def _cross_first(n, T):
    return {(i, j, m): sum(_lc(i, p, q) * n[p] * T[(q, j, m)] for p in range(3) for q in range(3))
            for (i, j, m) in TRIPLES}


def _harmonic(T):
    S = _sym3(T)
    t = [sum(S[(p, p, m)] for p in range(3)) for m in range(3)]
    return {(i, j, m): S[(i, j, m)] - (kd(i, j) * t[m] + kd(i, m) * t[j] + kd(j, m) * t[i]) / 5
            for (i, j, m) in TRIPLES}


def _col(T, scale=1):
    out = sp.zeros(N, 1)
    for t in TRIPLES:
        out[index(*t)] = scale * T[t]
    return out


def _outer3(a, b, c):
    return {(i, j, m): a[i] * b[j] * c[m] for (i, j, m) in TRIPLES}


def _cross(a, b):
    return [sum(_lc(i, j, k) * a[j] * b[k] for j in range(3) for k in range(3)) for i in range(3)]


def _sym_outer(a, b):
    return sp.Matrix(3, 3, lambda i, j: (a[i] * b[j] + a[j] * b[i]) / 2)


def _cross_mat(n, X):
    return sp.Matrix(3, 3, lambda j, m: sum(_lc(j, p, q) * n[p] * X[q, m]
                                            for p in range(3) for q in range(3)))


def _outer_vec_mat(n, X):
    return {(i, j, m): n[i] * X[j, m] for (i, j, m) in TRIPLES}


def _embeddings_exact(n, a):
    n, a, b, z = _complex_axis(n, a)
    h = _h_tensor(z)
    k = _sym3(_cross_first(n, h))
    vb = [a, b]
    Xb = [sp.Matrix(3, 3, lambda i, j: (a[i] * a[j] - b[i] * b[j]) / sp.sqrt(2)),
          sp.Matrix(3, 3, lambda i, j: (a[i] * b[j] + b[i] * a[j]) / sp.sqrt(2))]
    nn = sp.Matrix(3, 3, lambda i, j: n[i] * n[j])
    X0 = sp.sqrt(R(3, 2)) * (nn - sp.eye(3) / 3)
    emb = {
        "A1": {"2": embed_h2(X0)},
        "A2": {"3": _col(_harmonic(_outer3(n, n, n)), sp.sqrt(R(5, 2))),
               "s": embed_vector_s(n), "r": embed_vector_r(n)},
        "B1": {"3": _col(h, R(1, 2))},
        "B2": {"3": _col(k, R(1, 2))},
        "E1": {"3": sp.Matrix.hstack(*[_col(_harmonic(_outer3(n, n, v)), sp.sqrt(15) / 2) for v in vb]),
               "2": sp.Matrix.hstack(*[embed_h2(sp.sqrt(2) * _sym_outer(n, _cross(n, v))) for v in vb]),
               "s": sp.Matrix.hstack(*[embed_vector_s(v) for v in vb]),
               "r": sp.Matrix.hstack(*[embed_vector_r(v) for v in vb])},
        "E2": {"3": sp.Matrix.hstack(*[_col(_sym3(_outer_vec_mat(n, _cross_mat(n, X))), sp.sqrt(3))
                                       for X in Xb]),
               "2": sp.Matrix.hstack(*[embed_h2(X) for X in Xb])},
    }
    return emb


def structure_tensors(n=(0, 0, 1), a=(1, 0, 0), exact=False):
    """(h, k, H6) for axis n and a-axis a: h, k as 3 x 3 x 3 arrays, H6 as 27 x 27.

    With exact=True, h and k are dicts {(i, j, m): value} and H6 a sympy matrix.
    """
    _n, _a, _b, z = _complex_axis(n, a)
    h = _h_tensor(z)
    k = _sym3(_cross_first(_n, h))
    H6 = sp.Matrix(N, N, lambda r, c: sp.expand(sp.re(sp.expand(
        sp.prod([z[t] for t in TRIPLES[r] + TRIPLES[c]])))))
    if exact:
        return h, k, H6
    arr = lambda T: np.array([float(T[t]) for t in TRIPLES]).reshape(3, 3, 3)
    return arr(h), arr(k), to_float(H6)


def algebra_from_axes(n, a, exact=True):
    """Hexagonal BlockAlgebra built from the formulas for axis n and a-axis a.

    n and a must be orthonormal; the entries may be sympy numbers (exact) or floats.
    This is the coordinate-free construction (no frame enters the formulas).
    """
    emb = _embeddings_exact(n, a)
    if exact:
        emb = {p: {c: sp.ImmutableMatrix(M) for c, M in cs.items()} for p, cs in emb.items()}
    else:
        emb = {p: {c: to_float(M) for c, M in cs.items()} for p, cs in emb.items()}
    return BlockAlgebra("hexagonal", IRREPS, emb, exact)


@lru_cache(maxsize=None)
def _algebra(exact):
    return algebra_from_axes((0, 0, 1), (1, 0, 0), exact=exact)


def algebra(exact=False, rotation=None):
    """The hexagonal BlockAlgebra with n = e3 and the a-axis along e1 (or rotated by Q)."""
    alg = _algebra(bool(exact))
    return alg if rotation is None else alg.rotated(rotation)


def extra_modulus(blocks):
    """Delta = m_B1 - m_B2, the single modulus absent from transverse isotropy."""
    return blocks["B1"][0, 0] - blocks["B2"][0, 0]


def blocks_to_vector(blocks):
    """The 22 symmetric moduli in the order of MODULI."""
    return [blocks[p][i, j] for _n, p, i, j in MODULI]


def vector_to_blocks(v, exact=False):
    out = {p: (sp.zeros(len(c), len(c)) if exact else np.zeros((len(c), len(c))))
           for p, c, _d in IRREPS}
    for x, (_n, p, i, j) in zip(v, MODULI):
        out[p][i, j] = x
        out[p][j, i] = x
    return out


# ---------------------------------------------------------------- group
def d6_group(exact=False):
    """The 12 rotations of D6 (axis e3; binary axes every 30 deg from e1)."""
    def rz(c, s):
        return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def half(c, s):
        u = sp.Matrix([c, s, 0])
        return 2 * u * u.T - sp.eye(3)
    r3 = sp.sqrt(3) / 2
    h_ = R(1, 2)
    rots = [rz(c, s) for c, s in ((1, 0), (h_, r3), (-h_, r3), (-1, 0), (-h_, -r3), (h_, -r3))]
    axes = [half(c, s) for c, s in ((1, 0), (r3, h_), (h_, r3), (0, 1), (-h_, r3), (-r3, h_))]
    mats = rots + axes
    return mats if exact else [to_float(Q) for Q in mats]


# ---------------------------------------------------------------- Auffray et al. (2013)
# Auffray, Le Quang and He, J. Mech. Phys. Solids 61:1202-1223 (2013), transcribed
# from the rendered pages: orthonormal basis eq. (10), Table 2 (p. 1206), and
# A_D6 = diag(A11 + eta Ac, A11, H6 + f(J4), J4) (Sect. 3.3.5, p. 1211; A11 p. 1209,
# H6 and Ac p. 1210, J4 p. 1208, f(J4) p. 1211). Same e1 as ours (Delta = 4 eta).
AUFFRAY2013_TABLE2 = ("111", "221", "122", "331", "133", "222", "112", "121", "332", "233",
                      "333", "113", "131", "223", "232", "123", "132", "231")
AUFFRAY2013_PARAMS = sp.symbols("a11 a12 a13 a14 a15 a22 a34 a35 a44 a45 a55 eta "
                                "h11 h12 h13 h22 h23 h33 j11 j12 j22 j23")


def auffray2013_basis():
    """27 x 18 matrix of their orthonormal basis vectors (eq. 10, Table 2 order)."""
    B = sp.zeros(N, 18)
    for col, s in enumerate(AUFFRAY2013_TABLE2):
        i, j, k = (int(c) - 1 for c in s)
        c = R(1, 2) if i == j else 1 / sp.sqrt(2)
        B[index(i, j, k), col] += c
        B[index(j, i, k), col] += c
    return B


def _sym_from_upper(rows, n):
    M = sp.zeros(n, n)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            M[r, r + c] = v
            M[r + c, r] = v
    return M


def auffray2013_blocks(p=AUFFRAY2013_PARAMS):
    """Their blocks (A11, eta Ac, H6, f(J4), J4) as printed."""
    (a11, a12, a13, a14, a15, a22, a34, a35, a44, a45, a55, eta,
     h11, h12, h13, h22, h23, h33, j11, j12, j22, j23) = p
    s2 = sp.sqrt(2)
    aI, aII = a14 - s2 * a34, a15 - s2 * a35
    aIII, aIIIs = (a11 - a22) / 2, (a11 + a22) / 2
    A11 = _sym_from_upper([[a11, a12, a13, a14, a15],
                           [a22, -a13 + s2 * aIII, aI, aII],
                           [-a12 + aIIIs, a34, a35],
                           [a44, a45],
                           [a55]], 5)
    Ac = _sym_from_upper([[1, -1, -s2, 0, 0], [1, s2, 0, 0], [2, 0, 0], [0, 0], [0]], 5)
    H6 = _sym_from_upper([[h11, h12, h13, h12, h13], [h22, h23, h22, h23], [h33, h23, h33],
                          [h22, h23], [h33]], 5)
    J4 = _sym_from_upper([[j11, j12, j12], [j22, j23], [j22]], 3)
    fJ = _sym_from_upper([[0, 0, 0, 0, 0], [0, 0, -j11, -s2 * j12], [0, -s2 * j12, -(j22 + j23)],
                          [0, 0], [0]], 5)
    return A11, eta * Ac, H6, fJ, J4


def auffray2013_matrix(p=AUFFRAY2013_PARAMS):
    """Their 18 x 18 matrix A_D6 (p. 1211)."""
    A11, etaAc, H6, fJ, J4 = auffray2013_blocks(p)
    return sp.diag(A11 + etaAc, A11, H6 + fJ, J4)


def auffray2013_tensor(p=AUFFRAY2013_PARAMS, exact=None):
    """27 x 27 tensor of A_D6 for the 22 parameters p (sympy if any entry is symbolic)."""
    if exact is None:
        exact = any(isinstance(x, sp.Basic) and not x.is_number for x in p)
    B = auffray2013_basis()
    T = B * auffray2013_matrix([sp.sympify(x) for x in p]) * B.T
    return T if exact else to_float(T)


@lru_cache(maxsize=None)
def _auffray_map_exact():
    """22 x 22 matrix J with (our moduli) = J (their parameters); det 2048 sqrt2."""
    alg = algebra(exact=True)
    b = blocks_to_vector(alg.moduli(auffray2013_tensor(AUFFRAY2013_PARAMS, exact=True)))
    return sp.ImmutableMatrix(sp.Matrix(b).jacobian(AUFFRAY2013_PARAMS))


@lru_cache(maxsize=None)
def _auffray_map_float():
    J = to_float(_auffray_map_exact())
    return J, np.linalg.inv(J)


def from_auffray2013(params, exact=False):
    """Blocks of the hexagonal tensor with the 22 parameters of Auffray et al. (2013)."""
    if exact:
        return vector_to_blocks(list(_auffray_map_exact() * sp.Matrix(list(params))), exact=True)
    return vector_to_blocks(_auffray_map_float()[0] @ np.asarray(params, float))


def to_auffray2013(blocks, exact=False):
    """The 22 parameters of Auffray et al. (2013) (order AUFFRAY2013_PARAMS) of a symmetric element."""
    v = blocks_to_vector(blocks)
    if exact:
        return list(sp.simplify(_auffray_map_exact().inv() * sp.Matrix(v)))
    return _auffray_map_float()[1] @ np.asarray(v, float)
