"""Voigt, Reuss and Hill orientation averages in block moduli.

Principle. The uniform orientation average of an element X of a crystal
algebra (symmetric or not) is its orthogonal (Frobenius) projection on the
isotropic algebra (Krause and Boehlke 2025, sec. 4.2, for fourth order;
checked here for sixth order against direct orientation sums). Each isotropic matrix unit is a
sum of crystal units (cubic: P3 = U^T1_hh + U^T2_33 + U^A2, P2 = U^T2_22 + U^E,
Phi_ab = U^T1_ab; hexagonal: analogous), so with a modulus being
<X, U^p_ab>/d_p the isotropic moduli of the average are means of crystal block
entries weighted by the dimensions of the irreps in the branching of each H_l:

    m_l  = sum_p d_p/(2l+1) X^p_cc     (l = 3, 2; c = the copy of H_l in p)
    M_ab = sum_p d_p/3      X^p_ab     (a, b in {s, r})

The weights add up to one; couplings between copies of different H_l do not
survive. Here this "restriction" of the isotropic units to the crystal blocks
is computed from the algebras themselves (`restriction`), not transcribed.

    Voigt  D_V = <D>                       the rule applied to the blocks of D
    Reuss  D_R = <D^-1>^-1                 the rule applied to the block inverses,
                                           then each isotropic block inverted
    Hill   D_H = (D_V + D_R)/2

Reuss and Hill are formed only when the single-crystal D is positive definite
on V, checked block by
block; Fe and W are rejected. They are algebraic estimates (uniform double
stress), not bounds.

Cubic closed forms, with M = T1 (copies h, s, r),
N = T2 (copies 3, 2), m_E, m_A:
    1/m3^R = (3 (M^-1)_hh + 3 (N^-1)_33 + 1/m_A)/7
    1/m2^R = (3 (N^-1)_22 + 2/m_E)/5
    M^R    = M_(sr) - m_h m_h^T / M_hh   (Schur complement of M_hh; m_h = (M_hs, M_hr))
and m2^R/m_E = (5/2)/(1 + (3/2) m_E (N^-1)_22) -> 5/2 as m_E -> 0 (Cu MEAM).

Results are isotropic blocks {'H3': 1x1, 'H2': 1x1, 'H1': 2x2} as in
`gradinv.isotropic`; the constants a1..a5 of Lazar et al. (2022) follow from
`isotropic.to_lazar2022`.
"""
import numpy as np
import sympy as sp

from ._tensor import to_float
from .algebra import BlockAlgebra
from . import isotropic

__all__ = ["ISOTROPIC_KEYS", "restriction", "weights", "isotropic_part", "embed_isotropic",
           "voigt", "reuss", "hill", "reuss_cubic_closed", "reuss_mE_ratio"]

# isotropic matrix units, in the order (irrep, copy, copy)
ISOTROPIC_KEYS = (("H3", "3", "3"), ("H2", "2", "2"), ("H1", "s", "s"), ("H1", "s", "r"),
                  ("H1", "r", "s"), ("H1", "r", "r"))
_ISO_DIM = {"H3": 7, "H2": 5, "H1": 3}
_ISO_POS = {("H3", "3", "3"): (0, 0), ("H2", "2", "2"): (0, 0), ("H1", "s", "s"): (0, 0),
            ("H1", "s", "r"): (0, 1), ("H1", "r", "s"): (1, 0), ("H1", "r", "r"): (1, 1)}


def _is_sym(blocks):
    return any(isinstance(B, sp.MatrixBase) for B in blocks.values())


def _float_twin(alg):
    if not alg.exact:
        return alg
    emb = {p: {a: to_float(alg.embedding(p, a)) for a in copies} for p, copies, _d in alg.irreps}
    return BlockAlgebra(alg.name, alg.irreps, emb, exact=False)


_RESTRICTION_CACHE = {}


def _compute_restriction(alg):
    alg = _float_twin(alg)
    iso_units = isotropic.algebra().units()
    out = {}
    for k in ISOTROPIC_KEYS:
        blocks = alg.moduli(iso_units[k])
        ints = {}
        for p, B in blocks.items():
            Bi = np.rint(B)
            if np.abs(B - Bi).max() > 1e-10:
                raise ValueError(f"isotropic unit {k} is not a 0/1 combination of {alg.name} units")
            Bi = Bi.astype(int)
            Bi.setflags(write=False)
            ints[p] = Bi
        if np.abs(alg.assemble(blocks) - iso_units[k]).max() > 1e-12:
            raise ValueError(f"isotropic unit {k} does not lie in the {alg.name} algebra")
        out[k] = ints
    return out


def restriction(alg):
    """Isotropic matrix units written in the blocks of the crystal algebra `alg`.

    Returns {iso_key: {irrep: integer matrix}} with iso_key in ISOTROPIC_KEYS;
    every entry is 0 or 1 (the isotropic copies are crystal copies). Rotating
    the crystal does not change it, so it is cached by (name, irreps).
    """
    key = (alg.name, alg.irreps)
    if key not in _RESTRICTION_CACHE:
        _RESTRICTION_CACHE[key] = _compute_restriction(alg)
    return _RESTRICTION_CACHE[key]


def weights(alg):
    """Weights d_p/(2l+1) of each crystal irrep in the isotropic pieces.

    Returns {'H3': {p: w}, 'H2': {p: w}, 'H1': {p: w}} with exact rationals.
    """
    res = restriction(alg)
    dims = {p: d for p, _c, d in alg.irreps}
    out = {}
    for piece, key in (("H3", ("H3", "3", "3")), ("H2", ("H2", "2", "2")), ("H1", ("H1", "s", "s"))):
        out[piece] = {p: sp.Rational(dims[p], _ISO_DIM[piece])
                      for p, B in res[key].items() if B.any()}
    return out


def isotropic_part(alg, blocks):
    """Isotropic blocks {'H3', 'H2', 'H1'} of the orientation average of X.

    blocks: {irrep: square matrix} of an element X of the crystal algebra
    (symmetric or not; sympy or numpy). This is the weighted-mean rule.
    """
    res = restriction(alg)
    dims = {p: d for p, _c, d in alg.irreps}
    sym = _is_sym(blocks)
    vals = {}
    for k in ISOTROPIC_KEYS:
        tot = 0
        for p, C in res[k].items():
            if not C.any():
                continue
            X = blocks[p]
            if sym:
                tot += sp.Rational(dims[p], _ISO_DIM[k[0]]) * sum(
                    int(C[i, j]) * X[i, j] for i in range(C.shape[0]) for j in range(C.shape[1]) if C[i, j])
            else:
                tot += dims[p] / _ISO_DIM[k[0]] * float(np.sum(C * np.asarray(X, float)))
        vals[k] = tot
    mk = sp.Matrix if sym else (lambda x: np.array(x, dtype=float))
    return {"H3": mk([[vals[("H3", "3", "3")]]]), "H2": mk([[vals[("H2", "2", "2")]]]),
            "H1": mk([[vals[("H1", "s", "s")], vals[("H1", "s", "r")]],
                      [vals[("H1", "r", "s")], vals[("H1", "r", "r")]]])}


def embed_isotropic(alg, iso):
    """Blocks, in the crystal algebra `alg`, of the isotropic element `iso`.

    iso: isotropic blocks {'H3', 'H2', 'H1'} (as from `isotropic_part`).
    Cubic: T1 = m3 (+) M (copies h | s, r), T2 = diag(m3, m2), E = m2, A2 = m3.
    """
    res = restriction(alg)
    sym = _is_sym(iso)
    out = {}
    for p, copies, _d in alg.irreps:
        n = len(copies)
        out[p] = sp.zeros(n, n) if sym else np.zeros((n, n))
    for k in ISOTROPIC_KEYS:
        i, j = _ISO_POS[k]
        v = iso[k[0]][i, j]
        if v == 0:
            continue
        for p, C in res[k].items():
            if C.any():
                out[p] = out[p] + (v * sp.Matrix(C) if sym else float(v) * C)
    return out


def _check_pd(blocks):
    try:
        bad = BlockAlgebra.non_positive_blocks(blocks)
    except (TypeError, ValueError) as exc:
        raise ValueError("cannot check positive definiteness of symbolic blocks; "
                         "pass check_pd=False") from exc
    if bad:
        raise ValueError(f"Reuss and Hill need a positive definite single-crystal D; non-positive blocks: {bad}")


def voigt(alg, blocks):
    """Voigt estimate <D>: isotropic blocks of the average of D."""
    return isotropic_part(alg, blocks)


def reuss(alg, blocks, check_pd=True):
    """Reuss estimate <D^-1>^-1; refuses a D that is not positive definite on V."""
    if check_pd:
        _check_pd(blocks)
    return BlockAlgebra.inverse(isotropic_part(alg, BlockAlgebra.inverse(blocks)))


def hill(alg, blocks, check_pd=True):
    """Hill estimate (D_V + D_R)/2."""
    v, r = voigt(alg, blocks), reuss(alg, blocks, check_pd=check_pd)
    return {k: (v[k] + r[k]) / 2 for k in v}


def reuss_cubic_closed(blocks):
    """Reuss moduli of a cubic D written out (same result as `reuss`).

    1/m3^R = (3 (M^-1)_hh + 3 (N^-1)_33 + 1/m_A)/7,  1/m2^R = (3 (N^-1)_22 + 2/m_E)/5,
    M^R = M_(sr) - m_h m_h^T / M_hh  (Schur complement of M_hh in T1).
    """
    M, Nn, mE, mA = blocks["T1"], blocks["T2"], blocks["E"][0, 0], blocks["A2"][0, 0]
    sym = isinstance(M, sp.MatrixBase)
    if sym:
        Mi, Ni = M.inv(), Nn.inv()
        mh = M[1:, 0]
        MR = M[1:, 1:] - mh * mh.T / M[0, 0]
        m3 = 7 / (3 * Mi[0, 0] + 3 * Ni[0, 0] + 1 / mA)
        m2 = 5 / (3 * Ni[1, 1] + 2 / mE)
        return {"H3": sp.Matrix([[m3]]), "H2": sp.Matrix([[m2]]), "H1": MR}
    M, Nn = np.asarray(M, float), np.asarray(Nn, float)
    Mi, Ni = np.linalg.inv(M), np.linalg.inv(Nn)
    mh = M[1:, 0:1]
    MR = M[1:, 1:] - mh @ mh.T / M[0, 0]
    m3 = 7 / (3 * Mi[0, 0] + 3 * Ni[0, 0] + 1 / mA)
    m2 = 5 / (3 * Ni[1, 1] + 2 / mE)
    return {"H3": np.array([[m3]]), "H2": np.array([[m2]]), "H1": MR}


def reuss_mE_ratio(blocks):
    """m2^R / m_E = (5/2) / (1 + (3/2) m_E (N^-1)_22) for a cubic D (-> 5/2 as m_E -> 0).

    The 5/2 is 5/dim E: the E part of H2 takes 2 of its 5 dimensions and, being
    nearly zero, dominates the harmonic mean (Cu MEAM).
    """
    mE, Nn = blocks["E"][0, 0], blocks["T2"]
    if isinstance(Nn, sp.MatrixBase):
        return sp.Rational(5, 2) / (1 + sp.Rational(3, 2) * mE * Nn.inv()[1, 1])
    return 2.5 / (1 + 1.5 * mE * np.linalg.inv(np.asarray(Nn, float))[1, 1])
