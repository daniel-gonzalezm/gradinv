"""Solyaev's (2023) concentration tensor in block form.

1. Kinematics. Solyaev uses Mindlin's form I with kappa_ijk = d_i d_j u_k
   (symmetric in the first two indices; identity I_ijlm delta_kn). With that
   index order the space of kappa is the same 18-dimensional subspace V as the
   space of eta_ijm = d_m eps_ij, and
       kappa = T eta,   kappa_ijk = eta_kij + eta_kji - eta_ijk,   T = 3 SS - 2 I,
   an isotropic element diagonal in the GL(3) chain: +1 on S (H3, H1^s) and -2
   on R (H2, H1^r). In a crystal algebra each copy c carries t_c in {1, -2}
   (`form_factors`) and, with Lt = diag(t_c) in each block,
       stiffness-like (D, G):          X_I = Lt^-1 X_II Lt^-1
       strain-like (concentrators):    X_I = Lt X_II Lt^-1   (similarity)
       compliance-like (polarization): X_I = Lt X_II Lt
   (no mixing of copies; with Lazar's 2016 pair there would be).

2. Concentrator (Solyaev 2023, eq. 7): B = (I + P (G_gr - G))^-1, P = <S6> G^-1.
   With G and <S6> isotropic, P is an isotropic element (not symmetric in
   general) and, with a cubic (hexagonal) grain, B lies in the algebra of
   dimension 15 (32). Block by block B^p = (Id + P^p (G_gr^p - G^p))^-1, with
   the isotropic elements written in the crystal blocks by
   `averages.embed_isotropic` (cubic: 3x3, 2x2 and two scalar inversions).

3. Linear invariants. The orientation average keeps the isotropic part
   (y3, y2, Y) of B (`averages.isotropic_part`, valid for non-symmetric
   elements). A non-symmetric isotropic tensor on V has SIX linear invariants;
   Solyaev's eq. (13) lists five (B1..B5) and the sixth is B2' = B_ijjkki:
       B1  = 5 Yss - sqrt5 (Ysr + Yrs) + Yrr
       B2  = 5 Yss - sqrt5 Ysr + 2 sqrt5 Yrs - 2 Yrr
       B2' = 5 Yss + 2 sqrt5 Ysr - sqrt5 Yrs - 2 Yrr
       B3  = 5 Yss + 2 sqrt5 (Ysr + Yrs) + 4 Yrr
       B4  = 7 y3 + 5 y2 + 3 (Yss + Yrr)        (the trace)
       B5  = 7 y3 - (5/2) y2 + 3 Yss - (3/2) Yrr
   <B> = I  <=>  (B1, B2, B2', B3, B4, B5) = (6, 3, 3, 9, 18, 6).

4. Ingredients of Solyaev (2023) and Ma et al. (2018), form I, Solyaev's indices.
   Read eqs. (17) and (19) as form II expressions of the Helmholtz model
   l^2 C (x) delta; eq. (21) as printed is not Ma's <N^>: Ma's
   tensor is Solyaev's form with coefficients (n1, 2n2, 4n3, 2n4, 2n5) of the
   printed ones, i.e. (n^1, 2n^2, 2n^3, 2n^4, 2n^5) of Ma's (`ma2018_s6_isotropic`).
   Neither is the inclusion average. The exact <S6> of the sphere is
   `eshelby.sphere_N_isotropic` (`eshelby.helmholtz_sphere_N_isotropic` for this
   medium); its H1 block is not symmetric. The functions below are kept as
   transcriptions of the printed formulas (supplement, sections S6 and S7).

Isotropic elements are blocks {'H3', 'H2', 'H1'} as in `gradinv.isotropic`.
"""
import numpy as np
import sympy as sp

from ._tensor import to_float
from .averages import embed_isotropic, isotropic_part
from .skeleton import identity_V, sym_projector

__all__ = ["form_I_map", "form_factors", "to_form_I", "to_form_II", "concentrator",
           "INVARIANT_NAMES", "IDENTITY_VALUES", "invariants_from_isotropic", "invariants",
           "solyaev_phi", "solyaev_n", "solyaev_s4", "solyaev_G_isotropic",
           "solyaev_medium_isotropic", "solyaev_s6_isotropic", "ma2018_aifantis_constants",
           "ma2018_nhat", "ma2018_s6_isotropic"]

R = sp.Rational


# ------------------------------------------------------------------ kinematics
def form_I_map(exact=False):
    """T = 3 SS - 2 I on V (27 x 27): kappa = T eta with Solyaev's index order."""
    if exact:
        return 3 * sp.Matrix(sym_projector(exact=True)) - 2 * sp.Matrix(identity_V(exact=True))
    return 3 * sym_projector() - 2 * identity_V()


_FACTOR_CACHE = {}


def form_factors(alg):
    """{irrep: [t_c]} with t_c = 1 for a copy in S and -2 for a copy in R."""
    key = (alg.name, alg.irreps)
    if key not in _FACTOR_CACHE:
        S = sym_projector()
        out = {}
        for p, copies, _d in alg.irreps:
            ts = []
            for a in copies:
                E = to_float(alg.embedding(p, a))
                if np.abs(S @ E - E).max() < 1e-12:
                    ts.append(1)
                elif np.abs(S @ E).max() < 1e-12:
                    ts.append(-2)
                else:
                    raise ValueError(f"copy {a} of {p} lies neither in S nor in R")
            out[p] = tuple(ts)
        _FACTOR_CACHE[key] = out
    return _FACTOR_CACHE[key]


_KINDS = {"stiffness": (-1, -1), "strain": (1, -1), "compliance": (1, 1)}


def _scale(alg, blocks, kind, direction):
    left, right = _KINDS[kind]
    left, right = left * direction, right * direction
    tf = form_factors(alg)
    out = {}
    for p, X in blocks.items():
        t = tf[p]
        if isinstance(X, sp.MatrixBase):
            L = sp.diag(*[sp.Integer(x) ** left for x in t])
            Rm = sp.diag(*[sp.Integer(x) ** right for x in t])
            out[p] = L * X * Rm
        else:
            L = np.diag([float(x) ** left for x in t])
            Rm = np.diag([float(x) ** right for x in t])
            out[p] = L @ np.asarray(X, float) @ Rm
    return out


def to_form_I(alg, blocks, kind):
    """Blocks in Solyaev's form I from form II.

    kind: 'stiffness' (G = T^-T D T^-1), 'strain' (B -> T B T^-1) or
    'compliance' (P -> T P T^T); T is symmetric and diagonal in the copies.
    """
    return _scale(alg, blocks, kind, +1)


def to_form_II(alg, blocks, kind):
    """Inverse of `to_form_I`."""
    return _scale(alg, blocks, kind, -1)


# ------------------------------------------------------------------ concentrator
def concentrator(alg, grain, medium, polar):
    """Blocks of B = (I + P (G_gr - G))^-1 (Solyaev 2023, eq. 7).

    grain: blocks of G_gr in `alg`; medium, polar: isotropic blocks of G and of
    P = <S6> G^-1. All in the same form (form I for Solyaev).
    """
    Gm, Pm = embed_isotropic(alg, medium), embed_isotropic(alg, polar)
    out = {}
    for p, X in grain.items():
        if isinstance(X, sp.MatrixBase):
            out[p] = (sp.eye(X.shape[0]) + Pm[p] * (X - Gm[p])).inv()
        else:
            X = np.asarray(X, float)
            out[p] = np.linalg.inv(np.eye(X.shape[0]) + Pm[p] @ (X - Gm[p]))
    return out


INVARIANT_NAMES = ("B1", "B2", "B2'", "B3", "B4", "B5")
IDENTITY_VALUES = {"B1": 6, "B2": 3, "B2'": 3, "B3": 9, "B4": 18, "B5": 6}


def invariants_from_isotropic(iso):
    """Solyaev's B1..B5 and the sixth invariant B2' from the isotropic part (y3, y2, Y)."""
    y3, y2, Y = iso["H3"][0, 0], iso["H2"][0, 0], iso["H1"]
    sym = isinstance(Y, sp.MatrixBase)
    s5 = sp.sqrt(5) if sym else np.sqrt(5)
    half = R(1, 2) if sym else 0.5
    Yss, Ysr, Yrs, Yrr = Y[0, 0], Y[0, 1], Y[1, 0], Y[1, 1]
    return {"B1": 5 * Yss - s5 * (Ysr + Yrs) + Yrr,
            "B2": 5 * Yss - s5 * Ysr + 2 * s5 * Yrs - 2 * Yrr,
            "B2'": 5 * Yss + 2 * s5 * Ysr - s5 * Yrs - 2 * Yrr,
            "B3": 5 * Yss + 2 * s5 * (Ysr + Yrs) + 4 * Yrr,
            "B4": 7 * y3 + 5 * y2 + 3 * (Yss + Yrr),
            "B5": 7 * y3 - 5 * half * y2 + 3 * Yss - 3 * half * Yrr}


def invariants(alg, blocks):
    """The six linear invariants of the element with these blocks (e.g. B)."""
    return invariants_from_isotropic(isotropic_part(alg, blocks))


# ------------------------------------------------------------------ Solyaev (2023)
def _num(x):
    return isinstance(x, sp.Basic)


def _iso(m3, m2, M, sym):
    if sym:
        return {"H3": sp.Matrix([[m3]]), "H2": sp.Matrix([[m2]]), "H1": sp.Matrix(M)}
    return {"H3": np.array([[m3]], float), "H2": np.array([[m2]], float), "H1": np.array(M, float)}


def solyaev_phi(Rbar):
    """Phi(R/l) of Solyaev (2023), below eq. (21): R^-3 (1+R) e^-R (sinh R - R cosh R),
    evaluated as R^-3 (1+R) [(1-R) - (1+R) e^-2R] / 2 (the same function, without the
    overflow of sinh and cosh for R/l > ~700)."""
    return Rbar ** -3 * (1 + Rbar) * ((1 - Rbar) - (1 + Rbar) * np.exp(-2 * Rbar)) / 2


def solyaev_n(lam, mu, phi):
    """n_1..n_5 of Solyaev (2023), eq. (21), AS PRINTED (n_3 is half of Ma's;
    exact only on two of the six classes)."""
    c = lam + 2 * mu
    return (-(5 * lam + 12 * mu) / (35 * c) * phi, -(19 * lam + 26 * mu) / (70 * c) * phi,
            -(15 * lam + 38 * mu) / (70 * c) * phi, -2 * (5 * lam + 12 * mu) / (35 * c) * phi,
            -2 * (3 * lam - 4 * mu) / (35 * c) * phi)


def solyaev_s4(lam, mu, phi):
    """(s1, s2) of Solyaev (2023), eq. (20), AS PRINTED: <S4> = s1 d (x) d + s2 I.

    With I = (dd + dd)/2 the classical limit (phi -> 0) needs 2 s2.
    """
    c = lam + 2 * mu
    return ((3 * lam - 2 * mu) / (15 * c) * (1 + 3 * phi), (3 * lam + 8 * mu) / (15 * c) * (1 + 3 * phi))


def solyaev_G_isotropic(A1, A2, A3, A4, A5):
    """Isotropic blocks (form I, Solyaev's indices) of his eq. (3) with constants A1..A5."""
    sym = _num(A1)
    s5 = sp.sqrt(5) if sym else np.sqrt(5)
    off = -s5 * (2 * A1 - A2 - 8 * A3) / 6
    M = [[(5 * A1 + 5 * A2 + 10 * A3 + 3 * A4 + 3 * A5) / 3, off],
         [off, (2 * A1 - 4 * A2 + 16 * A3 + 6 * A4 - 3 * A5) / 6]]
    return _iso(A4 + A5, A4 - A5 / 2, M, sym)


def solyaev_medium_isotropic(lam, mu, l2):
    """Eq. (19) READ WITH FORM I INDICES (G = l^2 C_ijlm delta_kn): m3 = m2 = 2 mu l^2,
    M = (l^2/3) [[5 lam + 6 mu, 2 sqrt5 lam], [2 sqrt5 lam, 4 lam + 6 mu]].

    Kept as a transcription: the consistent reading is the form II Helmholtz model
    l^2 C (x) delta, i.e. `isotropic.from_lazar2022((0, lam l2/2, 0, mu l2, 0))`
    moved to form I.
    """
    sym = _num(lam)
    s5 = sp.sqrt(5) if sym else np.sqrt(5)
    M = [[l2 * (5 * lam + 6 * mu) / 3, 2 * s5 * l2 * lam / 3],
         [2 * s5 * l2 * lam / 3, 2 * l2 * (2 * lam + 3 * mu) / 3]]
    return _iso(2 * mu * l2, 2 * mu * l2, M, sym)


def solyaev_s6_isotropic(n1, n2, n3, n4, n5):
    """Isotropic blocks of the form of eq. (21) with coefficients n1..n5 (symmetric):
    m3 = 2 n3 + n4, m2 = n4 - n3, and the H1 block below."""
    sym = _num(n1)
    s5 = sp.sqrt(5) if sym else np.sqrt(5)
    off = s5 * (2 * n1 + n2 - 2 * n5) / 3
    M = [[(5 * n1 + 10 * n2 + 6 * n3 + 3 * n4 + 10 * n5) / 3, off],
         [off, (4 * n1 - 4 * n2 - 3 * n3 + 3 * n4 + 2 * n5) / 3]]
    return _iso(2 * n3 + n4, n4 - n3, M, sym)


# ------------------------------------------------------------------ Ma et al. (2018)
def ma2018_aifantis_constants(lam, mu, l2):
    """Eq. (4.3) of Ma et al. (2018): a1 = lam l^2/2, a2 = a3 = 0, a4 = a5 = mu l^2/2."""
    return (lam * l2 / 2, 0 * lam, 0 * lam, mu * l2 / 2, mu * l2 / 2)


def ma2018_nhat(lam, mu, a, l1, l2, R_):
    """n^_1..n^_5 of Ma et al. (2018), eq. (3.19), spherical inclusion of radius R_.

    Ma's functions are Phi_Ma(k) = -k (k + R) exp(-R/k) and
    Gamma_Ma(k) = R cosh(R/k) - k sinh(R/k) (checked on the printed p. 65);
    Phi_Ma(k) Gamma_Ma(k) / R^3 = Phi(R/k) of Solyaev. AS PRINTED: checked
    against the Fourier route, only n^_1 is the average;
    use `eshelby.sphere_N_isotropic`.
    """
    a1, a2, a3, a4, a5 = a
    c = lam + 2 * mu
    gx = lambda k: -k * (k + R_) * np.exp(-R_ / k) * (R_ * np.cosh(R_ / k) - k * np.sinh(R_ / k))
    f2 = gx(l2) / (35 * mu * R_ ** 3 * l2 ** 2)
    f1 = gx(l1) / (35 * c * R_ ** 3 * l1 ** 2)
    return (f2 * 2 * (a5 - 6 * a4 - 28 * a3) - f1 * (14 * a3 + 7 * a2 + 2 * (a4 + a5)),
            f2 * (8 * a4 - 13 * a5 - 28 * a2 + 28 * a3) - f1 * (8 * (a4 + a5) + 7 * (2 * a1 + 3 * a2 + 4 * a3)),
            f2 * (8 * a4 - 38 * a5) - f1 * 8 * (a4 + a5),
            f2 * 4 * (a5 - 6 * a4) - f1 * 4 * (a4 + a5),
            f2 * 2 * (7 * a2 + 4 * a4 + 4 * a5) - f1 * (8 * (a4 + a5) + 14 * (2 * a1 + a2)))


def ma2018_s6_isotropic(nhat):
    """Isotropic blocks of Ma's <N^> (eq. 3.18) AS PRINTED: the form of eq. (21) with
    coefficients (n^1, 2 n^2, 2 n^3, 2 n^4, 2 n^5). Not the average."""
    n1, n2, n3, n4, n5 = nhat
    return solyaev_s6_isotropic(n1, 2 * n2, 2 * n3, 2 * n4, 2 * n5)
