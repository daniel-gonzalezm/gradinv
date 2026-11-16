"""Solyaev's (2023) self-consistent scheme for untextured cubic polycrystals with
spherical grains and the one-parameter Helmholtz effective medium.

Unknowns of the medium: Poisson's ratio nu and length l (the bulk modulus is
fixed to the grain's, k = (C11 + 2 C12)/3, which satisfies A1 = 3 exactly).
Medium: C(lam, mu) and G = l^2 C (x) delta written in form II and converted to
Solyaev's form I. Grain: cubic C_gr and any cubic gradient tensor in form
I blocks (Solyaev's example: l0^2 C_gr (x) delta, `helmholtz_cubic_grain`).

Conditions (his eqs. 10-13): the linear invariants of the averaged strain
concentrator A (A1, A2) and of the gradient concentrator B (B1..B5, and the
sixth B2' omitted by him) equal those of the identity. Error function
(eq. 22, read with all terms squared):
    rho = sqrt(sum_t ((X_t - X_t^0)/X_t^0)^2),  t over 7 terms (his) or 8 (with B2').

Averaged Eshelby tensors of the sphere:
  <S4> = s1 d (x) d + s2 (dd + dd) with Ma's (s1, s2), `eshelby.sphere_S4`
         (Solyaev's printed s2 I is half of it);
  <S6> = "exact" (`eshelby.helmholtz_sphere_N_isotropic`, the average; default),
         "ma" (Ma et al. 2018, eqs. 3.18-3.19) or "printed" (Solyaev eq. 21);
         the last two are kept only to document the supplementary Fe case (S6).
"""
import numpy as np
from scipy.optimize import brentq, minimize

from . import averages
from . import concentrator as cc
from . import cubic, eshelby, isotropic
from .algebra import BlockAlgebra

__all__ = ["lame_from_k_nu", "helmholtz_medium", "helmholtz_cubic_grain", "averaged_S6",
           "kroner_invariants", "invariants", "error", "solve", "kroner_nu", "large_grain_limit",
           "TARGETS", "TERMS_7", "TERMS_8"]

TERMS_7 = ("A1", "A2", "B1", "B2", "B3", "B4", "B5")
TERMS_8 = TERMS_7 + ("B2'",)
TARGETS = {"A1": 3.0, "A2": 6.0, **{k: float(v) for k, v in cc.IDENTITY_VALUES.items()}}

_ISO = isotropic.algebra()
_CUB = cubic.algebra()


def lame_from_k_nu(k, nu):
    """Solyaev's eq. (24): (lam, mu) from the bulk modulus and Poisson's ratio."""
    return 3 * k * nu / (1 + nu), 3 * k * (1 - 2 * nu) / (2 * (1 + nu))


def helmholtz_medium(lam, mu, l2):
    """Isotropic blocks (Solyaev's form I) of l^2 C (x) delta (form II)."""
    return cc.to_form_I(_ISO, isotropic.from_lazar2022([0, lam * l2 / 2, 0, mu * l2, 0]), "stiffness")


def helmholtz_cubic_grain(C11, C12, C44, l2=1.0):
    """Cubic blocks (form I) of Solyaev's simplified grain l0^2 C_gr (x) delta, eq. (17) read as form II."""
    a = np.array([0, C12 / 2, 0, C44, 0, 0, 0, C11 - C12 - 2 * C44, 0, 0, 0]) * l2
    return cc.to_form_I(_CUB, cubic.from_lazar2022(a), "stiffness")


def averaged_S6(kind, lam, mu, ell, R):
    """Isotropic blocks of <S6> (form I) for the Helmholtz medium of length ell."""
    if kind == "exact":
        return eshelby.helmholtz_sphere_N_isotropic(lam, mu, ell ** 2, R)
    if kind == "ma":
        nh = cc.ma2018_nhat(lam, mu, cc.ma2018_aifantis_constants(lam, mu, ell ** 2), ell, ell, R)
        return cc.ma2018_s6_isotropic(nh)
    if kind == "printed":
        return cc.solyaev_s6_isotropic(*cc.solyaev_n(lam, mu, cc.solyaev_phi(R / ell)))
    raise ValueError(f"unknown <S6>: {kind!r}")


def kroner_invariants(C_grain, lam, mu, s1, s2):
    """A1 = A_iijj and A2 = A_ijij of A = (I + <S4> C^-1 (C_gr - C))^-1 for a cubic
    grain (C11, C12, C44) in an isotropic medium, <S4> = s1 dd + s2 (dd + dd).
    Closed form: <S4> has eigenvalues 3 s1 + 2 s2 (volumetric) and 2 s2 (deviatoric);
    the grain has shear moduli (C11 - C12)/2 (E, dim 2) and C44 (T2, dim 3)."""
    C11, C12, C44 = C_grain
    k, k_gr = lam + 2 * mu / 3, (C11 + 2 * C12) / 3
    alpha, beta = 3 * s1 + 2 * s2, 2 * s2
    a_v = 1 / (1 + alpha * (k_gr - k) / k)
    a_e = 1 / (1 + beta * ((C11 - C12) / 2 - mu) / mu)
    a_t = 1 / (1 + beta * (C44 - mu) / mu)
    return {"A1": 3 * a_v, "A2": a_v + 2 * a_e + 3 * a_t}


def invariants(C_grain, G_grain, k, nu, ell, R, s6="exact"):
    """The eight invariants (A1, A2, B1..B5, B2') for the medium (k, nu, ell) and a
    spherical grain of radius R. G_grain: cubic blocks in form I."""
    lam, mu = lame_from_k_nu(k, nu)
    s1, s2 = eshelby.sphere_S4(lam, mu, ell, ell, R)
    out = kroner_invariants(C_grain, lam, mu, s1, s2)
    G = helmholtz_medium(lam, mu, ell ** 2)
    P = BlockAlgebra.multiply(averaged_S6(s6, lam, mu, ell, R), BlockAlgebra.inverse(G))
    out.update(cc.invariants(_CUB, cc.concentrator(_CUB, G_grain, G, P)))
    return out


def error(inv, terms=TERMS_7):
    """Solyaev's eq. (22) with every term squared."""
    return float(np.sqrt(sum(((inv[t] - TARGETS[t]) / TARGETS[t]) ** 2 for t in terms)))


def solve(C_grain, G_grain, R, s6="exact", terms=TERMS_7, k=None,
          nu_grid=np.linspace(-0.9, 0.49, 70), ell_grid=np.linspace(0.05, 6.0, 80)):
    """Minimize rho(nu, ell) (eq. 25): coarse grid, then Nelder-Mead.
    Returns (rho, nu, ell). k defaults to the grain's bulk modulus."""
    if k is None:
        k = (C_grain[0] + 2 * C_grain[1]) / 3

    def f(p):
        nu, ell = p
        if not (-1 < nu < 0.5 and ell > 0):
            return np.inf
        with np.errstate(all="ignore"):
            v = error(invariants(C_grain, G_grain, k, nu, ell, R, s6), terms)
        return v if np.isfinite(v) else np.inf

    best = min(((f((nu, ell)), nu, ell) for nu in nu_grid for ell in ell_grid), key=lambda t: t[0])
    res = minimize(f, best[1:], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-11, maxiter=4000))
    return float(res.fun), float(res.x[0]), float(res.x[1])


# ------------------------------------------------------------- large-grain limit
def kroner_nu(C_grain, k=None):
    """Poisson's ratio of the classical Kroner estimate (A2 = 6 with the classical
    Eshelby tensor of the sphere, s1 = (3 lam - 2 mu)/(15 c), s2 = (3 lam + 8 mu)/(15 c),
    c = lam + 2 mu), with the bulk modulus fixed to the grain's."""
    if k is None:
        k = (C_grain[0] + 2 * C_grain[1]) / 3

    def g(nu):
        lam, mu = lame_from_k_nu(k, nu)
        c = lam + 2 * mu
        return kroner_invariants(C_grain, lam, mu, (3 * lam - 2 * mu) / (15 * c),
                                 (3 * lam + 8 * mu) / (15 * c))["A2"] - 6.0

    return float(brentq(g, -0.99, 0.4999, xtol=1e-14))


def _b_vector(iso, terms):
    inv = cc.invariants_from_isotropic(iso)
    return np.array([inv[t] / TARGETS[t] for t in terms])


def large_grain_limit(C_grain, G_grain, terms=TERMS_7, k=None):
    """Leading order of the scheme for R/l -> infinity.

    The exact <S6> of the Helmholtz medium is Phi(R/l) times a tensor S_hat(lam, mu)
    that does not depend on l or R, and G = l^2 G_hat. To first order in Phi,
        <B> - I = -Phi S_hat G_hat^-1 (s V - G_hat),   s = 1/l^2,
    with V the Voigt average (isotropic part) of the grain's form-I tensor, because
    the orientation average commutes with left multiplication by isotropic elements.
    The A terms fix nu to Kroner's at leading order (A1 = 3 exactly with k fixed),
    and the B terms give rho = |Phi(R/l)| |s a - b|. Since Phi(R/l) ~ -l/(2R) also
    depends on l, the minimum is that of s^(-1/2) |s a - b|, at s = |b| / |a|
    (closed form). Returns {"nu", "ell", "residual"}, residual = lim rho/|Phi|.
    """
    if k is None:
        k = (C_grain[0] + 2 * C_grain[1]) / 3
    nu = kroner_nu(C_grain, k)
    lam, mu = lame_from_k_nu(k, nu)
    S_hat = {p: v / cc.solyaev_phi(10.0) for p, v in averaged_S6("exact", lam, mu, 1.0, 10.0).items()}
    G_hat_inv = BlockAlgebra.inverse(helmholtz_medium(lam, mu, 1.0))
    V = averages.voigt(_CUB, G_grain)
    bterms = [t for t in terms if t.startswith("B")]
    a = _b_vector(BlockAlgebra.multiply(BlockAlgebra.multiply(S_hat, G_hat_inv), V), bterms)
    b = _b_vector(S_hat, bterms)
    s = float(np.linalg.norm(b) / np.linalg.norm(a))
    return {"nu": nu, "ell": float(1 / np.sqrt(s)), "residual": float(np.linalg.norm(s * a - b))}
