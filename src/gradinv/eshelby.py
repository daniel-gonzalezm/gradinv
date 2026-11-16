"""Averaged Eshelby tensors of a spherical inclusion in an isotropic first
strain-gradient medium, in block form.

Kinematics and constants of Ma et al. (2018): form I with eta_ijk = u_k,ij
(symmetric in ij; the index order of Solyaev 2023, `concentrator.form_I_map`),
energy W = lam/2 eps_ii eps_jj + mu eps_ij eps_ij + a1 eta_ijj eta_ikk
+ a2 eta_iik eta_kjj + a3 eta_iik eta_jjk + a4 eta_ijk eta_ijk + a5 eta_ijk eta_kji
(their eqs. 2.4-2.6) and lengths l1^2 = 2(a1+..+a5)/(lam+2mu), l2^2 = 2(a3+a4)/mu
(eq. 2.8).

Route (independent of Ma's Green-function derivation). For a uniform
eigen-gradient eta* (eigenstrain eps*) in the sphere Omega of radius R,
    <eta>_Omega = <N> eta*,   <N> = (1/V)(2 pi)^-3 int |chi(xi)|^2 xi xi K^-1 xi xi T d^3 xi,
with chi the Fourier transform of the indicator of Omega and
K = C xi xi + T xi xi xi xi = xi^2 [(lam+2mu)(1 + l1^2 xi^2) nn + mu(1 + l2^2 xi^2)(I - nn)].
Radial and angular parts separate:
    <N> = P T,   P = F(l1)/(lam+2mu) <n^6> + F(l2)/mu (<n^4> (x) delta - <n^6>),
    F(l) = -3 Phi(l) Gamma(l) / (R^3 l^2)
         = 3 [2 l R E - l^2 (1 - E) + R^2 (1 + E)] / (2 R^3 l),   E = exp(-2R/l),
with Phi(l) = -l(l+R) e^(-R/l), Gamma(l) = R cosh(R/l) - l sinh(R/l) (Ma's
notation). P is symmetric (the gradient analogue of Hill's polarization
tensor) and depends on the medium only through lam, mu, l1, l2. In the
isotropic blocks {H3, H2, H1} (GL(3) skeleton, copies (s, r)):
    <n^6>:            H3 = 2/35,  H2 = 0,     H1 = [[1/5, 0], [0, 0]]
    <n^4> (x) delta:  H3 = 2/15,  H2 = 2/15,  H1 = [[11, 2 sqrt5], [2 sqrt5, 10]] / 45
    T (eq. 2.6):      H3 = 2 (a4 + a5),  H2 = 2 a4 - a5,
                      H1 = [[2(5a1+5a2+5a3+3a4+3a5)/3, -sqrt5 (2a1-a2-4a3)/3],
                            [-sqrt5 (2a1-a2-4a3)/3, (2a1-4a2+8a3+6a4-3a5)/3]]
The fourth-order average is <S> = s1 d (x) d + s2 (dd + dd) with Ma's s1, s2
(eq. 3.19), which this route confirms; Ma's <N^> (eqs. 3.18-3.19) and
Solyaev's eq. (21) are not the average: use `sphere_N_isotropic`.

Range: l1, l2 > 0. F(l) l^2 -> 1 for R/l -> 0 and F -> 0 for R/l -> inf; the
closed form loses about 2 log10(l/R) digits when l >> R.
"""
import numpy as np
import sympy as sp

__all__ = ["lengths_squared", "radial_factor", "ma2018_T_isotropic", "angular_polarization",
           "polarization_isotropic",
           "sphere_N_isotropic", "sphere_S4", "helmholtz_sphere_N_isotropic"]


def _sym(*xs):
    return any(isinstance(x, sp.Basic) for x in xs)


def _iso(m3, m2, M, sym):
    if sym:
        return {"H3": sp.Matrix([[m3]]), "H2": sp.Matrix([[m2]]), "H1": sp.Matrix(M)}
    return {"H3": np.array([[m3]], float), "H2": np.array([[m2]], float), "H1": np.array(M, float)}


def lengths_squared(lam, mu, a):
    """(l1^2, l2^2) of Ma et al. (2018), eq. (2.8)."""
    a1, a2, a3, a4, a5 = a
    return 2 * (a1 + a2 + a3 + a4 + a5) / (lam + 2 * mu), 2 * (a3 + a4) / mu


def radial_factor(l, R):
    """F(l) = -3 Phi(l) Gamma(l) / (R^3 l^2), the radial weight of xi^2/(1 + l^2 xi^2)."""
    if _sym(l, R):
        E = sp.exp(-2 * R / l)
        return 3 * (2 * l * R * E - l ** 2 * (1 - E) + R ** 2 * (1 + E)) / (2 * R ** 3 * l)
    E = np.exp(-2 * R / l)
    return 3 * (2 * l * R * E + l ** 2 * np.expm1(-2 * R / l) + R ** 2 * (1 + E)) / (2 * R ** 3 * l)


def ma2018_T_isotropic(a):
    """Isotropic blocks of the gradient tensor T of Ma et al. (2018), eq. (2.6) (form I)."""
    a1, a2, a3, a4, a5 = a
    sym = _sym(*a)
    s5 = sp.sqrt(5) if sym else np.sqrt(5)
    off = -s5 * (2 * a1 - a2 - 4 * a3) / 3
    M = [[2 * (5 * a1 + 5 * a2 + 5 * a3 + 3 * a4 + 3 * a5) / 3, off],
         [off, (2 * a1 - 4 * a2 + 8 * a3 + 6 * a4 - 3 * a5) / 3]]
    return _iso(2 * (a4 + a5), 2 * a4 - a5, M, sym)


def angular_polarization(lam, mu, w1, w2):
    """Blocks of w1/(lam+2mu) <n^6> + w2/mu (<n^4> (x) delta - <n^6>) for radial weights
    w1, w2 (F(l1), F(l2) for the sphere; xi^2/(1 + l^2 xi^2) at a fixed |xi|)."""
    sym = _sym(lam, mu, w1, w2)
    s5 = sp.sqrt(5) if sym else np.sqrt(5)
    if sym:
        r = sp.Rational
    else:
        def r(p, q):
            return p / q
    f1 = w1 / (lam + 2 * mu)
    f2 = w2 / mu
    M = [[f1 * r(1, 5) + f2 * r(2, 45), f2 * 2 * s5 / 45],
         [f2 * 2 * s5 / 45, f2 * r(2, 9)]]
    return _iso(f1 * r(2, 35) + f2 * r(8, 105), f2 * r(2, 15), M, sym)


def polarization_isotropic(lam, mu, l1, l2, R):
    """Blocks of P with <N> = P T (symmetric; form I, Ma's and Solyaev's index order)."""
    return angular_polarization(lam, mu, radial_factor(l1, R), radial_factor(l2, R))


def sphere_N_isotropic(lam, mu, a, R):
    """Blocks of the inclusion average <N> (eta = <N> eta*) for constants a = (a1..a5).

    Not symmetric: its H1 block is P_H1 T_H1 (a non-symmetric isotropic element
    has six linear invariants).
    """
    l1s, l2s = lengths_squared(lam, mu, a)
    sym = _sym(lam, mu, R, *a)
    sqrt = sp.sqrt if sym else np.sqrt
    P = polarization_isotropic(lam, mu, sqrt(l1s), sqrt(l2s), R)
    T = ma2018_T_isotropic(a)
    return {p: (P[p] * T[p] if sym else P[p] @ T[p]) for p in P}


def helmholtz_sphere_N_isotropic(lam, mu, l2, R):
    """<N> for the Helmholtz (Aifantis) medium l^2 C (x) delta, Ma's eq. (4.3)
    (a1 = lam l^2/2, a2 = a3 = 0, a4 = a5 = mu l^2/2, so l1 = l2 = l)."""
    return sphere_N_isotropic(lam, mu, (lam * l2 / 2, 0 * lam, 0 * lam, mu * l2 / 2, mu * l2 / 2), R)


def sphere_S4(lam, mu, l1, l2, R):
    """(s1, s2) of the fourth-order average <S> = s1 d (x) d + s2 (dd + dd), from
    <S> = G(l1)/(lam+2mu) ... + G(l2)/mu ... with G(l) = 1 - l^2 F(l); equal to
    Ma's eq. (3.19). Note Solyaev's eq. (20) writes s2 I with
    I = (dd + dd)/2."""
    g1 = l1 ** 2 * radial_factor(l1, R)
    g2 = l2 ** 2 * radial_factor(l2, R)
    c = lam + 2 * mu
    s1 = (3 * lam - 2 * mu) / (15 * c) - (5 * lam + 2 * mu) / (15 * c) * g1 + 2 * g2 / 15
    s2 = (3 * lam + 8 * mu) / (15 * c) - 2 * mu / (15 * c) * g1 - g2 / 5
    return s1, s2
