"""Self-contained implementation of Solyaev's (2023) self-consistent scheme for
the Fe example (spherical grains, one-parameter Helmholtz medium), with three
choices of the averaged sixth-order Eshelby tensor <S^(6)>:

  "printed": Solyaev's eq. (21) as printed (n1..n5 below it, I = sym. identity);
  "ma":      Ma et al. (2018), eq. (3.18) with eq. (3.19) and the Helmholtz
             constants of their eq. (4.3) (= eq. 21 form with (n1,2n2,4n3,2n4,2n5));
  "exact":   the exact ball average <N> = P T (gradinv.eshelby; checked in real
             space by tests/test_eshelby_real_space.py).

Kinematics: form I with Solyaev's index order, kappa_{ijk} = u_{k,ij}. The grain
and medium gradient tensors are the Helmholtz model l^2 C (x) delta written in
form II and converted to form I. <S^(4)> = s1 dd + 2 s2 I (Solyaev's
printed s2 I is half the Eshelby deviatoric eigenvalue).
"""
import itertools

import numpy as np

from formI_index import (I4, Phi_ma, d, fourier_P, from_function, helmholtz_a, identity_formI, idx,
                    ma_N_printed, ma_T, solyaev_Phi)

PI6 = identity_formI()
# orthonormal basis of V (form I, symmetric in the first two indices)
_w, _v = np.linalg.eigh(PI6)
Q6 = _v[:, _w > 0.5]                     # 27 x 18

# fourth order: 9x9 matrices on (i,j)
def idx2(i, j):
    return 3 * i + j


def from4(f):
    X = np.zeros((9, 9))
    for i, j, k, l in itertools.product(range(3), repeat=4):
        X[idx2(i, j), idx2(k, l)] = f(i, j, k, l)
    return X


PI4 = from4(lambda i, j, k, l: float(I4(i, j, k, l)))
_w4, _v4 = np.linalg.eigh(PI4)
Q4 = _v4[:, _w4 > 0.5]                   # 9 x 6


def C_iso(lam, mu):
    return from4(lambda i, j, k, l: lam * d(i, j) * d(k, l) + mu * (d(i, k) * d(j, l) + d(i, l) * d(j, k)))


def C_cubic(C11, C12, C44):
    """Solyaev eq. (16)."""
    return from4(lambda i, j, k, l: C12 * d(i, j) * d(k, l) + C44 * (d(i, k) * d(j, l) + d(i, l) * d(j, k))
                 + (C11 - C12 - 2 * C44) * (1 if i == j == k == l else 0))


# form II -> form I: eta_{ijk} = eps_{ij,k} = (kappa_{jki} + kappa_{ikj})/2,
# kappa_{abc} = u_{c,ab}.  eta = P kappa, G_I = P^T D_II P.
P_II_I = from_function(lambda i, j, k, a, b, c:
                       0.5 * (d(a, j) * d(b, k) * d(c, i) + d(a, i) * d(b, k) * d(c, j))) @ PI6


def helmholtz_formII(C4, ell):
    """D_{ijk,lmn} = l^2 C_{ijlm} delta_{kn} (form II, eta_{ijk} = eps_{ij,k})."""
    return from_function(lambda i, j, k, l, m, n: ell**2 * C4[idx2(i, j), idx2(l, m)] * d(k, n))


def helmholtz_formI(C4, ell):
    return P_II_I.T @ helmholtz_formII(C4, ell) @ P_II_I


def s1s2(lam, mu, Phi):
    s1 = (3 * lam - 2 * mu) / (15 * (lam + 2 * mu)) * (1 + 3 * Phi)
    s2 = (3 * lam + 8 * mu) / (15 * (lam + 2 * mu)) * (1 + 3 * Phi)
    return s1, s2


def S4_avg(lam, mu, Phi, printed=False):
    s1, s2 = s1s2(lam, mu, Phi)
    f = 1.0 if printed else 2.0
    return from4(lambda i, j, k, l: s1 * d(i, j) * d(k, l) + f * s2 * float(I4(i, j, k, l)))


def solyaev_n(lam, mu, Phi):
    L = lam + 2 * mu
    return (-(5 * lam + 12 * mu) / (35 * L) * Phi, -(19 * lam + 26 * mu) / (70 * L) * Phi,
            -(15 * lam + 38 * mu) / (70 * L) * Phi, -2 * (5 * lam + 12 * mu) / (35 * L) * Phi,
            -2 * (3 * lam - 4 * mu) / (35 * L) * Phi)


def eq21_tensor(n):
    """Solyaev eq. (21): <S^(6)>_{ijklmn}."""
    n1, n2, n3, n4, n5 = n
    I = lambda a, b, c, e: float(I4(a, b, c, e))
    return from_function(lambda i, j, k, l, m, nn:
                         n1 * d(i, j) * d(l, m) * d(k, nn)
                         + n2 * (I(i, j, k, nn) * d(l, m) + I(l, m, k, nn) * d(i, j))
                         + n3 * (I(i, j, l, nn) * d(k, m) + I(i, j, m, nn) * d(k, l))
                         + n4 * I(i, j, l, m) * d(k, nn)
                         + n5 * (I(l, m, i, nn) * d(j, k) + I(l, m, j, nn) * d(i, k)))


def S6_avg(kind, lam, mu, ell, R):
    Phi = solyaev_Phi(R / ell)
    if kind == "printed":
        return PI6 @ eq21_tensor(solyaev_n(lam, mu, Phi)) @ PI6
    if kind == "ma_eq21":
        n1, n2, n3, n4, n5 = solyaev_n(lam, mu, Phi)
        return PI6 @ eq21_tensor((n1, 2 * n2, 4 * n3, 2 * n4, 2 * n5)) @ PI6
    if kind == "ma":
        return PI6 @ ma_N_printed(lam, mu, helmholtz_a(lam, mu, ell), R) @ PI6
    if kind == "exact":
        return PI6 @ fourier_P(lam, mu, ell, ell, R) @ ma_T(helmholtz_a(lam, mu, ell)) @ PI6
    raise ValueError(kind)


def red6(X):
    return Q6.T @ X @ Q6


def full6(X18):
    return Q6 @ X18 @ Q6.T


def red4(X):
    return Q4.T @ X @ Q4


def full4(X6):
    return Q4 @ X6 @ Q4.T


def lam_mu(k, nu):
    """Solyaev eq. (24)."""
    return 3 * k * nu / (1 + nu), 3 * k * (1 - 2 * nu) / (2 * (1 + nu))


INV_NAMES = ("A1", "A2", "B1", "B2", "B3", "B4", "B5", "B2p")
INV_TARGET = np.array([3.0, 6.0, 6.0, 3.0, 9.0, 18.0, 6.0, 3.0])


def invariants(A, B):
    A4 = A.reshape(3, 3, 3, 3)
    B6 = B.reshape(3, 3, 3, 3, 3, 3)
    return np.array([np.einsum("iijj", A4), np.einsum("ijij", A4),
                     np.einsum("ijjikk", B6), np.einsum("iikkjj", B6), np.einsum("iikjjk", B6),
                     np.einsum("ijkijk", B6), np.einsum("ijkkji", B6), np.einsum("ijjkki", B6)])


class FeCase:
    """Grain data of Solyaev's example (his ref. [8]); l0 = 1."""
    C11, C12, C44 = 108.0, 62.0, 28.3

    def __init__(self, kind, s4_printed=False):
        self.kind = kind
        self.s4_printed = s4_printed
        self.Cgr = C_cubic(self.C11, self.C12, self.C44)
        self.kgr = (self.C11 + 2 * self.C12) / 3
        self.Ggr = helmholtz_formI(self.Cgr, 1.0)

    def concentrators(self, nu, ell, R):
        lam, mu = lam_mu(self.kgr, nu)
        C = C_iso(lam, mu)
        G = helmholtz_formI(C, ell)
        Phi = solyaev_Phi(R / ell)
        S4 = S4_avg(lam, mu, Phi, printed=self.s4_printed)
        S6 = S6_avg(self.kind, lam, mu, ell, R)
        A = full4(np.linalg.inv(np.eye(6) + red4(S4) @ np.linalg.inv(red4(C)) @ red4(self.Cgr - C)))
        B = full6(np.linalg.inv(np.eye(18) + red6(S6) @ np.linalg.inv(red6(G)) @ red6(self.Ggr - G)))
        return A, B

    def invariants(self, nu, ell, R):
        return invariants(*self.concentrators(nu, ell, R))

    def error(self, nu, ell, R, terms=7, squared=True):
        """Solyaev eq. (22). terms=7: A1, A2, B1..B5; terms=8 adds B2'.
        squared=False: the printed form (B terms not squared)."""
        v = self.invariants(nu, ell, R)
        rel = (v - INV_TARGET) / INV_TARGET
        rel = rel[:terms]
        if squared:
            return float(np.sqrt(np.sum(rel**2)))
        s = rel[0] ** 2 + rel[1] ** 2 + np.sum(rel[2:])
        return float(np.sqrt(s)) if s >= 0 else float("nan")


# ---------------------------------------------------------------------------
# Fast version: every tensor is a fixed linear combination of precomputed bases.
class _Bases:
    def __init__(self):
        I = lambda a, b, c, e: float(I4(a, b, c, e))
        self.eq21 = [PI6 @ eq21_tensor(e) @ PI6 for e in np.eye(5)]
        self.N6 = PI6 @ from_function(lambda r, t, s, i, j, k: _n6(r, t, s, i, j, k)) @ PI6
        self.N4d = PI6 @ from_function(lambda r, t, s, i, j, k: _n4(r, t, i, j) * d(s, k)) @ PI6
        self.T = [PI6 @ ma_T(tuple(e)) @ PI6 for e in np.eye(5)]
        self.HL = helmholtz_formI(C_iso(1.0, 0.0), 1.0)   # lambda part, l = 1
        self.HM = helmholtz_formI(C_iso(0.0, 1.0), 1.0)   # mu part, l = 1
        self.CL, self.CM = C_iso(1.0, 0.0), C_iso(0.0, 1.0)
        self.dd = from4(lambda i, j, k, l: d(i, j) * d(k, l))
        self.I4 = PI4
        # Ma (3.18) class tensors
        from formI_index import CLASS
        self.cls = {c: PI6 @ CLASS(c) @ PI6 for c in ("c1", "c2a", "c2b", "c3", "c4", "c5")}


from formI_index import n4 as _n4, n6 as _n6  # noqa: E402

_B = None


def bases():
    global _B
    if _B is None:
        _B = _Bases()
    return _B


def S6_fast(kind, lam, mu, ell, R):
    b = bases()
    Phi = solyaev_Phi(R / ell)
    if kind in ("printed", "ma"):
        n = np.array(solyaev_n(lam, mu, Phi))
        if kind == "ma":
            n = n * np.array([1, 2, 4, 2, 2])
        return sum(c * t for c, t in zip(n, b.eq21))
    if kind == "exact":
        from formI_index import F_radial
        F = F_radial(ell, R)
        P = F / (lam + 2 * mu) * b.N6 + F / mu * (b.N4d - b.N6)
        T = ell**2 * (lam * b.HL + mu * b.HM)
        return P @ T
    raise ValueError(kind)


class FeCaseFast(FeCase):
    def concentrators(self, nu, ell, R):
        b = bases()
        lam, mu = lam_mu(self.kgr, nu)
        C = lam * b.CL + mu * b.CM
        G = ell**2 * (lam * b.HL + mu * b.HM)
        Phi = solyaev_Phi(R / ell)
        s1, s2 = s1s2(lam, mu, Phi)
        S4 = s1 * b.dd + (1.0 if self.s4_printed else 2.0) * s2 * b.I4
        S6 = S6_fast(self.kind, lam, mu, ell, R)
        A = full4(np.linalg.inv(np.eye(6) + red4(S4) @ np.linalg.inv(red4(C)) @ red4(self.Cgr - C)))
        B = full6(np.linalg.inv(np.eye(18) + red6(S6) @ np.linalg.inv(red6(G)) @ red6(self.Ggr - G)))
        return A, B


# ---------------------------------------------------------------------------
# Readings of eqs. (17) and (19): "coherent" = both form II Helmholtz models
# converted to form I; "literal" = l^2 C_ijlm delta_kn with Solyaev's form I
# indices (C on the derivative indices). A mixed reading uses one for the grain
# and the other for the medium.
def G_literal(C4, ell):
    return PI6 @ from_function(lambda i, j, k, l, m, n: ell**2 * C4[idx2(i, j), idx2(l, m)] * d(k, n)) @ PI6


class FeCaseVariant(FeCaseFast):
    def __init__(self, kind, grain_literal=False, medium_literal=False, s4_printed=False,
                 C=(108.0, 62.0, 28.3)):
        super().__init__(kind, s4_printed)
        self.C11, self.C12, self.C44 = C
        self.Cgr = C_cubic(*C)
        self.kgr = (C[0] + 2 * C[1]) / 3
        self.medium_literal = medium_literal
        self.Ggr = G_literal(self.Cgr, 1.0) if grain_literal else helmholtz_formI(self.Cgr, 1.0)

    def concentrators(self, nu, ell, R):
        b = bases()
        lam, mu = lam_mu(self.kgr, nu)
        C = lam * b.CL + mu * b.CM
        G = G_literal(C, ell) if self.medium_literal else ell**2 * (lam * b.HL + mu * b.HM)
        Phi = solyaev_Phi(R / ell)
        s1, s2 = s1s2(lam, mu, Phi)
        S4 = s1 * b.dd + (1.0 if self.s4_printed else 2.0) * s2 * b.I4
        S6 = S6_fast(self.kind, lam, mu, ell, R)
        A = full4(np.linalg.inv(np.eye(6) + red4(S4) @ np.linalg.inv(red4(C)) @ red4(self.Cgr - C)))
        B = full6(np.linalg.inv(np.eye(18) + red6(S6) @ np.linalg.inv(red6(G)) @ red6(self.Ggr - G)))
        return A, B


def minimize_error(case, R, nu_grid=np.linspace(-0.9, 0.49, 70), ell_grid=np.linspace(0.05, 6.0, 80)):
    from scipy.optimize import minimize
    f = lambda p: case.error(p[0], p[1], R) if (-1 < p[0] < 0.5 and p[1] > 0) else np.inf
    with np.errstate(all="ignore"):
        best = min(((f((nu, ell)), nu, ell) for nu in nu_grid for ell in ell_grid
                    if np.isfinite(f((nu, ell)))), key=lambda t: t[0])
        res = minimize(f, best[1:], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-11, maxiter=4000))
    return float(res.fun), float(res.x[0]), float(res.x[1])
