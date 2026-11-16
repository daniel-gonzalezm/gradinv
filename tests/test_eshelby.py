"""Averaged Eshelby tensors of the sphere by the Fourier route (`gradinv.eshelby`).

Independent routes: the symbol K(xi) computed from C and Ma's T (eq. 2.6,
transcription in references/solyaev_sources.py); the angular average by Gauss
product quadrature with K inverted numerically at a fixed |xi|; the radial
integral by quadrature (slow); exact isotropic moments <n^4>, <n^6> (exact).
Ma's <N^> (eqs. 3.18-3.19) and Solyaev's eq. (21) are compared class by class.
"""
import itertools

import numpy as np
import pytest
import sympy as sp
from scipy import integrate

from gradinv import concentrator as cc
from gradinv import eshelby as es
from gradinv import isotropic
from gradinv.skeleton import identity_V
from helpers import is_zero
from references import solyaev_sources as ss

ISO = isotropic.algebra()
IV = identity_V()
RNG = np.random.default_rng(3505)
d = np.eye(3)
LAM, MU = 1.3, 0.7
A_GEN = (0.11, -0.05, 0.21, 0.37, 0.13)         # generic a1..a5 (l1 != l2)


# ------------------------------------------------------------------ helpers
def matchings(lst):
    if not lst:
        yield []
        return
    for q in range(1, len(lst)):
        for rest in matchings(lst[1:q] + lst[q + 1:]):
            yield [(lst[0], lst[q])] + rest


def moment(ix, exact=False):
    """<n_i1 ... n_ik> over the unit sphere: sum of delta pairings / (k+1)!!."""
    k = len(ix)
    tot = sum(int(all(ix[p] == ix[q] for p, q in mm)) for mm in matchings(list(range(k))))
    den = int(np.prod(range(k + 1, 0, -2)))
    return sp.Rational(tot, den) if exact else tot / den


def moment_tensors(exact=False):
    """<n^6> and <n^4> (x) delta as 27 x 27 (rows (r,t,s), columns (i,j,k))."""
    Z = sp.zeros if exact else (lambda a, b: np.zeros((a, b)))
    M6, A4 = Z(27, 27), Z(27, 27)
    for (r, t, s) in itertools.product(range(3), repeat=3):
        for (i, j, k) in itertools.product(range(3), repeat=3):
            M6[9 * r + 3 * t + s, 9 * i + 3 * j + k] = moment((r, t, s, i, j, k), exact)
            A4[9 * r + 3 * t + s, 9 * i + 3 * j + k] = moment((r, t, i, j), exact) * (1 if s == k else 0)
    return M6, A4


def Cvoigt(lam, mu):
    return lam * np.einsum("ij,kl->ijkl", d, d) + mu * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))


def Tma(a):
    return IV @ ss.ma2018_eq26(*a) @ IV


def sphere_dirs(nt=12, nphi=24):
    x, w = np.polynomial.legendre.leggauss(nt)
    out = []
    for c, wc in zip(x, w):
        s = np.sqrt(1 - c * c)
        for p in np.linspace(0, 2 * np.pi, nphi, endpoint=False):
            out.append((np.array([s * np.cos(p), s * np.sin(p), c]), wc * (2 * np.pi / nphi) / (4 * np.pi)))
    return out


def ma_phi_gamma(k, R):
    return -k * (k + R) * np.exp(-R / k) * (R * np.cosh(R / k) - k * np.sinh(R / k))


SIX = [  # isotropic pairing classes on V (rows (r,t,s): r,t derivative; columns (l,m,n))
    lambda r, t, s, l, m, n: d[s, n] * d[m, l] * d[r, t],
    lambda r, t, s, l, m, n: d[s, l] * d[m, n] * d[r, t] + d[s, m] * d[n, l] * d[r, t],
    lambda r, t, s, l, m, n: d[n, r] * d[t, s] * d[l, m] + d[n, t] * d[r, s] * d[l, m],
    lambda r, t, s, l, m, n: d[s, l] * d[m, r] * d[t, n] + d[s, l] * d[m, t] * d[r, n] + d[s, m] * d[l, r] * d[t, n] + d[s, m] * d[l, t] * d[r, n],
    lambda r, t, s, l, m, n: d[s, n] * d[m, r] * d[l, t] + d[s, n] * d[m, t] * d[l, r],
    lambda r, t, s, l, m, n: d[m, n] * d[r, s] * d[l, t] + d[m, n] * d[t, s] * d[l, r] + d[l, n] * d[t, s] * d[m, r] + d[l, n] * d[r, s] * d[m, t],
]
X6 = np.array([(ss.ten(f) @ IV).ravel() for f in SIX]).T       # 729 x 6


def coords6(N27):
    c, *_ = np.linalg.lstsq(X6, (N27 @ IV).ravel(), rcond=None)
    return c, np.abs(X6 @ c - (N27 @ IV).ravel()).max()


def N27(blocks):
    return ISO.assemble(blocks)


# ------------------------------------------------------------------ A ingredients
def test_six_classes_are_a_basis_of_the_isotropic_algebra_on_V():
    assert np.linalg.matrix_rank(X6) == 6


def test_T_of_eq26_blocks():
    for a in (A_GEN, tuple(RNG.normal(size=5))):
        T = Tma(a)
        assert np.abs(ISO.residual(T)).max() < 1e-13
        bl = es.ma2018_T_isotropic(a)
        assert all(np.allclose(bl[k], ISO.moduli(T)[k], atol=1e-13) for k in bl)


@pytest.mark.parametrize("trial", range(3))
def test_symbol_K_has_the_longitudinal_transverse_form_of_eq28(trial):
    a = tuple(RNG.normal(size=5) * 0.3 + np.array([0, 0, 0.5, 0.5, 0.3]))
    l1s, l2s = es.lengths_squared(LAM, MU, a)
    T = Tma(a).reshape(3, 3, 3, 3, 3, 3)
    n = RNG.normal(size=3)
    n /= np.linalg.norm(n)
    xi = 0.7
    K = xi ** 2 * np.einsum("ikmn,i,m->kn", Cvoigt(LAM, MU), n, n) + xi ** 4 * np.einsum("ijklmn,i,j,l,m->kn", T, n, n, n, n)
    P = np.outer(n, n)
    Kp = xi ** 2 * ((LAM + 2 * MU) * (1 + l1s * xi ** 2) * P + MU * (1 + l2s * xi ** 2) * (np.eye(3) - P))
    assert np.allclose(K, Kp, atol=1e-13)


def test_moment_blocks():
    M6, A4 = moment_tensors()
    for _ in range(2):
        w1, w2 = RNG.uniform(0.1, 1, size=2)
        ref = w1 / (LAM + 2 * MU) * M6 + w2 / MU * (A4 - M6)
        assert np.abs(N27(es.angular_polarization(LAM, MU, w1, w2)) - IV @ ref @ IV).max() < 1e-15


@pytest.mark.parametrize("xi", [0.3, 1.1, 4.0])
def test_angular_assembly_by_brute_force_at_fixed_xi(xi):
    """xi^4 nn K^-1 nn T averaged over directions with K inverted numerically,
    against P(w1, w2) T with w = xi^2/(1 + l^2 xi^2)."""
    a = A_GEN
    T = Tma(a)
    T6 = T.reshape(3, 3, 3, 3, 3, 3)
    C = Cvoigt(LAM, MU)
    acc = np.zeros((27, 27))
    for n, w in sphere_dirs():
        X = xi * n
        K = np.einsum("ikmn,i,m->kn", C, X, X) + np.einsum("ijklmn,i,j,l,m->kn", T6, X, X, X, X)
        G = np.einsum("r,t,sk,i,j->rtsijk", X, X, np.linalg.inv(K), X, X).reshape(27, 27)
        acc += w * G @ T
    l1s, l2s = es.lengths_squared(LAM, MU, a)
    P = es.angular_polarization(LAM, MU, xi ** 2 / (1 + l1s * xi ** 2), xi ** 2 / (1 + l2s * xi ** 2))
    assert np.abs(acc - N27(P) @ T).max() < 1e-14


def test_radial_factor_closed_forms_and_limits():
    for R, l in [(1.0, 0.5), (3.0, 1.0), (10.0, 1.0), (0.4, 2.0)]:
        assert np.isclose(es.radial_factor(l, R), -3 * ma_phi_gamma(l, R) / (R ** 3 * l ** 2), rtol=1e-12)
        assert np.isclose(l ** 2 * es.radial_factor(l, R), -3 * cc.solyaev_phi(R / l), rtol=1e-12)
    assert abs(1e6 * es.radial_factor(1e3, 1.0) - 1) < 2e-3          # F l^2 -> 1 as R/l -> 0
    assert es.radial_factor(1.0, 1e4) < 2e-4                          # F -> 0 as R/l -> inf
    assert np.isfinite(es.radial_factor(0.01, 50.0))                   # no overflow


# ------------------------------------------------------------------ B averaged tensors
def test_N_is_P_T_with_P_symmetric_and_in_the_algebra():
    R = 2.0
    l1s, l2s = es.lengths_squared(LAM, MU, A_GEN)
    P = es.polarization_isotropic(LAM, MU, np.sqrt(l1s), np.sqrt(l2s), R)
    assert np.allclose(P["H1"], P["H1"].T)
    N = es.sphere_N_isotropic(LAM, MU, A_GEN, R)
    assert np.abs(N27(N) - N27(P) @ Tma(A_GEN)).max() < 1e-14
    assert abs(N["H1"][0, 1] - N["H1"][1, 0]) > 1e-3                 # <N> itself is not symmetric


@pytest.mark.parametrize("R", [0.5, 2.0, 30.0])
def test_S4_equals_ma_eq319(R):
    ell = 1.2
    s1, s2, _ = ss.ma2018_s1_s2(LAM, MU, ell, R)
    assert np.allclose(es.sphere_S4(LAM, MU, ell, ell, R), (s1, s2), rtol=1e-12)
    # general l1 != l2: eq. (3.19) with Phi(l)Gamma(l)/R^3
    l1, l2 = 0.8, 1.7
    c = LAM + 2 * MU
    g1, g2 = ma_phi_gamma(l1, R) / R ** 3, ma_phi_gamma(l2, R) / R ** 3
    s1m = (3 * LAM - 2 * MU) / (15 * c) + (5 * LAM + 2 * MU) / (5 * c) * g1 - 2 / 5 * g2
    s2m = (3 * LAM + 8 * MU) / (15 * c) + 2 * MU / (5 * c) * g1 + 3 / 5 * g2
    assert np.allclose(es.sphere_S4(LAM, MU, l1, l2, R), (s1m, s2m), rtol=1e-12)


def test_S4_classical_limit():
    nu = LAM / (2 * (LAM + MU))
    s1, s2 = es.sphere_S4(LAM, MU, 1e-3, 1e-3, 1e3)
    assert np.isclose(s1, (5 * nu - 1) / (15 * (1 - nu)), atol=1e-5)
    assert np.isclose(s2, (4 - 5 * nu) / (15 * (1 - nu)), atol=1e-5)


def _trace_V(bl):
    return 7 * bl["H3"][0, 0] + 5 * bl["H2"][0, 0] + 3 * np.trace(bl["H1"])


def test_trace_of_N_tends_to_3_as_R_over_l_tends_to_0():
    """Average of rank-3 projections xi xi (x) u when the gradient term dominates."""
    errs = [abs(_trace_V(es.helmholtz_sphere_N_isotropic(LAM, MU, 1.0, R)) - 3) for R in (1e-1, 1e-2, 1e-3)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-2
    errs_g = abs(_trace_V(es.sphere_N_isotropic(LAM, MU, tuple(1e4 * np.array(A_GEN)), 1e-2)) - 3)
    assert errs_g < 1e-2


def test_helmholtz_closed_form_blocks():
    """Notes 4.19: m3 = -4 Phi (4 lam + 11 mu)/(35 c), m2 = -Phi/5,
    M = [[-Phi (19 lam + 26 mu)/(15 c), 2 sqrt5 Phi (lam - mu)/(15 c)], [-4 sqrt5 Phi/15, -Phi/3]]."""
    for R, ell in [(1.0, 1.0), (5.0, 1.9), (0.3, 2.0)]:
        phi = cc.solyaev_phi(R / ell)
        c = LAM + 2 * MU
        bl = es.helmholtz_sphere_N_isotropic(LAM, MU, ell ** 2, R)
        M = np.array([[-phi * (19 * LAM + 26 * MU) / (15 * c), 2 * np.sqrt(5) * phi * (LAM - MU) / (15 * c)],
                      [-4 * np.sqrt(5) * phi / 15, -phi / 3]])
        assert np.isclose(bl["H3"][0, 0], -4 * phi * (4 * LAM + 11 * MU) / (35 * c))
        assert np.isclose(bl["H2"][0, 0], -phi / 5) and np.allclose(bl["H1"], M)


# ------------------------------------------------------------------ C Ma (2018) and Solyaev (2023)
def _ma_N27(a, R):
    l1s, l2s = es.lengths_squared(LAM, MU, a)
    return ss.ma2018_eq318(cc.ma2018_nhat(LAM, MU, a, np.sqrt(l1s), np.sqrt(l2s), R)) @ IV


@pytest.mark.parametrize("R", [0.7, 3.0])
def test_ma_eq318_coefficients_against_the_average(R):
    a = A_GEN
    cF, rF = coords6(N27(es.sphere_N_isotropic(LAM, MU, a, R)))
    cM, rM = coords6(_ma_N27(a, R))
    assert rF < 1e-14 and rM < 1e-14
    assert np.isclose(cM[0], cF[0], rtol=1e-12)                      # nhat_1 exact
    assert np.isclose(cM[1], cM[2]) and not np.isclose(cF[1], cF[2])  # Ma lumps c2a and c2b
    assert np.isclose(cM[4], 2 * cF[4], rtol=1e-12)                  # nhat_4 = 2 x
    assert np.isclose(cM[5], 4 * cF[5], rtol=1e-12)                  # nhat_5 = 4 x
    assert not np.isclose(cM[3], cF[3])                              # nhat_3 wrong
    # the five tensors of eq. (3.18) cannot represent the average
    X5 = np.column_stack([X6[:, 0], X6[:, 1] + X6[:, 2], X6[:, 3], X6[:, 4], X6[:, 5]])
    c5, *_ = np.linalg.lstsq(X5, (N27(es.sphere_N_isotropic(LAM, MU, a, R)) @ IV).ravel(), rcond=None)
    assert np.abs(X5 @ c5 - (N27(es.sphere_N_isotropic(LAM, MU, a, R)) @ IV).ravel()).max() > 1e-4


def test_ma_polarization_is_not_symmetric():
    from helpers import basis_V
    BV = basis_V()
    T = BV.T @ Tma(A_GEN) @ BV
    for N, ok in ((N27(es.sphere_N_isotropic(LAM, MU, A_GEN, 2.0)), True), (_ma_N27(A_GEN, 2.0), False)):
        P = BV.T @ N @ BV @ np.linalg.inv(T)
        assert bool(np.abs(P - P.T).max() < 1e-13) is ok


@pytest.mark.parametrize("R", [1.0, 5.0])
def test_solyaev_eq21_exact_only_on_c1_and_c4(R):
    ell = 1.9
    exact, _ = coords6(N27(es.helmholtz_sphere_N_isotropic(LAM, MU, ell ** 2, R)))
    printed, r = coords6(ss.solyaev_eq21_form(*cc.solyaev_n(LAM, MU, cc.solyaev_phi(R / ell))))
    assert r < 1e-14
    same = [bool(np.isclose(x, y, rtol=1e-10)) for x, y in zip(exact, printed)]
    assert same == [True, False, False, False, True, False]
    ma, _ = coords6(_ma_N27(cc.ma2018_aifantis_constants(LAM, MU, ell ** 2), R))
    assert [bool(np.isclose(x, y, rtol=1e-10)) for x, y in zip(exact, ma)] == [True] + [False] * 5


def test_trace_limit_fails_for_ma_and_solyaev():
    ell, R = 1.0, 1e-3
    ma = _trace_V(ISO.moduli(_ma_N27(cc.ma2018_aifantis_constants(LAM, MU, ell ** 2), R)))
    sol = _trace_V(ISO.moduli(ss.solyaev_eq21_form(*cc.solyaev_n(LAM, MU, cc.solyaev_phi(R / ell)))))
    c = LAM + 2 * MU
    assert np.isclose(ma, 11 * (31 * LAM + 56 * MU) / (35 * c), rtol=1e-2)
    assert np.isclose(sol, 2 * (74 * LAM + 125 * MU) / (35 * c), rtol=1e-2)
    assert abs(ma - 3) > 1 and abs(sol - 3) > 0.5


# ------------------------------------------------------------------ exact
@pytest.mark.exact
def test_exact_moment_and_T_blocks():
    alg = isotropic.algebra(exact=True)
    M6, A4 = moment_tensors(exact=True)
    b6, b4 = alg.moduli(M6), alg.moduli(A4)
    s5 = sp.sqrt(5)
    assert is_zero(alg.residual(M6)) and is_zero(alg.residual(A4))
    assert b6["H3"][0, 0] == sp.Rational(2, 35) and b6["H2"][0, 0] == 0 and is_zero(b6["H1"] - sp.diag(sp.Rational(1, 5), 0))
    assert b4["H3"][0, 0] == sp.Rational(2, 15) and b4["H2"][0, 0] == sp.Rational(2, 15)
    assert is_zero(b4["H1"] - sp.Matrix([[11, 2 * s5], [2 * s5, 10]]) / 45)
    a = sp.symbols("a1:6", real=True)
    T = sp.Matrix(ss.ma2018_eq26(*[1, 0, 0, 0, 0])) * 0
    for q in range(5):
        e = [0] * 5
        e[q] = 1
        T += a[q] * sp.Matrix(ss.ma2018_eq26(*e)).applyfunc(sp.nsimplify)
    T = sp.Matrix(IV).applyfunc(sp.nsimplify) * T * sp.Matrix(IV).applyfunc(sp.nsimplify)
    bT, ref = alg.moduli(T), es.ma2018_T_isotropic(a)
    assert all(is_zero(bT[k] - ref[k]) for k in ref) and is_zero(alg.residual(T))


@pytest.mark.exact
def test_exact_radial_integral_and_classical_normalization():
    xi, R, l = sp.symbols("xi R l", positive=True)
    Ex = sp.exp(-2 * R / l)
    J = sp.pi / 4 * (2 * l * R * Ex - l ** 2 * (1 - Ex) + R ** 2 * (1 + Ex)) / l
    PhiG = -l * (l + R) * sp.exp(-R / l) * (R * sp.cosh(R / l) - l * sp.sinh(R / l))
    assert sp.simplify((J + sp.pi / (2 * l ** 2) * PhiG).rewrite(sp.exp)) == 0
    assert sp.simplify((es.radial_factor(l, R) - 6 * J / (sp.pi * R ** 3)).rewrite(sp.exp)) == 0
    assert sp.simplify(sp.integrate((sp.sin(xi) - xi * sp.cos(xi)) ** 2 / xi ** 4, (xi, 0, sp.oo)) - sp.pi / 6) == 0


@pytest.mark.exact
def test_exact_S4_from_moments_equals_ma():
    lam, mu, G1, G2 = sp.symbols("lambda mu G1 G2", positive=True)
    C = lambda i, k, m, n: lam * int(i == k) * int(m == n) + mu * (int(i == m) * int(k == n) + int(i == n) * int(k == m))  # noqa: E731
    ok = True
    for s, t, m, n in itertools.product(range(3), repeat=4):
        val = 0
        for aa, bb in ((s, t), (t, s)):
            for k, i in itertools.product(range(3), repeat=2):
                aL = moment((bb, aa, k, i), True)
                aT = moment((bb, i), True) * int(aa == k) - aL
                val += (G1 / (lam + 2 * mu) * aL + G2 / mu * aT) * C(i, k, m, n) / 2
        g1, g2 = (G1 - 1) / 3, (G2 - 1) / 3    # = Phi Gamma / R^3, with G = 1 - l^2 F
        c = lam + 2 * mu
        s1 = (3 * lam - 2 * mu) / (15 * c) + (5 * lam + 2 * mu) / (5 * c) * g1 - sp.Rational(2, 5) * g2
        s2 = (3 * lam + 8 * mu) / (15 * c) + 2 * mu / (5 * c) * g1 + sp.Rational(3, 5) * g2
        ref = s1 * int(s == t) * int(m == n) + s2 * (int(s == m) * int(t == n) + int(s == n) * int(t == m))
        ok &= sp.simplify(val - ref) == 0
    assert ok


# ------------------------------------------------------------------ slow
@pytest.mark.slow
@pytest.mark.parametrize("R,l", [(1.0, 0.5), (1.0, 2.0), (3.0, 1.0)])
def test_radial_factor_by_quadrature(R, l):
    f = lambda x: (np.sin(x * R) - x * R * np.cos(x * R)) ** 2 / (x ** 2 * (1 + l * l * x * x))  # noqa: E731
    X = 4000.0 / R
    edges = np.linspace(0, X, 4001)
    J = sum(integrate.quad(f, edges[q], edges[q + 1], limit=200)[0] for q in range(4000)) + R * R / (2 * l * l * X)
    assert np.isclose(6 * J / (np.pi * R ** 3), es.radial_factor(l, R), rtol=1e-6)
