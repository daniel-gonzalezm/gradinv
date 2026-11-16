"""Second, independent verification of the exact ball average of the gradient
Eshelby tensor (gradinv.eshelby): real-space route
with the Green function of Ma et al. (2018) written with the potentials psi,
phi, M, and the location of their error (eq. 3.12 is right; eq. 3.18 is not).
"""
import itertools

import numpy as np
import pytest
import sympy as sp
from scipy import integrate

from gradinv import eshelby, isotropic
from gradinv.concentrator import ma2018_nhat, ma2018_s6_isotropic
from references import ma2018_real_space as mr
from references import solyaev_sources as ss

ISO = isotropic.algebra()
MEDIA = [(1.0, 0.8, (0.13, -0.05, 0.07, 0.31, 0.12), 1.4),
         (2.5, 1.1, (0.40, 0.10, -0.02, 0.55, 0.30), 0.9),
         (0.6, 1.3, (-0.10, 0.05, 0.20, 0.25, -0.08), 2.7)]


@pytest.mark.exact
def test_ma_green_function_is_the_inverse_acoustic_tensor():
    lam, mu = sp.symbols("lambda mu", positive=True)
    a = sp.symbols("a1:6")
    xi = sp.symbols("xi1:4", real=True)
    basis = [sp.Matrix(ss.ma2018_eq26(*e)).applyfunc(sp.nsimplify) for e in np.eye(5)]
    T = sum((ai * Bi for ai, Bi in zip(a, basis)), sp.zeros(27, 27))
    K = sp.zeros(3, 3)
    for k, n in itertools.product(range(3), repeat=2):
        kc = sum((lam * int(i == k) * int(m == n) + mu * int(i == m) * int(k == n) + mu * int(i == n) * int(k == m)) * xi[i] * xi[m]
                 for i in range(3) for m in range(3))
        kt = sum(sp.sympify(T[9 * i + 3 * j + k, 9 * l + 3 * m + n]) * xi[i] * xi[j] * xi[l] * xi[m]
                 for i, j, l, m in itertools.product(range(3), repeat=4))
        K[k, n] = sp.expand(kc + kt)
    l1s, l2s = mr.lengths_sq(lam, mu, a)
    G = mr.green_hat(lam, mu, l1s, l2s, xi)
    assert sp.simplify(K * G - sp.eye(3)) == sp.zeros(3, 3)


def test_yukawa_potential_inside_the_ball():
    """M = (1/4pi) int_ball e^{-r/k}/r: inside, k^2 + k Phi(k) sinh(r/k)/r (eq. B.5),
    from the ODE (lap - k^-2) M = -chi with C^1 matching, and by direct integration."""
    r, R, k = sp.symbols("r R k", positive=True)
    A, B = sp.symbols("A B")
    Min, Mout = k ** 2 + A * sp.sinh(r / k) / r, B * sp.exp(-r / k) / r
    sol = sp.solve([(Min - Mout).subs(r, R), sp.diff(Min - Mout, r).subs(r, R)], [A, B], dict=True)[0]
    assert sp.simplify(sol[A] + k ** 2 * (k + R) * sp.exp(-R / k)) == 0
    kv, Rv = 0.7, 1.3
    for dist in (0.2, 0.75, 1.1):
        inner = lambda s: 2 * np.pi * s * kv / dist * (np.exp(-abs(dist - s) / kv) - np.exp(-(dist + s) / kv))
        direct = integrate.quad(inner, 0, Rv, points=[dist], limit=200)[0] / (4 * np.pi)
        formula = kv ** 2 + kv * mr.Phi(kv, Rv) * np.sinh(dist / kv) / dist
        assert abs(direct - formula) < 1e-12


def test_ball_averages_of_the_derivatives_of_M():
    """Eq. (B.12), by the divergence theorem on the sphere (surface quadrature)."""
    X, Y, Z = sp.symbols("X Y Z", real=True)
    kv, Rv = 0.7, 1.3
    rr = sp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    M = kv ** 2 + kv * mr.Phi(kv, Rv) * sp.sinh(rr / kv) / rr
    gt, wt = np.polynomial.legendre.leggauss(40)
    ph = 2 * np.pi * np.arange(64) / 64
    CT, PH = np.meshgrid(gt, ph, indexing="ij")
    W = wt[:, None] * (2 * np.pi / 64) * Rv ** 2
    ST = np.sqrt(1 - CT ** 2)
    N = (ST * np.cos(PH), ST * np.sin(PH), CT)
    vol = 4 * np.pi * Rv ** 3 / 3

    def avg(counts):
        ax = next(i for i, c in enumerate(counts) if c)
        rest = list(counts); rest[ax] -= 1
        f = sp.lambdify((X, Y, Z), sp.diff(M, X, rest[0], Y, rest[1], Z, rest[2]), "numpy")
        return float(np.sum(W * N[ax] * f(*(Rv * c for c in N))) / vol)
    q2, q4, q6 = mr.ball_averages(kv, Rv)
    assert np.isclose(avg((2, 0, 0)), q2, rtol=1e-12)
    for c, v in {(4, 0, 0): 3 * q4, (2, 2, 0): q4, (3, 1, 0): 0}.items():
        assert abs(avg(c) - v) < 1e-11 * abs(q4)
    for c, v in {(6, 0, 0): 15 * q6, (4, 2, 0): 3 * q6, (2, 2, 2): q6, (4, 1, 1): 0}.items():
        assert abs(avg(c) - v) < 1e-10 * abs(q6)


def _real_space_average(lam, mu, a, R):
    """<N> = A T with A_{rts,ijk} = <d_r d_t d_i d_j g_ks>,
    g_ks = phi d_ks/mu - (l+m)/(2m(l+2m)) psi_,ks + l1^2/(l+2m)(phi - M1)_,ks
           - l2^2/mu (phi - M2)_,ks - M2 d_ks/mu, with d^6 psi = d^4 phi = 0 inside."""
    l1s, l2s = mr.lengths_sq(lam, mu, a)
    _, q41, q61 = mr.ball_averages(np.sqrt(l1s), R)
    _, q42, q62 = mr.ball_averages(np.sqrt(l2s), R)
    A = mr.ten(lambda r, t, s, i, j, k: (-l1s / (lam + 2 * mu) * q61 + l2s / mu * q62) * mr.pairing_sum(r, t, i, j, k, s)
               - mr.d[k, s] * q42 * mr.pairing_sum(r, t, i, j) / mu)
    return A @ ss.ma2018_eq26(*a)


@pytest.mark.core
@pytest.mark.parametrize("lam, mu, a, R", MEDIA)
def test_real_space_average_equals_the_package(lam, mu, a, R):
    IV = ISO.assemble(isotropic.from_lazar2022([0, 0, 0, 0.5, 0]))      # identity on V
    N_rs = IV @ _real_space_average(lam, mu, a, R) @ IV
    N_pkg = ISO.assemble(eshelby.sphere_N_isotropic(lam, mu, a, R))
    assert np.linalg.norm(N_rs - N_pkg) < 1e-13 * np.linalg.norm(N_pkg)


@pytest.mark.core
@pytest.mark.parametrize("lam, mu, a, R", MEDIA)
def test_ma_eq_312_is_right_and_eq_318_is_not(lam, mu, a, R):
    """Averaging Ma's own J (eq. 3.12) gives the exact <N>; their eq. (3.18) does not:
    the error enters when the average is collected into five terms."""
    IV = ISO.assemble(isotropic.from_lazar2022([0, 0, 0, 0.5, 0]))
    N_pkg = ISO.assemble(eshelby.sphere_N_isotropic(lam, mu, a, R))
    N_312 = IV @ mr.eq312_average(lam, mu, a, R) @ IV
    assert np.linalg.norm(N_312 - N_pkg) < 1e-13 * np.linalg.norm(N_pkg)
    l1s, l2s = mr.lengths_sq(lam, mu, a)
    N_318 = ISO.assemble(ma2018_s6_isotropic(ma2018_nhat(lam, mu, a, np.sqrt(l1s), np.sqrt(l2s), R)))
    assert np.linalg.norm(N_318 - N_pkg) > 0.5 * np.linalg.norm(N_pkg)
