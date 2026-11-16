"""Transcriptions of Ma et al. (2018) used by tests/test_eshelby_real_space.py, cotejadas en la página renderizada (pp. 62-64, 72-74):
Green function, eqs. (2.16)-(2.18), as Fourier transforms; the interior Yukawa
potential, eq. (B.5); the ball averages (B.12); and J_slmn, eq. (3.12), whose
ball average N^_rtslmn = <J_slmn,rt> is assembled here.
Index order: form I, eta_ijk = u_k,ij; 27 x 27 storage, row (a,b,c) -> 9a+3b+c.
"""
import itertools

import numpy as np
import sympy as sp

d = np.eye(3)


def _pairings(idx):
    if not idx:
        yield []
        return
    for p in range(1, len(idx)):
        for m in _pairings(idx[1:p] + idx[p + 1:]):
            yield [(idx[0], idx[p])] + m


def pairing_sum(*ind):
    """Sum over the perfect matchings of products of Kronecker deltas (3 terms for
    four indices, 15 for six): Ma's A~ and E~ of eq. (B.11)."""
    return sum(np.prod([d[ind[a], ind[b]] for a, b in m]) for m in _pairings(list(range(len(ind)))))


def ten(f):
    X = np.zeros((27, 27))
    for a, b, c, e, g, h in itertools.product(range(3), repeat=6):
        X[9 * a + 3 * b + c, 9 * e + 3 * g + h] = f(a, b, c, e, g, h)
    return X


def lengths_sq(lam, mu, a):
    a1, a2, a3, a4, a5 = a
    return 2 * (a1 + a2 + a3 + a4 + a5) / (lam + 2 * mu), 2 * (a3 + a4) / mu


def green_hat(lam, mu, L1s, L2s, xi):
    """Fourier transform of eqs. (2.17)-(2.18), using 1/(4 pi r) -> 1/xi^2,
    -r/(8 pi) -> 1/xi^4, exp(-r/l)/(4 pi r) -> 1/(xi^2 + l^-2), d_k d_s -> -xi_k xi_s."""
    xi2 = sum(x ** 2 for x in xi)
    nn = sp.Matrix(xi) * sp.Matrix(xi).T
    return (sp.eye(3) / (mu * xi2) - (lam + mu) / (mu * (lam + 2 * mu)) * nn / xi2 ** 2
            - L1s / (lam + 2 * mu) * nn * (1 / xi2 - 1 / (xi2 + 1 / L1s))
            + L2s / mu * nn * (1 / xi2 - 1 / (xi2 + 1 / L2s))
            - sp.eye(3) / (mu * (xi2 + 1 / L2s)))


def Phi(k, R):
    return -k * (k + R) * np.exp(-R / k)


def Gamma(k, R):
    return R * np.cosh(R / k) - k * np.sinh(R / k)


def ball_averages(k, R):
    """Eq. (B.12): <M,ij> = PG/R^3 d, <M,ijkl> = PG/(5R^3k^2) A~, <M,ijklmn> = PG/(35R^3k^4) E~."""
    PG = Phi(k, R) * Gamma(k, R)
    return PG / R ** 3, PG / (5 * R ** 3 * k ** 2), PG / (35 * R ** 3 * k ** 4)


def eq312_average(lam, mu, a, R):
    """<N^_rtslmn> = <J_slmn,rt> with J of eq. (3.12) as printed, using
    d^6 psi = d^4 phi = 0 inside the ball and the averages (B.12)."""
    a1, a2, a3, a4, a5 = a
    l1s, l2s = lengths_sq(lam, mu, a)
    q1, q2 = ball_averages(np.sqrt(l1s), R), ball_averages(np.sqrt(l2s), R)
    M2 = lambda q, i, j: q[0] * d[i, j]
    M4 = lambda q, *ii: q[1] * pairing_sum(*ii)
    M6 = lambda q, *ii: q[2] * pairing_sum(*ii)
    L = lam + 2 * mu

    def f(r, t, s, l, m, n):
        return (-2 * (a4 + a5) / L * l1s * M6(q1, s, l, m, n, r, t)
                - (2 * a1 + a2) / (2 * L) * (d[m, n] * M4(q1, s, l, r, t) + d[l, n] * M4(q1, s, m, r, t))
                - (2 * a3 + a2) / L * d[m, l] * M4(q1, s, n, r, t)
                + 2 * (a4 + a5) / mu * l2s * M6(q2, s, l, m, n, r, t)
                + a2 / (2 * mu) * (d[m, n] * M4(q2, s, l, r, t) + d[l, n] * M4(q2, s, m, r, t))
                + 2 * a3 / mu * d[m, l] * M4(q2, s, n, r, t)
                - 2 * a4 / mu * d[s, n] * M4(q2, m, l, r, t)
                - a5 / mu * d[s, l] * M4(q2, m, n, r, t) - a5 / mu * d[s, m] * M4(q2, l, n, r, t)
                - a2 / (2 * mu * l2s) * (d[m, n] * d[s, l] + d[l, n] * d[s, m]) * M2(q2, r, t)
                - 2 * a3 / (mu * l2s) * d[s, n] * d[m, l] * M2(q2, r, t))
    return ten(f)
