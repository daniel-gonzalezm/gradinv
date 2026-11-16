"""Self-contained helpers for analysis/solyaev_variants.py (do not import gradinv).

Storage: a sixth-order tensor X_{abc,def} is a 27x27 matrix, row (a,b,c) -> 9a+3b+c.
Form I with the index order of Ma et al. (2018) and Solyaev (2023):
eta_{ijk} = u_{k,ij} (symmetric in the first two indices).
"""
import itertools
import numpy as np
import sympy as sp

R3 = range(3)


def idx(a, b, c):
    return 9 * a + 3 * b + c


def d(i, j):
    return 1 if i == j else 0


def from_function(f, mod=np):
    """27x27 matrix with entries f(a,b,c,d,e,g)."""
    if mod is np:
        X = np.zeros((27, 27))
    else:
        X = sp.zeros(27, 27)
    for a, b, c, dd, e, g in itertools.product(R3, repeat=6):
        X[idx(a, b, c), idx(dd, e, g)] = f(a, b, c, dd, e, g)
    return X


def I4(i, j, k, l):
    return sp.Rational(1, 2) * (d(i, k) * d(j, l) + d(i, l) * d(j, k))


def identity_formI(mod=np):
    """I6_{ijklmn} = I_{ijlm} delta_{kn}: identity on V (form I)."""
    return from_function(lambda i, j, k, l, m, n: float(I4(i, j, l, m)) * d(k, n)
                         if mod is np else I4(i, j, l, m) * d(k, n), mod)


# ---------- Ma et al. (2018), eq. (2.6): T_{ijklmn} --------------------------
def ma_T(a, mod=np):
    a1, a2, a3, a4, a5 = a
    half = 0.5 if mod is np else sp.Rational(1, 2)

    def f(i, j, k, l, m, n):
        return (a1 * half * (d(i, l) * d(j, k) * d(m, n) + d(i, m) * d(j, k) * d(l, n)
                             + d(i, k) * d(j, m) * d(l, n) + d(i, k) * d(j, l) * d(m, n))
                + a2 * half * (d(i, j) * d(l, k) * d(m, n) + d(i, j) * d(m, k) * d(l, n)
                               + d(i, n) * d(j, k) * d(l, m) + d(i, k) * d(j, n) * d(m, l))
                + 2 * a3 * d(i, j) * d(k, n) * d(l, m)
                + a4 * (d(i, m) * d(j, l) * d(k, n) + d(i, l) * d(j, m) * d(k, n))
                + a5 * half * (d(i, n) * d(j, m) * d(k, l) + d(i, n) * d(j, l) * d(k, m)
                               + d(i, m) * d(j, n) * d(l, k) + d(i, l) * d(j, n) * d(m, k)))
    return from_function(f, mod)


def ma_lengths_sq(lam, mu, a):
    """Eq. (2.8): l1^2, l2^2."""
    a1, a2, a3, a4, a5 = a
    return 2 * (a1 + a2 + a3 + a4 + a5) / (lam + 2 * mu), 2 * (a3 + a4) / mu


def helmholtz_a(lam, mu, ell):
    """Eq. (4.3) of Ma et al. (Aifantis = Helmholtz model): a_i."""
    h = ell ** 2 / 2
    return (lam * h, 0 * h, 0 * h, mu * h, mu * h)


def Phi_ma(k, R, mod=np):
    return -k * (k + R) * mod.exp(-R / k)


def Gamma_ma(k, R, mod=np):
    return R * mod.cosh(R / k) - k * mod.sinh(R / k)


def F_radial(l, R, mod=np):
    """F(l) = -3 Phi(l) Gamma(l) / (R^3 l^2)."""
    return -3 * Phi_ma(l, R, mod) * Gamma_ma(l, R, mod) / (R ** 3 * l ** 2)


def solyaev_Phi(Rb, mod=np):
    """Solyaev (2023), below eq. (21): Phi(Rbar), Rbar = R/l."""
    return Rb ** -3 * (1 + Rb) * mod.exp(-Rb) * (mod.sinh(Rb) - Rb * mod.cosh(Rb))


# ---------- angular moments <n^4>, <n^6> ------------------------------------
def pairings(idxs):
    """All perfect matchings of a list of positions."""
    if not idxs:
        yield []
        return
    first, rest = idxs[0], idxs[1:]
    for p in range(len(rest)):
        pair = (first, rest[p])
        for m in pairings(rest[:p] + rest[p + 1:]):
            yield [pair] + m


def delta_sym(ind):
    return sum(np.prod([d(ind[p], ind[q]) for p, q in m]) for m in pairings(list(range(len(ind)))))


def n4(i, j, k, l):
    return delta_sym((i, j, k, l)) / 15.0


def n6(*ind):
    return delta_sym(ind) / 105.0


def fourier_P(lam, mu, l1, l2, R):
    """Exact <N> = P T: P = F1/(lam+2mu) <n6> + F2/mu (<n4> x delta - <n6>).

    Index layout of P: rows (r,t,s) = output eta_{rts} = u_{s,rt};
    columns (i,j,k) contracted with T_{ijk lmn}; the delta is between s and k.
    """
    F1, F2 = F_radial(l1, R), F_radial(l2, R)
    return from_function(lambda r, t, s, i, j, k:
                         F1 / (lam + 2 * mu) * n6(r, t, s, i, j, k)
                         + F2 / mu * (n4(r, t, i, j) * d(s, k) - n6(r, t, s, i, j, k)))


# ---------- Ma et al. (2018), eqs. (3.18)-(3.19), as printed -----------------
def ma_nhat(lam, mu, a, R, mod=np):
    a1, a2, a3, a4, a5 = a
    l1s, l2s = ma_lengths_sq(lam, mu, a)
    l1, l2 = mod.sqrt(l1s), mod.sqrt(l2s)
    X2 = Phi_ma(l2, R, mod) * Gamma_ma(l2, R, mod) / (35 * mu * R ** 3 * l2s)
    X1 = Phi_ma(l1, R, mod) * Gamma_ma(l1, R, mod) / (35 * (lam + 2 * mu) * R ** 3 * l1s)
    n1 = X2 * 2 * (a5 - 6 * a4 - 28 * a3) - X1 * (14 * a3 + 7 * a2 + 2 * (a4 + a5))
    n2 = X2 * (8 * a4 - 13 * a5 - 28 * a2 + 28 * a3) - X1 * (8 * (a4 + a5) + 7 * (2 * a1 + 3 * a2 + 4 * a3))
    n3 = X2 * (8 * a4 - 38 * a5) - X1 * 8 * (a4 + a5)
    n4_ = X2 * 4 * (a5 - 6 * a4) - X1 * 4 * (a4 + a5)
    n5 = X2 * 2 * (7 * a2 + 4 * a4 + 4 * a5) - X1 * (8 * (a4 + a5) + 14 * (2 * a1 + a2))
    return n1, n2, n3, n4_, n5


# The six isotropic pairing classes on V, in Ma's index order <N>_{rts,lmn}
# (rows r,t,s; columns l,m,n). c2a and c2b are the two transposed halves of
# Ma's n2 term: c2a carries delta_rt (output trace), c2b carries delta_lm
# (input trace).
def CLASS(name):
    table = {
        "c1": lambda r, t, s, l, m, n: d(s, n) * d(m, l) * d(r, t),
        "c2a": lambda r, t, s, l, m, n: (d(s, l) * d(m, n) + d(s, m) * d(n, l)) * d(r, t),
        "c2b": lambda r, t, s, l, m, n: (d(n, r) * d(t, s) + d(n, t) * d(r, s)) * d(l, m),
        "c3": lambda r, t, s, l, m, n: (d(s, l) * d(m, r) * d(t, n) + d(s, l) * d(m, t) * d(r, n)
                                        + d(s, m) * d(l, r) * d(t, n) + d(s, m) * d(l, t) * d(r, n)),
        "c4": lambda r, t, s, l, m, n: d(s, n) * (d(m, r) * d(l, t) + d(m, t) * d(l, r)),
        "c5": lambda r, t, s, l, m, n: (d(m, n) * d(r, s) * d(l, t) + d(m, n) * d(t, s) * d(l, r)
                                        + d(l, n) * d(t, s) * d(m, r) + d(l, n) * d(r, s) * d(m, t)),
    }
    return from_function(table[name])


CLASS_NAMES = ("c1", "c2a", "c2b", "c3", "c4", "c5")


def ma_N_printed(lam, mu, a, R):
    n1, n2, n3, n4_, n5 = ma_nhat(lam, mu, a, R)
    return (n1 * CLASS("c1") + n2 * (CLASS("c2a") + CLASS("c2b")) + n3 * CLASS("c3")
            + n4_ * CLASS("c4") + n5 * CLASS("c5"))


def class_coefficients(X, Pi):
    """Least-squares coefficients of Pi X Pi on the six classes (restricted to V)."""
    cols = [(Pi @ CLASS(c) @ Pi).ravel() for c in CLASS_NAMES]
    A = np.array(cols).T
    coef, res, rank, _ = np.linalg.lstsq(A, (Pi @ X @ Pi).ravel(), rcond=None)
    resid = np.linalg.norm(A @ coef - (Pi @ X @ Pi).ravel())
    return coef, rank, resid


class Checker:
    def __init__(self):
        self.n = 0
        self.fail = 0

    def check(self, ok, msg):
        self.n += 1
        if ok:
            print(f"[PASS] {msg}")
        else:
            self.fail += 1
            print(f"[FAIL] {msg}")
        return ok

    def done(self):
        print(f"\n{self.n - self.fail} PASS, {self.fail} FAIL")
        raise SystemExit(1 if self.fail else 0)
