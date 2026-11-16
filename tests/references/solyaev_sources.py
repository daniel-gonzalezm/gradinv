"""Index-form transcriptions for the concentration tensor.

Solyaev (2023, Mech. Res. Commun. 132:104162): eqs. (3), (13), (17), (19), (21);
Ma et al. (2018, Int. J. Eng. Sci. 132:60-78): eqs. (2.6), (3.18);
Gao and Ma (2009, Acta Mech. 207:163-181): eq. (3.3), the Helmholtz model in
form II (equations checked on the rendered pages: Ma pp. 61, 65, 66; Gao and
Ma p. 164; Solyaev eqs. 20-21). Every tensor is a 27 x 27 float array with
rows (i, j, k) and columns (l, m, n) in Solyaev's form I order
kappa_ijk = d_i d_j u_k, unless stated otherwise.
"""
import itertools
import json
from pathlib import Path

import numpy as np

d = np.eye(3)


def ten(fn):
    M = np.zeros((27, 27))
    for (i, j, m) in itertools.product(range(3), repeat=3):
        for (k, l, n) in itertools.product(range(3), repeat=3):
            M[9 * i + 3 * j + m, 9 * k + 3 * l + n] = fn(i, j, m, k, l, n)
    return M


def Isym(a, b, c, e):
    return 0.5 * (d[a, c] * d[b, e] + d[a, e] * d[b, c])


def solyaev_identity():
    """I(6)_ijklmn = I_ijlm delta_kn (Solyaev 2023, below eq. 7)."""
    return ten(lambda i, j, k, l, m, n: Isym(i, j, l, m) * d[k, n])


def solyaev_contractions(A27):
    """Solyaev's B1..B5 (eq. 13) and the sixth invariant B2' as index contractions."""
    B6 = np.asarray(A27).reshape((3,) * 6)
    return {"B1": np.einsum("ijjikk", B6), "B2": np.einsum("iikkjj", B6),
            "B2'": np.einsum("ijjkki", B6), "B3": np.einsum("iikjjk", B6),
            "B4": np.einsum("ijkijk", B6), "B5": np.einsum("ijkkji", B6)}


def solyaev_contraction_tensors():
    """The six contraction patterns as 27 x 27 integer tensors J (invariant = <B, J>)."""
    def kd(a, b):
        return 1 if a == b else 0
    pats = {"B1": lambda a, b, c, e, f, g: kd(a, e) * kd(b, c) * kd(f, g),
            "B2": lambda a, b, c, e, f, g: kd(a, b) * kd(c, e) * kd(f, g),
            "B2'": lambda a, b, c, e, f, g: kd(b, c) * kd(e, f) * kd(a, g),
            "B3": lambda a, b, c, e, f, g: kd(a, b) * kd(e, f) * kd(c, g),
            "B4": lambda a, b, c, e, f, g: kd(a, e) * kd(b, f) * kd(c, g),
            "B5": lambda a, b, c, e, f, g: kd(a, g) * kd(b, f) * kd(c, e)}
    return {k: ten(f).astype(int) for k, f in pats.items()}


def solyaev_eq3(A1, A2, A3, A4, A5):
    """Solyaev (2023), eq. (3): isotropic G with constants A1..A5 (form I)."""
    return ten(lambda i, j, k, l, m, n: A1 / 2 * (Isym(l, m, i, n) * d[j, k] + Isym(l, m, j, n) * d[i, k])
               + A2 / 2 * (Isym(i, j, k, n) * d[l, m] + Isym(l, m, k, n) * d[i, j])
               + 2 * A3 * d[i, j] * d[l, m] * d[k, n]
               + A4 * Isym(i, j, l, m) * d[k, n]
               + A5 / 2 * (Isym(i, j, l, n) * d[k, m] + Isym(i, j, m, n) * d[k, l]))


def solyaev_eq19(lam, mu, l2):
    """Eq. (19) (and eq. 17 with the grain C) read literally with form I indices:
    G_ijklmn = l^2 C_ijlm delta_kn, C isotropic (lam, mu)."""
    return ten(lambda i, j, k, l, m, n: l2 * (lam * d[i, j] * d[l, m]
                                              + mu * (d[i, l] * d[j, m] + d[i, m] * d[j, l])) * d[k, n])


def solyaev_eq17_cubic(C11, C12, C44, l2=1.0):
    """Eq. (17) with a cubic grain C, literally with form I indices: l^2 C_ijlm delta_kn."""
    def C(i, j, l, m):
        return (C12 * d[i, j] * d[l, m] + C44 * (d[i, l] * d[j, m] + d[i, m] * d[j, l])
                + (C11 - C12 - 2 * C44) * (1.0 if i == j == l == m else 0.0))
    return ten(lambda i, j, k, l, m, n: l2 * C(i, j, l, m) * d[k, n])


def solyaev_eq21_form(n1, n2, n3, n4, n5):
    """The form of eq. (21) with coefficients n1..n5 (Solyaev's I = (dd + dd)/2)."""
    return ten(lambda i, j, k, l, m, n: n1 * d[i, j] * d[l, m] * d[k, n]
               + n2 * (Isym(i, j, k, n) * d[l, m] + Isym(l, m, k, n) * d[i, j])
               + n3 * (Isym(i, j, l, n) * d[k, m] + Isym(i, j, m, n) * d[k, l])
               + n4 * Isym(i, j, l, m) * d[k, n]
               + n5 * (Isym(l, m, i, n) * d[j, k] + Isym(l, m, j, n) * d[i, k]))


def helmholtz_form_II(lam, mu, l2):
    """Gao and Ma (2009), eq. (3.3) = Lazar and Po: l^2 C_ijkl delta_mn in FORM II
    (rows (i, j, m) of eta_ijm = d_m eps_ij), C isotropic."""
    return ten(lambda i, j, m, k, l, n: l2 * (lam * d[i, j] * d[k, l]
                                              + mu * (d[i, k] * d[j, l] + d[i, l] * d[j, k])) * d[m, n])


def ma2018_eq26(a1, a2, a3, a4, a5):
    """Ma et al. (2018), eq. (2.6): isotropic tensor with constants a1..a5;
    eta_ijk = u_k,ij (Solyaev's index order)."""
    return ten(lambda i, j, k, l, m, n: a1 / 2 * (d[i, l] * d[j, k] * d[m, n] + d[i, m] * d[j, k] * d[l, n]
                                                + d[i, k] * d[j, m] * d[l, n] + d[i, k] * d[j, l] * d[m, n])
               + a2 / 2 * (d[i, j] * d[l, k] * d[m, n] + d[i, j] * d[m, k] * d[l, n] + d[i, n] * d[j, k] * d[l, m]
                           + d[i, k] * d[j, n] * d[m, l]) + 2 * a3 * d[i, j] * d[k, n] * d[l, m]
               + a4 * (d[i, m] * d[j, l] * d[k, n] + d[i, l] * d[j, m] * d[k, n])
               + a5 / 2 * (d[i, n] * d[j, m] * d[k, l] + d[i, n] * d[j, l] * d[k, m] + d[i, m] * d[j, n] * d[l, k]
                           + d[i, l] * d[j, n] * d[m, k]))


def ma2018_eq318(nh):
    """Ma et al. (2018), eq. (3.18): <N^_rtslmn>, rows (r, t, s), columns (l, m, n)."""
    n1, n2, n3, n4, n5 = nh
    return ten(lambda r, t, s, l, m, n: n1 * d[s, n] * d[m, l] * d[r, t]
               + n2 * (d[n, r] * d[t, s] * d[l, m] + d[n, t] * d[r, s] * d[l, m] + d[s, l] * d[m, n] * d[r, t]
                       + d[s, m] * d[n, l] * d[r, t])
               + n3 * (d[s, l] * d[m, r] * d[t, n] + d[s, l] * d[m, t] * d[r, n] + d[s, m] * d[l, r] * d[t, n]
                       + d[s, m] * d[l, t] * d[r, n])
               + n4 * (d[s, n] * d[m, r] * d[l, t] + d[s, n] * d[m, t] * d[l, r])
               + n5 * (d[m, n] * d[r, s] * d[l, t] + d[m, n] * d[t, s] * d[l, r] + d[l, n] * d[t, s] * d[m, r]
                       + d[l, n] * d[r, s] * d[m, t]))


def ma2018_s1_s2(lam, mu, ell, R_):
    """Ma et al. (2018), the fourth-order Eshelby coefficients s1, s2 of the sphere
    (term s2 (dd + dd) = 2 s2 I in Solyaev's notation: his printed s2 I is half of it)."""
    c = lam + 2 * mu
    gx = -ell * (ell + R_) * np.exp(-R_ / ell) * (R_ * np.cosh(R_ / ell) - ell * np.sinh(R_ / ell)) / R_ ** 3
    s1 = (3 * lam - 2 * mu) / (15 * c) + (5 * lam + 2 * mu) / (5 * c) * gx - 2 / 5 * gx
    s2 = (3 * lam + 8 * mu) / (15 * c) + 2 * mu / (5 * c) * gx + 3 / 5 * gx
    return s1, s2, gx


# Solyaev's Fe example (section 5): C in GPa, bulk modulus k and Poisson ratio of
# the classical self-consistent medium; l0 = 1.
# Single source: analysis/data/published_reference_values.json.
FE_EXAMPLE = {k: json.loads((Path(__file__).resolve().parents[2] / "analysis" / "data"
                             / "published_reference_values.json").read_text())["solyaev2023_fe"][k]
              for k in ("C11", "C12", "C44", "k", "nu")}

# Solyaev (2023), Fig. 2 (p. 5), digitized from the page rendered at 220 dpi
# (marker centroids; axes calibrated on the tick labels; uncertainty
# about +-0.01 in l/l0 and +-0.002 in nu). 2R/l0 -> value. The nu markers at 90
# and 100 are hidden behind the dashed classical line.
FIG2_ELL = {10: 1.186, 15: 1.405, 20: 1.563, 30: 1.725, 40: 1.805, 50: 1.863,
            60: 1.884, 70: 1.905, 80: 1.913, 90: 1.925, 100: 1.925}
FIG2_NU = {10: 0.189, 15: 0.267, 20: 0.301, 30: 0.327, 40: 0.335, 50: 0.341,
           60: 0.343, 70: 0.345, 80: 0.346}
# Fig. 1: minima of the error function (black dots), 2R/l0 -> (l/l0, rho).
FIG1_MIN = {10: (1.19, 0.375), 50: (1.86, 0.12)}

# The eight invariants of the Fe example at nu = 0.348, l = 1.9, 2R = 10
# from the self-contained index implementation (analysis/solyaev_variants.py,
# 27 x 27 tensors, 9 x 9 Kroner concentrator), order A1, A2, B1, B2, B3, B4, B5, B2'.
FE_INVARIANTS_INDEX = {
    "printed": (3.0, 6.0018, 7.2870, 4.6441, 11.0174, 19.6941, 7.6217, 4.6440),
    "ma": (3.0, 6.0018, 11.5580, 9.5310, 15.8683, 24.3258, 12.9347, 9.5321),
    "exact": (3.0, 6.0018, 6.6111, 4.0576, 10.2142, 19.1589, 6.7949, 3.4048),
}
