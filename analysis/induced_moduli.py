"""Gradient moduli induced by the heterogeneity of the classical stiffness (Malyi 2021),
compared with the orientation average of the crystal gradient tensor.

Malyi (2021, eqs. 10.21-10.22) gives, to second order in the cubic anisotropy
c = C11 - C12 - 2 C44, the isotropic gradient tensor B of an untextured polycrystal of
spherical grains of radius R, in form I (energy B^{pqv}_{yzw} u_{p,qv} u_{y,zw} / 2),
with L^2 = 2 R^2 / 5 (his w(r) for spherical grains, p. 153). Here it is taken to form II
(kappa = T eta, gradinv.forms), restricted to V and written in isotropic blocks; it grows
as R^2. The crossover radius is the R at which its norm on V equals that of the Voigt
estimate of D (section 6 of the paper): above it the moduli that no average of D contains
dominate. The estimate is perturbative in c and assumes spherical grains with independent
random orientations; it is an order of magnitude.

Units: C in eV/A^3, R in A, blocks in eV/A.

Run: python analysis/induced_moduli.py
"""
import itertools
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import cubic, forms, isotropic  # noqa: E402
from gradinv.skeleton import identity_V  # noqa: E402

import inputs  # noqa: E402

ISO, CUB = isotropic.algebra(), cubic.algebra()
MATERIALS = ("Al", "CuMEAM", "CuEAM")


def _d(a, b):
    return 1.0 if a == b else 0.0


def malyi_coefficients(C11, C12, C44, R):
    """B1..B5 of Malyi (2021), eq. (10.22), with L^2 = 2R^2/5."""
    c = C11 - C12 - 2 * C44
    lam_v, mu_v = (C11 + 4 * C12 - 2 * C44) / 5, (C11 - C12 + 3 * C44) / 5
    k = (lam_v + mu_v) / (lam_v + 2 * mu_v)
    Cm = (2 * R ** 2 / 5) * c ** 2 / (18375 * mu_v)
    return (-4 * Cm * (105 - 82 * k + 152 * k * k), 4 * Cm * (140 - 37 * k + 212 * k * k),
            2 * Cm * (35 - 172 * k + 32 * k * k), -Cm * (77 - 160 * k + 48 * k * k),
            -4 * Cm * (7 - 89 * k + 12 * k * k))


def malyi_form_I(C11, C12, C44, R):
    """27 x 27 matrix of B^{pqv}_{yzw} (rows 9p + 3q + v), Malyi (2021), eq. (10.21)."""
    B1, B2, B3, B4, B5 = malyi_coefficients(C11, C12, C44, R)
    d = _d
    B = np.zeros((27, 27))
    for p, q, v, y, z, w in itertools.product(range(3), repeat=6):
        B[9 * p + 3 * q + v, 9 * y + 3 * z + w] = (
            B1 * d(p, q) * d(y, z) * d(v, w)
            + B2 * (d(p, y) * d(q, z) * d(v, w) + d(p, z) * d(q, y) * d(v, w))
            + B3 * (d(p, q) * d(v, y) * d(z, w) + d(p, q) * d(v, z) * d(y, w)
                    + d(p, v) * d(q, w) * d(y, z) + d(p, w) * d(q, v) * d(y, z))
            + B4 * (d(p, v) * d(q, y) * d(z, w) + d(p, v) * d(q, z) * d(y, w)
                    + d(p, y) * d(q, v) * d(z, w) + d(p, z) * d(q, v) * d(y, w))
            + B5 * (d(p, w) * d(q, y) * d(z, v) + d(p, w) * d(q, z) * d(y, v)
                    + d(p, y) * d(q, w) * d(z, v) + d(p, z) * d(q, w) * d(y, v)))
    return B


@lru_cache(maxsize=None)
def _maps():
    T = np.zeros((27, 27))                      # kappa = T eta, kappa_pqv = u_{p,qv}
    for n in range(27):
        e = np.zeros(27)
        e[n] = 1.0
        T[:, n] = forms.form2_to_form1(e.reshape(3, 3, 3)).reshape(27)
    Ps = np.zeros((27, 27))                     # symmetrizer in the derivative indices
    for p, q, v in itertools.product(range(3), repeat=3):
        Ps[9 * p + 3 * q + v, 9 * p + 3 * q + v] += 0.5
        Ps[9 * p + 3 * q + v, 9 * p + 3 * v + q] += 0.5
    return T, Ps, identity_V()


def induced_tensor(C, R):
    """Form II tensor on V (27 x 27) with the energy of Malyi's B."""
    T, Ps, IV = _maps()
    B = Ps @ malyi_form_I(*C, R) @ Ps
    B = 0.5 * (B + B.T)
    return IV @ (T.T @ B @ T) @ IV


def induced_blocks(C, R):
    return ISO.moduli(induced_tensor(C, R))


def norm_V(blocks):
    """Frobenius norm on V of an isotropic element given by its blocks."""
    return float(np.sqrt(sum(d * np.sum(np.asarray(blocks[p], float) ** 2) for p, _c, d in ISO.irreps)))


def crossover_radius(key):
    """R (A) at which the induced tensor and the Voigt estimate of D have equal norms."""
    C = inputs.C(key)
    voigt = av.voigt(CUB, cubic.from_lazar2022(cubic.a_from_voigt(inputs.D(key))))
    return float(np.sqrt(norm_V(voigt) / norm_V(induced_blocks(C, 1.0))))


def ratio_at(key, R):
    """Norm of the induced tensor over that of the Voigt estimate, at radius R (A)."""
    return (R / crossover_radius(key)) ** 2


def summary():
    return {k: {"crossover_radius_A": crossover_radius(k),
                "crossover_diameter_nm": 2 * crossover_radius(k) / 10,
                "ratio_at_diameter_100nm": ratio_at(k, 500.0)} for k in MATERIALS}


if __name__ == "__main__":
    for k, v in summary().items():
        print(f"{k:7s} crossover R = {v['crossover_radius_A']:.2f} A (2R = {v['crossover_diameter_nm']:.2f} nm);"
              f" ratio at 2R = 100 nm: {v['ratio_at_diameter_100nm']:.3g}")
