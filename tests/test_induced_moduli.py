"""The moduli induced by the heterogeneity of C (Malyi 2021) and their comparison with the
Voigt estimate of D (analysis/induced_moduli.py; section 6(b) of the paper).

Consistency: the form II tensor has the energy of Malyi's form I tensor on random strain
gradients (an index contraction written out separately from the 27 x 27 matrices); it is
isotropic; it vanishes for an isotropic crystal and grows as R^2; L^2 = 2R^2/5 is the
integral of Malyi's w(r). Cross-comparison: Malyi's statement that the moduli scale as
c^2 R^2 / mu. The macros of the text against the module.
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from gradinv import forms

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = dict(re.findall(r"\\newcommand\{\\(Num[A-Za-z]*)\}\{([^}]*)\}",
                          (ROOT / "paper" / "paper_numbers.tex").read_text()))


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


im = _load("induced_moduli")
CU = im.inputs.C("CuMEAM")


def _malyi_energy(C, R, kappa):
    """B^{pqv}_{yzw} kappa_pqv kappa_yzw / 2 term by term from eq. (10.21), kappa symmetric
    in its last two indices."""
    B1, B2, B3, B4, B5 = im.malyi_coefficients(*C, R)
    k = kappa
    a = np.einsum("ppv->v", k)                # kappa_ppv
    b = np.einsum("pqq->p", k)                # kappa_pqq
    kk = np.einsum("pqv,pqv", k, k)
    x = np.einsum("pqv,qpv", k, k)            # kappa_pqv kappa_qpv
    # each delta product contracted by hand, using kappa_pqv = kappa_pvq
    e = (B1 * a @ a
         + B2 * (kk + x)
         + B3 * (2 * a @ b + 2 * a @ a)
         + B4 * (2 * a @ b + a @ a + b @ b)
         + B5 * (3 * x + kk))
    return 0.5 * e


def test_form_II_tensor_has_the_energy_of_malyi_form_I():
    rng = np.random.default_rng(3)
    for C, R in ((CU, 3.0), ((1.7, 1.2, 0.5), 7.0)):
        D = im.induced_tensor(C, R)
        for _ in range(5):
            eta = rng.normal(size=(3, 3, 3))
            eta = 0.5 * (eta + eta.transpose(1, 0, 2))
            kappa = forms.form2_to_form1(eta)
            assert np.isclose(0.5 * eta.reshape(27) @ D @ eta.reshape(27),
                              _malyi_energy(C, R, kappa), rtol=1e-12)


def test_induced_tensor_is_isotropic_vanishes_without_anisotropy_and_grows_as_R2():
    D = im.induced_tensor(CU, 4.0)
    assert np.abs(im.ISO.residual(D)).max() < 1e-12 * np.abs(D).max()
    C11, C12 = 1.2, 0.6
    assert np.abs(im.induced_tensor((C11, C12, (C11 - C12) / 2), 5.0)).max() < 1e-14
    b1, b3 = im.induced_blocks(CU, 1.0), im.induced_blocks(CU, 3.0)
    for p in b1:
        assert np.allclose(np.asarray(b3[p]), 9 * np.asarray(b1[p]), rtol=1e-12)


def test_malyi_scaling_c2_R2_over_mu():
    # doubling c at fixed lambda_V and mu_V (C12 and C44 adjusted) multiplies B by four
    def C_of(c, lam=0.6, mu=0.4):
        # C11 + 4 C12 - 2 C44 = 5 lam, C11 - C12 + 3 C44 = 5 mu, C11 - C12 - 2 C44 = c
        A = np.array([[1, 4, -2], [1, -1, 3], [1, -1, -2]], float)
        return tuple(np.linalg.solve(A, [5 * lam, 5 * mu, c]))
    n1, n2 = (im.norm_V(im.induced_blocks(C_of(c), 2.0)) for c in (0.1, 0.2))
    assert np.isclose(n2 / n1, 4.0, rtol=1e-12)


def test_L2_is_the_integral_of_malyi_w():
    R = 3.0
    w = lambda r: 1 - 3 * r / (4 * R) + r ** 3 / (16 * R ** 3)   # noqa: E731
    L2 = quad(lambda r: r * w(r), 0, 2 * R)[0]
    assert np.isclose(L2, 2 * R ** 2 / 5, rtol=1e-12)
    c, mu_v = CU[0] - CU[1] - 2 * CU[2], (CU[0] - CU[1] + 3 * CU[2]) / 5
    lam_v = (CU[0] + 4 * CU[1] - 2 * CU[2]) / 5
    k = (lam_v + mu_v) / (lam_v + 2 * mu_v)
    B5 = im.malyi_coefficients(*CU, R)[4]
    assert np.isclose(B5, -4 * L2 * c ** 2 / (18375 * mu_v) * (7 - 89 * k + 12 * k * k), rtol=1e-12)


def test_crossover_macros():
    s = im.summary()
    for k, v in s.items():
        assert NUMBERS[f"Num{k}InducedCrossoverNm"] == f"{v['crossover_diameter_nm']:.1f}"
        # at the crossover radius the two norms are equal
        R = v["crossover_radius_A"]
        voigt = im.av.voigt(im.CUB, im.cubic.from_lazar2022(im.cubic.a_from_voigt(im.inputs.D(k))))
        assert np.isclose(im.norm_V(im.induced_blocks(im.inputs.C(k), R)), im.norm_V(voigt), rtol=1e-10)
    assert s["Al"]["crossover_diameter_nm"] > 5 * s["CuEAM"]["crossover_diameter_nm"]
