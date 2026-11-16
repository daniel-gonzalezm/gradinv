"""Kroner's self-consistent estimate for cubic grains against two read sources.

gradinv.selfconsistent.kroner_nu (used for the C of Kroner in the application) is
checked against
- Blaschke (2017), eq. (3.7): the cubic equation for the shear modulus (attributed to
  Kroner) and its Table VI (lambda, mu from the Table III data of Al, Cu, Fe, Nb;
  printed to 0.1 GPa);
- the root of the same cubic for four crystals near a mechanical instability, with
  the DFT constants printed by Kube and de Jong (2016), tables I and II (AlV3:
  c44 almost zero; LiSiNi2 and Li3Pd: c11 close to c12; SrVO3), and for their W (DFT),
  nearly isotropic. These only test the routine; they are not data of the paper.
Kube and de Jong (2016), eq. (16b) (Hershey's quartic reduced to Kroner's cubic) is
the same equation in another form; it is not transcribed here.
"""
import numpy as np
import pytest

from gradinv.selfconsistent import kroner_nu


def _mu_lambda(C):
    c11, c12, _ = C
    k = (c11 + 2 * c12) / 3
    nu = kroner_nu(C)
    mu = 3 * k * (1 - 2 * nu) / (2 * (1 + nu))
    return mu, k - 2 * mu / 3


def _blaschke_eq37(C):
    """Positive real root of 8 mu^3 + mu^2 (5 c11 + 4 c12) + mu (4 c12 - 7 c11) c44
    - c44 (c11 - c12)(c11 + 2 c12) = 0."""
    c11, c12, c44 = C
    r = np.roots([8, 5 * c11 + 4 * c12, (4 * c12 - 7 * c11) * c44, -c44 * (c11 - c12) * (c11 + 2 * c12)])
    pos = [x.real for x in r if abs(x.imag) < 1e-9 and x.real > 0]
    assert len(pos) == 1
    return pos[0]


# Blaschke (2017), Table III (c11, c12, c44 in GPa) and Table VI (lambda, mu in GPa)
BLASCHKE = {"Al": ((106.75, 60.41, 28.34), (58.4, 26.2)),
            "Cu": ((166.1, 119.9, 75.6), (103.3, 47.9)),
            "Fe": ((226.0, 140.0, 116.0), (116.2, 78.7)),
            "Nb": ((246.5, 133.3, 28.4), (146.0, 37.6))}

# Kube and de Jong (2016), tables I and II (c11, c12, c44 in GPa; DFT, Materials Project)
KUBE_DEJONG = {"AlV3": (212.16, 132.92, 2.41), "LiSiNi2": (159.78, 146.44, 110.20),
               "Li3Pd": (39.79, 35.54, 30.00), "SrVO3": (223.69, 158.62, 102.29),
               "W": (509.82, 201.00, 142.69)}


@pytest.mark.core
@pytest.mark.parametrize("name", sorted(BLASCHKE))
def test_blaschke_table_vi(name):
    C, (lam_printed, mu_printed) = BLASCHKE[name]
    mu, lam = _mu_lambda(C)
    assert abs(mu - mu_printed) <= 0.05 + 1e-9
    assert abs(lam - lam_printed) <= 0.05 + 1e-9
    assert np.isclose(mu, _blaschke_eq37(C), rtol=1e-10)


@pytest.mark.parametrize("name", sorted(KUBE_DEJONG))
def test_near_instability_and_isotropic_cases(name):
    C = KUBE_DEJONG[name]
    mu, _lam = _mu_lambda(C)
    assert np.isclose(mu, _blaschke_eq37(C), rtol=1e-10)
    # Kroner's mu lies between Reuss and Voigt (classical bounds)
    c11, c12, c44 = C
    mu_v = (c11 - c12 + 3 * c44) / 5
    mu_r = 5 * c44 * (c11 - c12) / (4 * c44 + 3 * (c11 - c12))
    assert mu_r - 1e-9 <= mu <= mu_v + 1e-9
