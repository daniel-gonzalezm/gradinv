"""Solyaev's (2023) self-consistent scheme (gradinv.selfconsistent): Kroner part in
closed form, consistency limits, the Fe example with the three <S6> tensors, and
the weight of the sixth condition with full cubic grains (exact average).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gradinv import concentrator as cc
from gradinv import cubic, eshelby
from gradinv import selfconsistent as sc
from gradinv.algebra import BlockAlgebra as BA
from references import solyaev_sources as ss

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LZ, PO = DATA["datasets"]["lazar2022"], DATA["datasets"]["po2019"]
GRAINS = {"Al": (LZ["C"]["Al"], cubic.a_from_voigt(LZ["D_voigt"]["Al"])),
          "Cu_EAM": (PO["C"]["Cu_EAM"], cubic.a_from_voigt(PO["D_voigt"]["Cu_EAM"]))}
CUB = cubic.algebra()

FE = ss.FE_EXAMPLE
FE_C = (FE["C11"], FE["C12"], FE["C44"])
FE_K = (FE["C11"] + 2 * FE["C12"]) / 3
FE_G = sc.helmholtz_cubic_grain(*FE_C)
KINDS = ("printed", "ma", "exact")


# ------------------------------------------------------------------ fourth order (Kroner)
def _C4(C11, C12, C44, Q=np.eye(3)):
    d = np.eye(3)
    C = np.zeros((3, 3, 3, 3))
    for i, j, k, l in np.ndindex(3, 3, 3, 3):
        C[i, j, k, l] = (C12 * d[i, j] * d[k, l] + C44 * (d[i, k] * d[j, l] + d[i, l] * d[j, k])
                         + (C11 - C12 - 2 * C44) * (i == j == k == l))
    return np.einsum("ai,bj,ck,dl,ijkl->abcd", Q, Q, Q, Q, C)


def test_kroner_invariants_closed_form_against_matrix_inversion():
    d = np.eye(3)
    I4 = 0.5 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
    DD = np.einsum("ij,kl->ijkl", d, d)
    sym = I4.reshape(9, 9)
    w, v = np.linalg.eigh(sym)
    Q4 = v[:, w > 0.5]
    red = lambda X: Q4.T @ X.reshape(9, 9) @ Q4
    rng = np.random.default_rng(36)
    for _ in range(5):
        Cg = (rng.uniform(150, 250), rng.uniform(60, 140), rng.uniform(30, 120))
        lam, mu = rng.uniform(40, 120), rng.uniform(20, 80)
        s1, s2 = rng.uniform(-0.1, 0.1), rng.uniform(0.1, 0.4)
        Q = Rotation.random(random_state=rng).as_matrix()
        Cgr = _C4(*Cg, Q)
        C = lam * DD + 2 * mu * I4
        S4 = s1 * DD + s2 * 2 * I4
        A6 = np.linalg.inv(np.eye(6) + red(S4) @ np.linalg.inv(red(C)) @ red(Cgr - C))
        A = (Q4 @ A6 @ Q4.T).reshape(3, 3, 3, 3)
        ref = sc.kroner_invariants(Cg, lam, mu, s1, s2)
        assert np.isclose(np.einsum("iijj", A), ref["A1"], rtol=1e-12)
        assert np.isclose(np.einsum("ijij", A), ref["A2"], rtol=1e-12)


def test_classical_limit_gives_kroner_nu_of_solyaev():
    """Large grains: <S4> classical; A2 = 6 gives nu = 0.34853 (Solyaev prints 0.348)
    and mu = 26.06 GPa; his mu_clas = 26.16 GPa is eq. (24) with nu rounded to 0.348."""
    from scipy.optimize import brentq

    def a2(nu):
        lam, mu = sc.lame_from_k_nu(FE_K, nu)
        s1, s2 = eshelby.sphere_S4(lam, mu, 1.0, 1.0, 1e6)
        return sc.kroner_invariants(FE_C, lam, mu, s1, s2)["A2"] - 6
    nu = brentq(a2, 0.2, 0.45)
    assert round(nu, 5) == 0.34853 and abs(nu - FE["nu"]) < 1e-3
    assert round(sc.lame_from_k_nu(FE_K, nu)[1], 2) == 26.06
    assert round(sc.lame_from_k_nu(FE["k"], FE["nu"])[1], 2) == 26.16


# ------------------------------------------------------------------ cross-implementation
@pytest.mark.parametrize("kind", KINDS)
def test_fe_invariants_match_the_independent_index_implementation(kind):
    inv = sc.invariants(FE_C, FE_G, FE_K, FE["nu"], 1.9, 5.0, kind)
    ref = dict(zip(("A1", "A2", "B1", "B2", "B3", "B4", "B5", "B2'"), ss.FE_INVARIANTS_INDEX[kind]))
    assert all(abs(inv[t] - ref[t]) < 6e-5 for t in ref)


def test_fe_recorded_figures_still_hold():
    """Notes 4.14: B1 = 7.29 (printed <S6>) and 11.56 (Ma's); with the exact average 6.61."""
    b1 = {k: sc.invariants(FE_C, FE_G, FE_K, FE["nu"], 1.9, 5.0, k)["B1"] for k in KINDS}
    assert [round(b1[k], 2) for k in KINDS] == [7.29, 11.56, 6.61]


def test_sixth_condition_is_not_negligible_with_the_exact_average():
    """With the exact <S6> the Fe example itself has B2 - B2' = 0.65 at nu = 0.348,
    l = 1.9, 2R = 10 (with Ma's and the printed tensors, < 1.1e-3)."""
    inv = sc.invariants(FE_C, FE_G, FE_K, FE["nu"], 1.9, 5.0, "exact")
    assert round(inv["B2"] - inv["B2'"], 3) == 0.653


# ------------------------------------------------------------------ consistency limits
@pytest.mark.core
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("R", [2.5, 5.0, 25.0])
def test_isotropic_grain_is_an_exact_solution(kind, R):
    """Isotropic grain l0^2 C_gr (x) delta: the medium (nu_gr, l0) gives B = I and A = I,
    so rho = 0 with 7 and with 8 terms, for every <S6>."""
    C11, C12 = FE["C11"], FE["C12"]
    Cg = (C11, C12, (C11 - C12) / 2)
    nu_gr = C12 / (C11 + C12)
    inv = sc.invariants(Cg, sc.helmholtz_cubic_grain(*Cg), (C11 + 2 * C12) / 3, nu_gr, 1.0, R, kind)
    assert sc.error(inv, sc.TERMS_8) < 1e-12


def test_eq22_unsquared_is_not_an_error_function():
    """With the five B terms not squared (as printed), the sum under the root
    is negative at some points."""
    T = sc.TARGETS
    sums = []
    for nu in (0.15, 0.2, 0.25):
        inv = sc.invariants(FE_C, FE_G, FE_K, nu, 0.6, 5.0, "exact")
        sums.append(sum(((inv[t] - T[t]) / T[t]) ** 2 for t in ("A1", "A2"))
                    + sum((inv[t] - T[t]) / T[t] for t in sc.TERMS_7[2:]))
    assert max(sums) < 0


# ------------------------------------------------------------------ the Fe example
@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_fe_example_coherent_reading_gives_l_close_to_l0(kind):
    """With grain and medium both read as form II Helmholtz models, the
    minimum is l = l0 (to 2e-3), nu = Kroner's (0.3485) and rho < 5e-4, for
    every <S6>; Solyaev's Fig. 1-2 give l = 1.19 and 1.86 l0 with rho 0.375, 0.12."""
    for twoR in (10, 50):
        D, nu, ell = sc.solve(FE_C, FE_G, twoR / 2, kind)
        assert D < 5e-4 and abs(ell - 1) < 2e-3 and abs(nu - 0.3485) < 2e-3
        l_fig, d_fig = ss.FIG1_MIN[twoR]
        assert abs(ell - l_fig) > 0.18 and d_fig > 100 * D


@pytest.mark.slow
def test_fe_strongly_anisotropic_grain_still_gives_l_close_to_l0():
    """Fe with Zener ratio 2.46 (C11, C12, C44 = 230, 135, 117 GPa), exact <S6>:
    l = 1.01-1.02 l0 for 2R = 10 and 50 (the 'about twice' of Solyaev is not obtained)."""
    Cg = (230.0, 135.0, 117.0)
    for twoR in (10, 50):
        D, nu, ell = sc.solve(Cg, sc.helmholtz_cubic_grain(*Cg), twoR / 2, "exact")
        assert D < 3e-3 and 1.005 < ell < 1.03


# ------------------------------------------------------------------ sixth condition, exact average
def test_full_cubic_grains_sixth_condition_weight_exact():
    """Fixed grid (Al MEAM, Cu EAM; Helmholtz medium with the Voigt C of the grain;
    l = 1, 1.5, 2 A; R = 5, 10, 20 A) with the exact <S6>: B2 - B2' from -0.649
    (Cu EAM, l = 1, R = 5) to 0.410 (Al, l = 2, R = 5); |B2 - B2'|/3 up to 22 %;
    the sign changes between l = 1 and 2 A. (Ma's tensor gave 0.10-0.49.)"""
    diffs = []
    for name, (C, a) in GRAINS.items():
        C11, C12, C44 = C
        lamV, muV = (C11 + 4 * C12 - 2 * C44) / 5, (C11 - C12 + 3 * C44) / 5
        grain = cc.to_form_I(CUB, cubic.from_lazar2022(a), "stiffness")
        for ell in (1.0, 1.5, 2.0):
            for R_ in (5.0, 10.0, 20.0):
                G = sc.helmholtz_medium(lamV, muV, ell ** 2)
                P = BA.multiply(sc.averaged_S6("exact", lamV, muV, ell, R_), BA.inverse(G))
                inv = cc.invariants(CUB, cc.concentrator(CUB, grain, G, P))
                diffs.append(inv["B2"] - inv["B2'"])
    diffs = np.array(diffs)
    assert round(diffs.min(), 3) == -0.649 and round(diffs.max(), 3) == 0.410
    assert round(np.abs(diffs).max() / 3, 2) == 0.22 and round(np.abs(diffs).min(), 3) == 0.007


# ------------------------------------------------------------------ large-grain limit
def test_phi_stable_form_equals_printed_form():
    """concentrator.solyaev_phi uses an overflow-free form of the printed Phi(R/l)."""
    for Rbar in (0.05, 0.7, 3.0, 25.0, 200.0):
        printed = Rbar ** -3 * (1 + Rbar) * np.exp(-Rbar) * (np.sinh(Rbar) - Rbar * np.cosh(Rbar))
        assert np.isclose(cc.solyaev_phi(Rbar), printed, rtol=1e-10, atol=1e-15)
    assert np.isfinite(cc.solyaev_phi(5000.0)) and np.isclose(-2 * 5000 * cc.solyaev_phi(5000.0), 1, rtol=1e-3)


def test_exact_S6_of_helmholtz_medium_is_phi_times_a_fixed_tensor():
    """<S6> exact = Phi(R/l) S_hat(lam, mu), with S_hat independent of l and R (basis of
    selfconsistent.large_grain_limit)."""
    lam, mu = 0.6, 0.35
    ref = {p: v / cc.solyaev_phi(10.0) for p, v in sc.averaged_S6("exact", lam, mu, 1.0, 10.0).items()}
    for ell, R_ in ((0.7, 3.0), (2.0, 40.0), (1.3, 900.0)):
        phi = cc.solyaev_phi(R_ / ell)
        for p, v in sc.averaged_S6("exact", lam, mu, ell, R_).items():
            assert np.allclose(v / phi, ref[p], rtol=1e-10, atol=1e-12)


def test_kroner_nu_closes_the_classical_equation():
    for C in (LZ["C"]["Al"], PO["C"]["Cu_EAM"], (230.0, 135.0, 117.0)):
        k = (C[0] + 2 * C[1]) / 3
        nu = sc.kroner_nu(C)
        lam, mu = sc.lame_from_k_nu(k, nu)
        c = lam + 2 * mu
        A = sc.kroner_invariants(C, lam, mu, (3 * lam - 2 * mu) / (15 * c), (3 * lam + 8 * mu) / (15 * c))
        assert np.isclose(A["A1"], 3) and np.isclose(A["A2"], 6, atol=1e-12)


def test_large_grain_limit_isotropic_grain_is_exact():
    """Isotropic Helmholtz grain (l0): the limit is (nu_grain, l0) with zero residual."""
    C11, C12 = 1.0, 0.4
    C = (C11, C12, (C11 - C12) / 2)
    for l0 in (0.8, 1.7):
        out = sc.large_grain_limit(C, sc.helmholtz_cubic_grain(*C, l2=l0 ** 2))
        assert np.isclose(out["nu"], C12 / (C11 + C12)) and np.isclose(out["ell"], l0)
        assert out["residual"] < 1e-12


def test_large_grain_limit_fe_example_close_to_l0():
    """Solyaev's nearly isotropic Fe (Zener 1.23): l_inf = l0 to 2e-3 (consistent with
    the full minimization)."""
    out = sc.large_grain_limit(FE_C, FE_G)
    assert abs(out["ell"] - 1) < 2e-3 and abs(out["nu"] - 0.3485) < 5e-4


# The check of the large-grain limit against the full minimization at R = 1000 A
# is in tests/test_application.py, for the three rows of the application table.
