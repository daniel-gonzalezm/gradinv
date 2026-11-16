"""Main results: block lengths, the rank-one
H1 difference of Voigt and Reuss in the cubic class, the results scripts for
the cubic data and the granular hexagonal tensors, and the generated tables."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gradinv import averages as av
from gradinv import cubic, hexagonal, isotropic
from gradinv import lengths as L
from gradinv._tensor import rotation_27
from helpers import random_rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LZ = DATA["datasets"]["lazar2022"]
CUB, ISO, HEX = cubic.algebra(), isotropic.algebra(), hexagonal.algebra()


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cubic_res():
    return _load("cubic_results").results()


@pytest.fixture(scope="module")
def hex_res():
    return _load("hexagonal_granular").results()


def _rand_cubic_pd(rng):
    bl = {}
    for p, c, _d in cubic.IRREPS:
        A = rng.normal(size=(len(c), len(c)))
        bl[p] = A @ A.T + 0.1 * np.eye(len(c))
    return bl


# ------------------------------------------------------------------ block lengths
@pytest.mark.parametrize("name", ["Al", "Cu", "Fe"])
def test_helmholtz_grain_has_equal_block_lengths_in_every_estimate(name):
    """Limit case: D = l0^2 C (x) delta gives l_p = l0 for the crystal and for
    Voigt, Reuss and Hill paired with the same classical estimate."""
    C, l0 = LZ["C"][name], 1.37
    D = L.helmholtz_cubic(*C, l2=l0 ** 2)
    ref = L.helmholtz_cubic(*C)
    assert all(np.allclose(v, l0, atol=1e-12) for v in L.block_lengths(D, ref).values())
    cl = L.cubic_classical_estimates(*C)
    for est, f in (("voigt", av.voigt), ("reuss", av.reuss), ("hill", av.hill)):
        lens = L.block_lengths(f(CUB, D), L.helmholtz_isotropic(*cl[est]))
        assert all(np.allclose(v, l0, atol=1e-12) for v in lens.values()), est
        assert abs(L.lazar_po_length(ISO, f(CUB, D), L.helmholtz_isotropic(*cl[est])) - l0) < 1e-12


def test_classical_estimates_of_an_isotropic_C_coincide():
    lam, mu = 0.4, 0.3
    est = L.cubic_classical_estimates(lam + 2 * mu, lam, mu)
    assert all(np.allclose(v, (lam, mu)) for v in est.values())


@pytest.mark.parametrize("name", ["Al", "Cu"])
def test_lazar_po_length_is_their_eq107_and_the_published_value(name):
    """l_LP^2 = tr_V(G_hat^-1 D)/18 = S_ijkl D_ijmklm / 18 (Lazar and Po 2024, eq. 107)."""
    fcc = _load("selfconsistent_fcc")
    C = LZ["C"][name]
    a = cubic.a_from_voigt(LZ["D_voigt"][name])
    lp = L.lazar_po_length(CUB, cubic.from_lazar2022(a), L.helmholtz_cubic(*C))
    direct = fcc.lazar_po_length(cubic.lazar2022_tensor(a), fcc._C4(*C))
    assert abs(lp - direct) < 1e-12
    assert abs(lp - DATA["datasets"]["lazarpo2024"]["ell_table2"][name]) < 1e-4


def test_lazar_po_length_is_the_trace_and_the_weighted_rms():
    """On a random cubic element: l_LP^2 = tr(G_hat^+ D)/18 on the 27 x 27 storage, and it
    is the dimension-weighted mean of the squared block lengths; rotating both keeps it."""
    rng = np.random.default_rng(4301)
    C = LZ["C"]["Al"]
    D = _rand_cubic_pd(rng)
    ref = L.helmholtz_cubic(*C)
    lens = L.block_lengths(D, ref)
    rms = np.sqrt(sum(d * np.sum(lens[p] ** 2) for p, _c, d in cubic.IRREPS) / 18)
    lp = L.lazar_po_length(CUB, D, ref)
    assert abs(lp - rms) < 1e-12
    D27, G27 = CUB.assemble(D), CUB.assemble(ref)
    assert abs(np.sqrt(np.trace(np.linalg.pinv(G27, rcond=1e-10) @ D27) / 18) - lp) < 1e-10
    RQ = rotation_27(random_rotation(rng))
    tr = np.trace(np.linalg.pinv(RQ @ G27 @ RQ.T, rcond=1e-10) @ (RQ @ D27 @ RQ.T))
    assert abs(np.sqrt(tr / 18) - lp) < 1e-10


def test_block_lengths_are_nan_for_a_non_positive_block():
    a = cubic.a_from_voigt(LZ["D_voigt"]["Fe"])
    lens = L.block_lengths(cubic.from_lazar2022(a), L.helmholtz_cubic(*LZ["C"]["Fe"]))
    assert np.isnan(lens["T2"]).sum() == 1
    assert all(np.isfinite(lens[p]).all() for p in ("T1", "E", "A2"))


# ------------------------------------------------------------------ rank-one H1 difference
@pytest.mark.core
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cubic_voigt_minus_reuss_on_H1_is_rank_one(seed):
    """M^V = M_(sr), M^R = Schur complement of M_hh, so M^V - M^R = m_h m_h^T / M_hh:
    one generalized eigenvalue of (M^R, M^V) is exactly 1."""
    bl = _rand_cubic_pd(np.random.default_rng(4310 + seed))
    V, R = av.voigt(CUB, bl), av.reuss(CUB, bl)
    M = bl["T1"]
    mh = M[1:, 0:1]
    assert np.abs(V["H1"] - R["H1"] - mh @ mh.T / M[0, 0]).max() < 1e-12
    w = np.linalg.eigvals(np.linalg.solve(V["H1"], R["H1"])).real
    assert abs(w.max() - 1) < 1e-12 and w.min() < 1


# ------------------------------------------------------------------ cubic results
def test_cubic_results_checks(cubic_res):
    d = cubic_res["datasets"]
    assert all(d[k]["crystal"]["compliance_check_18x18"] < 1e-12 for k in d)
    assert d["Al"]["averages"]["voigt"]["table6_max_abs_diff"] < 1e-5
    assert d["W"]["averages"]["voigt"]["table6_max_abs_diff"] < 1e-5
    assert d["Fe"]["crystal"]["non_positive_blocks"] == ["T2"]
    assert d["W"]["crystal"]["non_positive_blocks"] == ["T2"]
    for k in ("Al", "CuMEAM", "CuEAM"):
        rv = d[k]["averages"]["reuss_over_voigt"]
        assert 0 < rv["m3"] < 1 and 0 < rv["m2"] < 1 and abs(rv["H1"][1] - 1) < 1e-12
        assert d[k]["averages"]["voigt_minus_reuss_H1_rank_one_residual"] < 1e-12
    assert abs(d["CuMEAM"]["averages"]["reuss_mE_ratio"] - 2.4960) < 5e-5   # inputs: analysis/inputs.py


# ------------------------------------------------------------------ hexagonal granular
def test_hexagonal_granular_proposition_3(hex_res):
    for e, s in hex_res["sweeps"].items():
        assert s["voigt_max_variation"] < 1e-12, e
        assert s["reuss_evenness"] < 1e-12, e
        assert s["reuss_non_H3_variation"] < 1e-12, e
        assert s["reuss_closed_form_error"] < 1e-12, e
        c = s["checks"]
        assert c["ti_rotation_residual_delta0"] < 1e-12 and c["ti_rotation_residual_delta08"] > 1e-3
        assert c["delta_from_H6_error"] < 1e-12
        assert abs(c["eta_auffray_over_delta"] - 0.25) < 1e-12
        assert c["compliance_check_18x18"] < 1e-12 and c["positive_definite_delta0"]


def test_hexagonal_isotropic_base_closed_value(hex_res):
    """Isotropic base: m3^R(Delta = sigma)/m3 = 21/23 (c = 5/m3, sigma = m3)."""
    rows = hex_res["sweeps"]["iso"]["rows"]
    one = [r for r in rows if abs(r["delta_over_sigma"] - 1) < 1e-9][0]
    assert abs(one["reuss"][0] / rows[len(rows) // 2]["reuss"][0] - 21 / 23) < 1e-12


def test_hexagonal_isotropic_base_curve_is_universal():
    """The isotropic-base curve m3^R(Delta)/m3^R(0) does not depend on the base: granular
    isotropic fabric and the Voigt estimate of Al MEAM give the same ratios."""
    hx = _load("hexagonal_granular")
    cr = _load("cubic_results")
    al = av.voigt(cr.CUB, cr.crystal("Al")[1])
    bases = [hx.base_blocks("iso"), av.embed_isotropic(HEX, al)]
    for f in (0.3, 1.0, 1.7):
        r = [hx.reuss_m3_closed(b, f * b["B1"][0, 0]) / hx.reuss_m3_closed(b, 0.0) for b in bases]
        assert abs(r[0] - r[1]) < 1e-12


@pytest.mark.core
def test_granular_hexagonal_tensor(hex_res):
    """Granular micromechanics of Pirmoradi et al. (2024): uniform contacts give their
    eq. (74); the sixfold harmonic a66 moves only the B1/B2 split, with
    Delta = (1440/13)(k_n - k_s) a66; a66 = 0 is transversely isotropic (D and C)."""
    g = hex_res["granular_checks"]
    assert g["eq74_max_abs_diff"] < 1e-15
    for key in ("iso", "ti"):
        c = g[key]
        assert c["distance_to_algebra"] < 1e-14 and c["positive_definite"]
        assert c["ti_residual_D_a66_0"] < 1e-14 and c["ti_residual_C"] < 1e-14
        assert abs(c["delta_a66_0"]) < 1e-14
        assert c["delta_coeff_error"] < 1e-9 and c["other_blocks_change"] < 1e-14 and c["sigma_change"] < 1e-14
        assert 0.15 < c["delta_max_over_sigma"] < 0.3
    for k in ("delta_coeff_error_kn1_ks0", "delta_coeff_error_kn0_ks1"):
        assert g[k] < 1e-9
    assert hex_res["delta_coefficient"] == "1440/13"


# ------------------------------------------------------------------ generated tables
def test_tables_match_the_generator(cubic_res):
    gen = _load("make_tables")
    for name, text in gen.render_all(cubic_res).items():
        path = ROOT / "analysis" / "tables" / name
        assert path.read_text() == text, f"{name} is out of date: run python analysis/make_tables.py"


def test_hexagonal_voigt_minus_reuss_on_H1_is_not_rank_one():
    """Contrast: in the hexagonal class H1 enters A2 and E1, which also hold H3 and H2
    copies, so no generalized eigenvalue of (M^R, M^V) is 1 in general."""
    rng = np.random.default_rng(4320)
    bl = {}
    for p, c, _d in hexagonal.IRREPS:
        A = rng.normal(size=(len(c), len(c)))
        bl[p] = A @ A.T + 0.1 * np.eye(len(c))
    V, R = av.voigt(HEX, bl), av.reuss(HEX, bl)
    assert np.linalg.svd(V["H1"] - R["H1"], compute_uv=False).min() > 1e-2
    w = np.linalg.eigvals(np.linalg.solve(V["H1"], R["H1"])).real
    assert w.max() < 1 - 1e-2
