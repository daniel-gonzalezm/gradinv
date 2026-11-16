"""Solyaev's Fe example (analysis/fe_case.py, analysis/results/fe_case.json;
supplement, section S6).

Fast: the stored file was made with the script's current inputs (constants,
sizes, tensors, variants), its closed-form entries equal the live ones, and it
supports the conclusions quoted in the text (coherent reading: l = l0 and
Kroner's nu with a tiny residual for every tensor and size; mixed readings far
from the digitized Fig. 2).
Slow: four cases recomputed and compared with the file.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from gradinv import selfconsistent as sc
from references import solyaev_sources as ss

ROOT = Path(__file__).resolve().parents[1]
STORED = json.loads((ROOT / "analysis" / "results" / "fe_case.json").read_text())


def _load():
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location("fe_case", ROOT / "analysis" / "fe_case.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fe():
    return _load()


def test_stored_file_has_the_current_inputs(fe):
    assert STORED["inputs"]["C"] == list(fe.FE_C) and STORED["anisotropic"]["C"] == list(fe.ANISO_C)
    assert STORED["inputs"]["fig2_sizes"] == fe.FIG2_SIZES == sorted(ss.FIG2_ELL)
    assert sorted(STORED["coherent"]) == sorted(fe.KINDS)
    for k in fe.KINDS:
        assert [r["two_R"] for r in STORED["coherent"][k]] == fe.FIG2_SIZES
    assert [r["two_R"] for r in STORED["anisotropic"]["rows"]] == list(fe.SIZES_SHORT)
    got = sorted((m["kind"], m["grain_literal"], m["medium_literal"], m["s4_printed"]) for m in STORED["mixed"])
    assert got == sorted(fe.MIXED_VARIANTS) and len(got) == 12
    for m in STORED["mixed"]:
        assert [r["two_R"] for r in m["rows"]] == list(fe.SIZES_SHORT)


def test_closed_form_entries_are_live(fe):
    """Zener ratio, shear moduli, Kroner's nu and the large-grain limit (no minimization)."""
    C = fe.FE_C
    assert abs(STORED["inputs"]["zener"] - 2 * C[2] / (C[0] - C[1])) < 1e-14
    assert abs(STORED["inputs"]["mu_voigt"] - (C[0] - C[1] + 3 * C[2]) / 5) < 1e-12
    assert abs(STORED["kroner"]["nu"] - sc.kroner_nu(C)) < 1e-14
    assert STORED["kroner"]["limit_7"] == sc.large_grain_limit(C, sc.helmholtz_cubic_grain(*C))
    assert abs(STORED["anisotropic"]["zener"] - 234 / 95) < 1e-14


def test_mixed_rms_is_recomputed_from_the_rows():
    for m in STORED["mixed"]:
        rms = np.sqrt(np.mean([(r["ell"] - ss.FIG2_ELL[r["two_R"]]) ** 2 for r in m["rows"]]))
        assert abs(rms - m["rms_ell_vs_fig2"]) < 1e-14


def test_stored_results_support_the_quoted_conclusions():
    """Coherent reading -> l = l0 to 2e-3, nu = Kroner's to 2e-3, rho < 5e-4 (all
    tensors and sizes), far from Fig. 2 (l = 1.19-1.93 l0); the large-grain limit agrees;
    a strongly anisotropic grain still gives l within 3 % of l0; no mixed reading
    reproduces Fig. 2 (RMS in l above 0.1)."""
    nuK = STORED["kroner"]["nu"]
    for rows in STORED["coherent"].values():
        for r in rows:
            assert abs(r["ell"] - 1) < 2e-3 and abs(r["nu"] - nuK) < 2e-3 and r["rho"] < 5e-4
            assert ss.FIG2_ELL[r["two_R"]] - r["ell"] > 0.18
    lim = STORED["kroner"]["limit_7"]
    assert abs(lim["ell"] - 1) < 2e-3 and abs(lim["nu"] - nuK) < 1e-12
    assert all(1.005 < r["ell"] < 1.03 for r in STORED["anisotropic"]["rows"])
    assert min(m["rms_ell_vs_fig2"] for m in STORED["mixed"]) > 0.1


@pytest.mark.slow
def test_four_cases_recomputed(fe):
    """Deterministic minimizations: the file equals a fresh run of the script."""
    for kind, twoR in (("exact", 10), ("exact", 100)):
        r = fe.coherent(fe.FE_C, kind, twoR)
        s = [x for x in STORED["coherent"][kind] if x["two_R"] == twoR][0]
        assert all(abs(r[q] - s[q]) < 1e-9 for q in ("ell", "nu", "rho"))
    r = fe.coherent(fe.ANISO_C, "exact", 10)
    s = STORED["anisotropic"]["rows"][0]
    assert all(abs(r[q] - s[q]) < 1e-9 for q in ("ell", "nu", "rho"))
    r = fe.mixed("exact", True, False, False, 10)
    m = [x for x in STORED["mixed"] if (x["kind"], x["grain_literal"], x["s4_printed"]) == ("exact", True, False)][0]
    s = m["rows"][0]
    assert all(abs(r[q] - s[q]) < 1e-9 for q in ("ell", "nu", "rho"))
