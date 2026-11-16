"""The application table (analysis/application.py).

Fast: limit case of an isotropic Helmholtz grain; the block lengths of the table
by an independent 18 x 18 generalized eigenproblem; the Lazar-Po length by the
direct index contraction of eq. (107); the table against the macros; the closed
large-grain limit against the stored minimizations at R = 500 A.
Slow: the closed limit against the full minimization at R = 1000 A for the three
rows and with 7 and 8 terms.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sl

from gradinv import averages as av
from gradinv import concentrator as cc
from gradinv import isotropic
from gradinv import lengths as L
from gradinv import selfconsistent as sc
from helpers import basis_V

ROOT = Path(__file__).resolve().parents[1]
ISO = isotropic.algebra()
NUMBERS = dict(re.findall(r"\\newcommand\{\\(Num[A-Za-z]*)\}\{([^}]*)\}",
                          (ROOT / "paper" / "paper_numbers.tex").read_text()))
FCC_NAMES = {"Al": "Al MEAM", "CuMEAM": "Cu MEAM", "CuEAM": "Cu EAM"}


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def app():
    return _load("application")


@pytest.fixture(scope="module")
def rows(app):
    return app.results()["rows"]


def test_isotropic_helmholtz_grain_gives_the_trivial_row(app):
    """Limit case: grain l0^2 C (x) delta with isotropic C gives nu_K = the grain's,
    l_inf = l0 and zero residual (7 and 8 terms), every Voigt block length = l0
    (spread 1) and l_LP = l0."""
    C11, C12, l0 = 1.0, 0.4, 1.3
    C = (C11, C12, (C11 - C12) / 2)
    r = app.summary("iso", C, L.helmholtz_cubic(*C, l2=l0 ** 2))
    assert abs(r["nu_kroner"] - C12 / (C11 + C12)) < 1e-10
    for t in ("7", "8"):
        assert abs(r["limit"][t]["ell"] - l0) < 1e-10 and r["limit"][t]["residual"] < 1e-10
    assert abs(r["voigt_kroner_spread"] - 1) < 1e-10 and abs(r["voigt_kroner_lazar_po"] - l0) < 1e-10


@pytest.mark.parametrize("key", ["Al", "CuMEAM", "CuEAM"])
def test_block_lengths_by_the_18x18_generalized_eigenproblem(app, rows, key):
    """Independent route: the 18 generalized eigenvalues of (V, C_K (x) delta) restricted
    to V are the squared block lengths with multiplicities 7, 5, 3, 3."""
    r = rows[key]
    _C, blocks, _G = app.grain(key)
    Bv = basis_V()
    V18 = Bv.T @ ISO.assemble(av.voigt(app.CUB, blocks)) @ Bv
    G18 = Bv.T @ ISO.assemble(L.helmholtz_isotropic(*r["lame_kroner"])) @ Bv
    ell = np.sqrt(sl.eigh(0.5 * (V18 + V18.T), 0.5 * (G18 + G18.T), eigvals_only=True))
    lens = r["voigt_kroner_lengths"]
    expected = np.sort(np.r_[[lens["H3"][0]] * 7, [lens["H2"][0]] * 5, [lens["H1"][0]] * 3, [lens["H1"][1]] * 3])
    assert np.abs(ell - expected).max() < 1e-12
    assert abs(ell.max() / ell.min() - r["voigt_kroner_spread"]) < 1e-12
    assert abs(np.sqrt(np.mean(ell ** 2)) - r["voigt_kroner_lazar_po"]) < 1e-12


@pytest.mark.parametrize("key", ["Al", "CuMEAM", "CuEAM"])
def test_lazar_po_length_by_the_index_contraction(app, rows, key):
    """l_LP of the table (block trace) = S_ijkl D_ijmklm / 18 with Kroner's S, contracted
    on the 27 x 27 storage (analysis/selfconsistent_fcc.py), and = the stored figure."""
    fcc = _load("selfconsistent_fcc")
    r = rows[key]
    _C, blocks, _G = app.grain(key)
    lam, mu = r["lame_kroner"]
    d = np.eye(3)
    CK = lam * np.einsum("ij,kl->ijkl", d, d) + mu * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
    direct = fcc.lazar_po_length(ISO.assemble(av.voigt(app.CUB, blocks)), CK)
    assert abs(direct - r["voigt_kroner_lazar_po"]) < 1e-12
    s = fcc.material_summary(FCC_NAMES[key])
    assert abs(s["lazar_po_length"]["voigt_with_kroner_S"] - r["voigt_kroner_lazar_po"]) < 1e-12
    assert abs(s["nu_kroner"] - r["nu_kroner"]) < 1e-14
    for t in ("7", "8"):
        assert s["limit"][t] == r["limit"][t]


@pytest.mark.parametrize("key", ["Al", "CuMEAM", "CuEAM"])
def test_table_cells_equal_the_macros(rows, key):
    """The table and the text quote the same rounded figures."""
    r = rows[key]
    lim = r["limit"]
    assert NUMBERS[f"Num{key}KronerNu"] == f"{r['nu_kroner']:.4f}"
    assert NUMBERS[f"Num{key}LimEllSeven"] == f"{lim['7']['ell']:.3f}"
    assert NUMBERS[f"Num{key}LimEllEight"] == f"{lim['8']['ell']:.3f}"
    assert NUMBERS[f"Num{key}LimResidualSeven"] == f"{lim['7']['residual']:.2f}"
    assert NUMBERS[f"Num{key}LimResidualEight"] == f"{lim['8']['residual']:.2f}"
    assert NUMBERS[f"Num{key}VoigtKronerSpread"] == f"{r['voigt_kroner_spread']:.2f}"
    assert NUMBERS[f"Num{key}LengthVoigt"] == f"{r['voigt_kroner_lazar_po']:.3f}"
    row = [ln for ln in (ROOT / "analysis" / "tables" / "application.tex").read_text().splitlines()
           if ln.startswith(r["label"] + " &")]
    assert len(row) == 1
    cells = [c.strip() for c in row[0].rstrip("\\ ").split("&")]
    assert cells[1:] == [NUMBERS[f"Num{key}KronerNu"],
                         f"{NUMBERS[f'Num{key}LimEllSeven']} ({NUMBERS[f'Num{key}LimEllEight']})",
                         f"{NUMBERS[f'Num{key}LimResidualSeven']} ({NUMBERS[f'Num{key}LimResidualEight']})",
                         NUMBERS[f"Num{key}VoigtKronerSpread"], NUMBERS[f"Num{key}LengthVoigt"]]


KEY_CELLS = {"Al": ("0.3356", "1.388", "0.26", "2.91", "1.170"),
             "CuMEAM": ("0.3360", "1.082", "0.40", "5.10", "0.753"),
             "CuEAM": ("0.3392", "1.492", "0.60", "3.87", "0.907")}


@pytest.mark.parametrize("key", sorted(KEY_CELLS))
def test_key_cells_do_not_move(key):
    names = ("KronerNu", "LimEllSeven", "LimResidualSeven", "VoigtKronerSpread", "LengthVoigt")
    assert tuple(NUMBERS[f"Num{key}{n}"] for n in names) == KEY_CELLS[key]


@pytest.mark.parametrize("key", ["Al", "CuMEAM", "CuEAM"])
@pytest.mark.parametrize("terms", ["7", "8"])
def test_closed_limit_against_the_stored_minimization_at_500(rows, key, terms):
    """The stored minimization at R = 500 A (analysis/results/selfconsistent_fcc.json)
    is already within 5e-4 of the closed limit in l, 2e-4 in nu, 1e-3 in rho/|Phi|."""
    stored = json.loads((ROOT / "analysis" / "results" / "selfconsistent_fcc.json").read_text())
    best = [c for c in stored["cases"][FCC_NAMES[key]][terms] if c["R"] == 500.0][0]["minima"][0]
    lim = rows[key]["limit"][terms]
    assert abs(best["ell"] - lim["ell"]) / lim["ell"] < 5e-4
    assert abs(best["nu"] - rows[key]["nu_kroner"]) < 2e-4
    assert abs(best["rho_over_Phi"] - lim["residual"]) < 1e-3


@pytest.mark.core
@pytest.mark.slow
@pytest.mark.parametrize("key", ["Al", "CuMEAM", "CuEAM"])
@pytest.mark.parametrize("terms", ["7", "8"])
def test_closed_limit_against_the_full_minimization_at_1000(app, rows, key, terms):
    """Independent check of the closed form: the full minimization of the
    scheme at R = 1000 A gives l and nu to 2e-4 and rho/|Phi| to 1e-3."""
    C, _blocks, G = app.grain(key)
    T = sc.TERMS_7 if terms == "7" else sc.TERMS_8
    D, nu, ell = sc.solve(C, G, 1000.0, terms=T)
    lim = rows[key]["limit"][terms]
    assert abs(ell - lim["ell"]) / lim["ell"] < 2e-4 and abs(nu - rows[key]["nu_kroner"]) < 2e-4
    assert abs(D / abs(cc.solyaev_phi(1000.0 / ell)) - lim["residual"]) < 1e-3


def test_stored_minimizations_support_supplement_S5():
    """Supplement S5: every one of the 72 stored cases has a single minimum inside
    the grid; nu and l tend to the closed limit; |B2 - B2'| of the seven-term solutions
    decreases with R; rho/|Phi| changes by less than 10 % across R."""
    stored = json.loads((ROOT / "analysis" / "results" / "selfconsistent_fcc.json").read_text())
    n = 0
    for name, by_terms in stored["cases"].items():
        for t, rows in by_terms.items():
            for r in rows:
                n += 1
                assert len(r["minima"]) == 1 and not r["minima"][0]["at_grid_edge"]
            lim = stored["materials"][name]["limit"][t]
            big = [r["minima"][0] for r in rows if r["R"] >= 10.0]
            assert all(abs(b["ell"] - lim["ell"]) / lim["ell"] < 0.02 for b in big)
            rho = [r["minima"][0]["rho_over_Phi"] for r in rows]
            assert max(rho) / min(rho) < 1.1
            if t == "7":
                gap = [abs(r["minima"][0]["B2_minus_B2p"]) for r in rows]
                assert all(a > b for a, b in zip(gap, gap[1:]))
    assert n == 72
