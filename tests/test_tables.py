"""The tables in their final form (analysis/make_tables.py).

- Structure of the algebras: the dimensions of the generated table equal the
  dimensions of the commutant of the point group on V computed by brute-force
  linear algebra, without the embeddings (isotropic: a few random rotations).
- Digits: the worst-case rounding bound of cubic_results is checked by sampling
  the published inputs inside their printed precision; every entry with a
  relative rounding uncertainty above 1 % carries a dagger and no other does;
  the formatter never shows more figures than the rounding supports.
- Provenance (supplement): one row per entry of provenance.csv, every unit
  mapped, and no internal pointer.
"""
import csv
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from gradinv import cubic, hexagonal
from helpers import commutant_dimension, random_rotation

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "analysis" / "tables"


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = _load("make_tables")
cr = _load("cubic_results")
inputs = _load("inputs")


@pytest.fixture(scope="module")
def cubic_res():
    return cr.results()


# ------------------------------------------------------------------ structure
def _totals():
    text = (TABLES / "algebra_structure.tex").read_text()
    return [tuple(map(int, m)) for m in re.findall(r"= 18\$ & & (\d+) & (\d+) \\\\", text)]


@pytest.mark.core
def test_structure_table_matches_brute_force_commutant():
    """The counts are also those of Poorsolhjouy (2026), Table B.2."""
    rng = np.random.default_rng(54)
    groups = {"isotropic": [random_rotation(rng) for _ in range(4)],
              "cubic": cubic.oh_group(), "hexagonal": hexagonal.d6_group()}
    expected = [(commutant_dimension(g), commutant_dimension(g, symmetric=True)) for g in groups.values()]
    assert _totals() == expected == [(6, 5), (15, 11), (32, 22)]


def test_structure_table_is_current():
    assert (TABLES / "algebra_structure.tex").read_text() == mt.table_algebra_structure()


# ------------------------------------------------------------------ digits
def test_rounding_bound_by_sampling():
    """Cu MEAM, the most sensitive set: inputs drawn inside their printed precision.
    m_E is linear in the inputs, so the linear bound must hold; the entries that depend
    on it nonlinearly (1/m_E, the Reuss m2, the Reuss l2) move by more than 1 % as the
    dagger says; a well-conditioned entry stays within its bound."""
    key = "CuMEAM"
    _label, C, Dv, _src = cr.DATASETS[key]
    unc = cr.rounding_uncertainty(key)
    base = cr.tabulated(key)
    rng = np.random.default_rng(5404)
    names = ("moduli/mE", "compliance/mE", "reuss/m2", "lengths/reuss/l2", "moduli/Mhh", "voigt/m3")
    worst = dict.fromkeys(names, 0.0)
    for _ in range(300):
        hC, hD = inputs.half_units(key)
        C2 = [x + rng.uniform(-1, 1) * h for x, h in zip(C, hC)]
        D2 = [x + rng.uniform(-1, 1) * h for x, h in zip(Dv, hD)]
        t = cr.tabulated(key, C2, D2)
        for n in names:
            worst[n] = max(worst[n], abs(t[n] - base[n]))
    for n in ("moduli/mE", "moduli/Mhh", "voigt/m3"):
        assert worst[n] <= 1.001 * unc[n] and worst[n] >= 0.3 * unc[n]
    for n in ("compliance/mE", "reuss/m2", "lengths/reuss/l2"):
        assert worst[n] > mt.FLAG_REL * abs(base[n])
    assert unc["moduli/mE"] / base["moduli/mE"] > 0.02   # about 4 % with the finer digits of Po et al.


def test_daggers_mark_exactly_the_uncertain_entries(cubic_res):
    n_flag = 0
    for k, e in cubic_res["datasets"].items():
        t = cr.tabulated(k)
        n_flag += sum(1 for q, u in e["rounding_uncertainty"].items() if u > mt.FLAG_REL * abs(t[q]))
    text = "".join((TABLES / f).read_text() for f in ("block_moduli.tex", "compliance.tex",
                                                     "averages.tex", "block_lengths.tex"))
    assert text.count("\\dagger") == n_flag == 6


@pytest.mark.parametrize("x, unc, expected", [
    (6.0e-5, 2e-5, "$0.00006^{\\dagger}$"),
    (16667.0, 5.7e3, "${2\\times10^{4}}^{\\dagger}$"),
    (69.654, 11.8, "$70^{\\dagger}$"),
    (0.71366, 1e-6, "0.71366"),
    (-0.014884, 2.0e-4, "$-0.0149^{\\dagger}$"),
])
def test_formatter(x, unc, expected):
    assert mt.num(x, mt.f5 if abs(x) < 1 else mt.f2 if x < 1000 else mt.fc, unc) == expected


def test_no_entry_shows_unsupported_figures(cubic_res):
    """Every tabulated entry, formatted as in the tables, shows at most the figures that
    analysis/digits.py allows for its rounding uncertainty."""
    dg = _load("digits")
    fmts = {"moduli": mt.f5, "compliance": mt.fc, "voigt": mt.f5, "reuss": mt.f5, "hill": mt.f5,
            "condition_number": mt.fcond}
    for k, e in cubic_res["datasets"].items():
        t = cr.tabulated(k)
        for q, u in e["rounding_uncertainty"].items():
            head = q.split("/")[0]
            fmt = mt.f2 if q.endswith("max_over_min") else fmts.get(head, mt.f3)
            s = mt.num(t[q], fmt, u).strip("$").replace("^{\\dagger}", "").strip("{}")
            assert dg.significant_figures(s) <= dg.allowed_figures(t[q], u), (k, q, s)


# ------------------------------------------------------------------ provenance
def test_provenance_table():
    rows = list(csv.DictReader((ROOT / "analysis" / "data" / "provenance.csv").open()))
    text = (TABLES / "provenance.tex").read_text()
    body = [ln for ln in text.splitlines() if re.match(r"P\d\d & ", ln)]
    assert [ln.split(" & ")[0] for ln in body] == [r["id"] for r in rows]
    assert all(len(ln.split(" & ")) == 7 for ln in body)
    assert not re.search(r"\bI\d+\b|\bnotes\b|\bstep\b|gradinv\.", text)
    assert set(r["units"] for r in rows) <= set(mt._UNITS)
    assert text == mt.table_provenance()


def test_text_and_supplement_lists_cover_every_table():
    assert sorted(mt.TEXT_TABLES + mt.SUPPLEMENT_TABLES) == sorted(mt.render_all().keys())
    assert sorted(p.name for p in TABLES.glob("*.tex")) == sorted(list(mt.render_all().keys())
                                                               + list(mt.OTHER_GENERATED))
    assert all((ROOT / g).exists() for g in mt.OTHER_GENERATED.values())
