"""Audit of the figures behind the claims and the application
(analysis/claim_audit.py, analysis/digits.py).

- every macro that depends on one cubic dataset is recomputed by a route separate
  from make_numbers.py and agrees with the macro to half a unit of its last digit;
- no such macro shows more figures than the rounding of the published inputs
  supports, and the macros fixed only roughly by the data are exactly the four of
  Cu MEAM that hang on its nearly vanishing m_E;
- every row of the claim audit points to an existing test and, if it names one,
  to an existing macro.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = dict(re.findall(r"\\newcommand\{\\(Num[A-Za-z]*)\}\{(.*?)\} %",
                          (ROOT / "paper" / "paper_numbers.tex").read_text()))


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("claim_audit")
dg = _load("digits")


def as_float(s):
    m = re.fullmatch(r"(-?[\d.]+)\\times10\^\{(-?\d+)\}", s)
    return float(m.group(1)) * 10 ** int(m.group(2)) if m else float(s)


def half_unit_of(s):
    m = re.fullmatch(r"(-?[\d.]+)\\times10\^\{(-?\d+)\}", s)
    mant, exp = (m.group(1), int(m.group(2))) if m else (s, 0)
    nd = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10 ** (exp - nd)


def test_every_dataset_macro_is_audited():
    vals = ca.values()
    assert set(vals) <= set(NUMBERS)
    assert len(vals) == 3 * 30 + 2 * 5   # Fe and W: N33, N32, det N, radius and ratio


@pytest.mark.core
@pytest.mark.parametrize("name", sorted(ca.values()))
def test_values_by_a_separate_route(name):
    x = ca.values()[name]
    s = NUMBERS[name]
    assert abs(as_float(s) - x) <= half_unit_of(s) * (1 + 1e-9) + 1e-12, (name, s, x)


@pytest.mark.parametrize("name", sorted(ca.values()))
def test_digits_supported_by_the_rounding(name):
    x, u = ca.values()[name], ca.uncertainties()[name]
    assert dg.significant_figures(NUMBERS[name]) <= dg.allowed_figures(x, u)


def test_rough_macros():
    vals, unc = ca.values(), ca.uncertainties()
    rough = sorted(k for k in vals if unc[k] > dg.FLAG_REL * abs(vals[k]))
    assert rough == ["NumCuMEAMCond", "NumCuMEAMRVmtwo", "NumCuMEAMReussmtwo", "NumCuMEAMmE"]
    # m_E of Cu MEAM is fixed to +-4 % by the finer digits of Po et al. (2019), table 1
    # (+-33 % with the five decimals of Lazar et al. 2022); analysis/inputs.py
    assert NUMBERS["NumCuMEAMmERoundingPercent"] == "4"


def test_claim_rows_point_to_existing_checks():
    for claim, _what, macro, check, level in ca.CLAIM_ROWS:
        assert claim in ("1", "2", "3", "app")
        assert set(level.replace(" ", "").split(",")) <= {"V", "CC", "C"}
        path, _, func = check.partition("::")
        text = (ROOT / path).read_text()
        if func:
            assert re.search(rf"^def {func}\(", text, flags=re.M), check
        if macro:
            assert macro in NUMBERS, macro


@pytest.mark.parametrize("x, default, u, expected", [
    (6.0e-5, "0.00006", 2e-5, ("0.00006", True)),
    (28398.5, "28400", 9.7e3, ("3\\times10^{4}", True)),
    (2.49521, "2.4952", 1.6e-3, ("2.495", False)),
    (0.71366, "0.71366", 1e-6, ("0.71366", False)),
    (69.654, "69.654", 11.8, ("70", True)),
    (16667.0, "16667", 5.7e3, ("2\\times10^{4}", True)),
])
def test_digit_rule(x, default, u, expected):
    assert dg.limit(x, default, u) == expected
