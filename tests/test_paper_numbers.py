"""Every computed figure of the paper is a macro of paper/paper_numbers.tex,
written by analysis/make_numbers.py. This test fails if the file is
out of date, if a macro name is malformed or repeated, if the manuscript uses a
\\Num... macro that is not defined, or if a key figure moves."""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "paper" / "paper_numbers.tex"
DEF_RE = re.compile(r"\\newcommand\{\\(Num[A-Za-z]*)\}\{(.*?)\} %")


def _generator():
    spec = importlib.util.spec_from_file_location("make_numbers", ROOT / "analysis" / "make_numbers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def defined():
    return dict(DEF_RE.findall(NUMBERS.read_text()))


@pytest.mark.core
def test_file_matches_the_generator():
    gen = _generator()
    assert NUMBERS.read_text() == gen.render(gen.numbers()), \
        "paper/paper_numbers.tex is out of date: run python analysis/make_numbers.py"


def test_macro_names_are_letters_only_and_unique():
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", NUMBERS.read_text())
    assert names and all(re.fullmatch(r"Num[A-Za-z]+", n) for n in names)
    assert len(names) == len(set(names))


@pytest.mark.core
def test_manuscript_uses_only_defined_macros(defined):
    used = set()
    for tex in (ROOT / "paper").rglob("*.tex"):
        if tex != NUMBERS:
            used |= set(re.findall(r"\\(Num[A-Za-z]+)", tex.read_text()))
    assert not (used - set(defined)), f"undefined macros: {sorted(used - set(defined))}"


# Key figures of the text; their digits are those the rounding of the inputs supports
# (analysis/claim_audit.py).
KEY = {"NumCuMEAMReussRatio": "2.4960", "NumAlVoigtmthree": "0.73882", "NumCuEAMReussmtwo": "0.12459",
       "NumFeNthreethreeT": "-0.238", "NumFeBonePrinted": "7.29", "NumFeBoneMa": "11.56",
       "NumFeBoneExact": "6.61", "NumFeCoherentEllTen": "1.001", "NumSixthMin": "-0.649",
       "NumSixthMax": "0.410", "NumSixthMaxPercent": "22",
       "NumAlLimEllSeven": "1.388", "NumAlLengthCrystal": "1.177", "NumCuMEAMLengthCrystal": "0.785",
       "NumAlLengthVoigt": "1.170", "NumCuEAMLimResidualSeven": "0.60",
       "NumCuMEAMRVmtwo": "0.0042", "NumAlRVHone": "0.4603", "NumCuMEAMCond": "3.4\\times10^{4}",
       "NumCuEAMVoigtLengthSpread": "3.69", "NumHexIsoReussAtSigma": "0.9130"}


@pytest.mark.parametrize("name", sorted(KEY))
def test_key_figures_do_not_move(defined, name):
    assert defined[name] == KEY[name]
