"""Level (b) of explicitness: the units in index form. The isotropic expansions are
checked explicitly against closed coefficients in the tensors T1..T6 of Monchiet
and Bonnet (2011a).
"""
import numpy as np
import pytest
import sympy as sp

from gradinv import cubic, hexagonal, index_forms, isotropic
from gradinv._tensor import R, to_float
from helpers import is_zero

S5 = sp.sqrt(5)
ISOTROPIC_CLOSED = {  # in T1..T6 of Monchiet and Bonnet (2011a)
    ("H3", "3", "3"): {"T1": -R(1, 15), "T2": R(1, 3), "T3": -R(2, 15), "T4": -R(2, 15), "T5": R(2, 3),
                       "T6": -R(4, 15)},
    ("H2", "2", "2"): {"T1": -R(1, 3), "T2": R(2, 3), "T3": R(1, 3), "T4": R(1, 3), "T5": -R(2, 3),
                       "T6": -R(1, 3)},
    ("H1", "s", "s"): {"T1": R(1, 15), "T3": R(2, 15), "T4": R(2, 15), "T6": R(4, 15)},
    ("H1", "r", "r"): {"T1": R(1, 3), "T3": -R(1, 3), "T4": -R(1, 3), "T6": R(1, 3)},
    ("H1", "s", "r"): {"T1": S5 / 15, "T3": 2 * S5 / 15, "T4": -S5 / 15, "T6": -2 * S5 / 15},
    ("H1", "r", "s"): {"T1": S5 / 15, "T3": -S5 / 15, "T4": 2 * S5 / 15, "T6": -2 * S5 / 15},
}


def _numeric_reconstruction(cls, alg):
    names, tensors = index_forms.spanning_set(cls)
    Tf = np.array([to_float(T).ravel() for T in tensors]).T
    for key, Uk in alg.units().items():
        coef, *_ = np.linalg.lstsq(Tf, Uk.ravel(), rcond=None)
        assert np.allclose(Tf @ coef, Uk.ravel(), atol=1e-12), key


@pytest.mark.exact
def test_isotropic_index_forms_explicit():
    exp = index_forms.expansions("isotropic")
    assert set(exp) == set(ISOTROPIC_CLOSED)
    Ux = isotropic.algebra(exact=True).units()
    for key, terms in ISOTROPIC_CLOSED.items():
        assert set(exp[key]) == set(terms), key
        assert all(sp.simplify(exp[key][n] - c) == 0 for n, c in terms.items()), key
        assert is_zero(index_forms.reconstruct("isotropic", terms) - Ux[key]), key


@pytest.mark.exact
def test_cubic_index_forms():
    names, tensors = index_forms.spanning_set("cubic")
    A = sp.Matrix.hstack(*[T.reshape(729, 1) for T in tensors])
    assert A.rank() == 15                                       # independent spanning set
    exp = index_forms.expansions("cubic")
    Ux = cubic.algebra(exact=True).units()
    assert len(exp) == 15
    for key, terms in exp.items():
        assert is_zero(index_forms.reconstruct("cubic", terms) - Ux[key]), key


@pytest.mark.exact
def test_hexagonal_index_forms():
    exp = index_forms.expansions("hexagonal")
    Ux = hexagonal.algebra(exact=True).units()
    assert len(exp) == 32
    for key, terms in exp.items():
        assert is_zero(index_forms.reconstruct("hexagonal", terms) - Ux[key]), key
    assert exp[("B1", "3", "3")] == {"T1": R(1, 2), "H6": R(1, 8)}
    assert exp[("B2", "3", "3")] == {"T1": R(1, 2), "H6": -R(1, 8)}


@pytest.mark.parametrize("cls", ["isotropic", "cubic", "hexagonal"])
def test_index_forms_span_numeric(cls):
    mod = {"isotropic": isotropic, "cubic": cubic, "hexagonal": hexagonal}[cls]
    _numeric_reconstruction(cls, mod.algebra())


# ------------------------------------------------------------------ LaTeX writer (S1)
import re
from pathlib import Path

INDEX_TEX = Path(__file__).resolve().parents[1] / "analysis" / "tables" / "index_forms.tex"


@pytest.mark.exact
def test_index_forms_file_matches_generator():
    assert INDEX_TEX.read_text() == index_forms.latex_index_forms()


def _parse_coef(tok):
    """'-\\tfrac{2\\sqrt{5}}{15}' -> sympy number (writer format)."""
    sign = -1 if tok.startswith("-") else 1
    tok = tok.lstrip("+-")
    m = re.fullmatch(r"\\tfrac\{(\d*)(?:\\sqrt\{(\d+)\})?\}\{(\d+)\}", tok)
    if m:
        num = int(m.group(1)) if m.group(1) else 1
        root = sp.sqrt(int(m.group(2))) if m.group(2) else 1
        return sign * num * root / int(m.group(3))
    m = re.fullmatch(r"(\d*)(?:\\sqrt\{(\d+)\})?", tok)
    num = int(m.group(1)) if m.group(1) else 1
    root = sp.sqrt(int(m.group(2))) if m.group(2) else 1
    return sign * num * root


_TEX_NAME = re.compile(r"\\mathbb (?:T\^\{\\mathrm\{iso\}\}_\{(\d+)\}|G_\{(\d+)\}(\^\{\\mathsf T\})?)")


def _parsed_blocks(text):
    """{lhs: {name: coef}} of the align* blocks (isotropic and cubic)."""
    out, lhs = {}, None
    for block in re.findall(r"\\begin\{align\*\}(.*?)\\end\{align\*\}", text, re.S):
        for line in block.strip().split("\n"):
            line = line.rstrip("\\").strip()
            if "&=" in line:
                lhs, rest = line.split("&=")
                out[lhs] = {}
            else:
                rest = line.replace(r"&\quad", "")
            pos = 0
            for m in _TEX_NAME.finditer(rest):
                coef = _parse_coef(rest[pos:m.start()].strip())
                if m.group(1):
                    name = f"T{m.group(1)}"
                else:
                    name = f"K{m.group(2)}" + ("T" if m.group(3) else "")
                out[lhs][name] = coef
                pos = m.end()
    return out


@pytest.mark.exact
def test_printed_index_forms_round_trip():
    parsed = _parsed_blocks(INDEX_TEX.read_text())
    n = 0
    for cls in ("isotropic", "cubic"):
        exp = index_forms.expansions(cls)
        for key in index_forms._upper_keys(cls, exp):
            lhs = index_forms._unit_tex(key)
            got = parsed[lhs]
            assert set(got) == set(exp[key]), (cls, key)
            assert all(sp.simplify(got[k] - exp[key][k]) == 0 for k in got), (cls, key)
            n += 1
    assert n == 5 + 11


@pytest.mark.exact
def test_hexagonal_similarity_form_rebuilds_units():
    alg = hexagonal.algebra(exact=True)
    sim = index_forms.hexagonal_from_similarity()
    exp = index_forms.expansions("hexagonal")
    assert set(sim) == set(exp) == set(alg.units())
    for key, terms in sim.items():
        assert set(terms) == set(exp[key]), key
        assert all(sp.simplify(terms[k] - exp[key][k]) == 0 for k in terms), key
        assert is_zero(index_forms.reconstruct("hexagonal", terms) - alg.units()[key]), key
    # the printed matrices and offsets
    P = index_forms.mb2013_similarity()
    assert P["E2"] == sp.Matrix([[2, 2], [-2 * sp.sqrt(2), sp.sqrt(2)]])
    for p, off in (("A2", 3), ("E1", 11), ("E2", 0)):
        g = index_forms.MB2013_GROUPS[p]
        n = len(g)
        assert all(g[c - 1][d - 1] == off + n * c + d for c in range(1, n + 1) for d in range(1, n + 1))


def test_transpose_rule_of_the_printed_expansions():
    names, tensors = index_forms.spanning_set("cubic")
    T = dict(zip(names, (to_float(t) for t in tensors)))
    for k in ("T1", "T2", "T5", "T6", "K6", "K8", "K11"):
        assert np.allclose(T[k], T[k].T), k
    assert np.allclose(T["T3"].T, T["T4"])
    for k in ("K7", "K9", "K10"):
        assert np.allclose(T[k].T, T[k + "T"]) and not np.allclose(T[k], T[k + "T"])


def test_cubic_tensors_are_the_groups_of_lazar2022():
    K = {k: to_float(v) for k, v in index_forms.cubic_tensors().items()}
    groups = {6: K["K6"], 7: K["K7"] + K["K7T"], 8: K["K8"], 9: K["K9"] + K["K9T"],
              10: K["K10"] + K["K10T"], 11: K["K11"]}
    for k, G in groups.items():
        e = np.zeros(11)
        e[k - 1] = 1.0
        assert np.allclose(cubic.lazar2022_tensor(e), G), k
