"""The paper figures (analysis/make_figures.py,
analysis/figstyle.py).

Checks: page widths of the PDFs (text width or one-panel width); the shaded zone
of the self-consistent figure (grains with R below the lattice parameter) covers
the same grid points for every material; one color and one marker per material;
and, with the matplotlib version and fonts that wrote them, redrawing from the
stored results gives the committed files byte for byte (so they are current).
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "analysis" / "figures"


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load("figstyle")
WIDTHS = {"cubic_averages": fs.TEXT_WIDTH_IN, "hexagonal_extra_modulus": fs.SINGLE_WIDTH_IN,
          "selfconsistent_fcc": fs.TEXT_WIDTH_IN}


def media_box(pdf):
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
    return float(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("name", list(WIDTHS))
def test_page_width(name):
    w, h = media_box(FIG / f"{name}.pdf")
    assert w == pytest.approx(72 * WIDTHS[name], abs=0.01)
    assert 1.5 * 72 < h < 3.0 * 72
    assert (FIG / f"{name}.png").exists()


def test_material_styles_distinct():
    colors = [c for c, _m in fs.MATERIAL_STYLE.values()]
    markers = [m for _c, m in fs.MATERIAL_STYLE.values()]
    assert len(set(colors)) == len(colors) and len(set(markers)) == len(markers)
    sc = _load("selfconsistent_fcc")
    assert set(sc.MATERIALS) == set(fs.MATERIAL_STYLE)


def test_shaded_zone_same_points_for_every_material():
    sc = _load("selfconsistent_fcc")
    shade = max(sc.LATTICE.values())                  # radius, A
    for a in sc.LATTICE.values():
        below_a = [R for R in sc.R_GRID if R < a]
        shaded = [R for R in sc.R_GRID if R < shade]
        assert below_a == shaded and len(shaded) == 2
    data = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
    assert shade == data["datasets"]["lazarpo2024"]["lattice_a_table2"]["Al"]


def test_fonts_embedded_as_truetype():
    for name in WIDTHS:
        pdf = (FIG / f"{name}.pdf").read_bytes()
        assert b"STIXGeneral" in pdf and b"/FontFile2" in pdf and b"/FontFile3" not in pdf


@pytest.mark.core
def test_redraw_gives_committed_files(tmp_path, monkeypatch):
    import matplotlib
    if matplotlib.__version__ != fs.MATPLOTLIB_VERSION or fs.apply() != fs.FAMILY:
        pytest.skip("byte comparison needs the matplotlib version and fonts of the committed figures")
    mf = _load("make_figures")
    for name, (draw, src) in mf.FIGURES.items():
        mod = sys.modules[draw.__module__]
        monkeypatch.setattr(mod, "OUT_FIG", tmp_path / name)
        draw(json.loads((ROOT / "analysis" / "results" / src).read_text()))
        for ext in (".pdf", ".png"):
            new = (tmp_path / name).with_suffix(ext).read_bytes()
            assert new == (FIG / f"{name}{ext}").read_bytes(), f"{name}{ext} is out of date: run make_figures.py"


# ------------------------------------------------------------------ capital Greek
_FIGURE_SOURCES = {"cubic_averages": "cubic_results", "hexagonal_extra_modulus": "hexagonal_granular",
                   "selfconsistent_fcc": "selfconsistent_fcc"}


@pytest.mark.parametrize("fig", sorted(_FIGURE_SOURCES))
def test_capital_greek_in_figure_labels_is_upright(fig):
    """The class (stix.sty) sets capital Greek upright, so the figure labels write a
    bare \\Delta (upright in matplotlib's mathtext), never \\mathit{\\Delta}."""
    src = (ROOT / "analysis" / f"{_FIGURE_SOURCES[fig]}.py").read_text()
    m = src.find("def figure(")
    body = src[m:]
    italic = re.findall(r"\\(?:mathit|varGamma|varDelta|varTheta|varLambda|varXi|varPi|varSigma|"
                        r"varUpsilon|varPhi|varPsi|varOmega)\b", body)
    assert not italic, italic
