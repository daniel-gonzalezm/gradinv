"""Common style of the paper figures.

Width: Elsevier's figure sizes, 140 mm (1.5 column) for two- and three-panel figures and
90 mm (single column) for one-panel figures, inside the 164.6 mm text block of the CAS
single-column class (cas-sc.cls; class not stored here). Figures are included at their
natural width.

Lettering: STIX, the text and math font that cas-sc.cls loads (stix.sty; the class uses
Charis for the text only where a charis.sty exists, which TeX Live does not ship), so
labels match the text: matplotlib's own STIXGeneral TrueType files for text and its
"stix" mathtext set, 8 pt labels and 7 pt ticks and legends. Capital Greek letters are
upright, as stix.sty sets them in the text. The PDFs embed CID TrueType fonts. Output:
PDF (first submission is a single PDF) and PNG at 300 dpi (final files may be PNG, TIFF,
JPG or EPS).

Categorical colors (one per material, fixed order, never cycled) were checked
with a colorblind-separation and contrast validator on a white surface; each
material also has its own marker, so identity never rests on color alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# matplotlib version that wrote the committed figures (tests/test_figures.py compares bytes only with it)
MATPLOTLIB_VERSION = "3.11.2"
TEXT_WIDTH_IN = 140 / 25.4    # two- and three-panel figures, 140 mm (see module docstring)
SINGLE_WIDTH_IN = 90 / 25.4   # one-panel figures, 90 mm
PT = 1 / 72

# material -> (color, marker); keys are the labels used in the analysis scripts
MATERIAL_STYLE = {
    "Al MEAM": ("#2a78d6", "o"),
    "Cu MEAM": ("#eb6834", "s"),
    "Cu EAM": ("#199e70", "^"),
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SHADE = "#ecebe6"

FAMILY = "STIXGeneral"   # shipped with matplotlib


def apply():
    """Set the rcParams of the paper figures; returns the text font family used."""
    family = FAMILY if FAMILY in {f.name for f in fm.fontManager.ttflist} else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [family, "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })
    return family


def panel_label(ax, letter, x=-0.02, y=1.02):
    """Panel letter '(a)' at the top left, outside the axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=INK)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, out_stem):
    """Write <out_stem>.pdf and <out_stem>.png (300 dpi), without dates, so that
    redrawing the same results gives the same bytes (tests/test_figures.py)."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), metadata={"CreationDate": None, "Creator": None})
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, metadata={"Software": None})
    plt.close(fig)
