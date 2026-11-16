"""Redraw the paper figures from the stored results, without recomputing.

    analysis/figures/cubic_averages.{pdf,png}           (text; from results/cubic_results.json)
    analysis/figures/hexagonal_extra_modulus.{pdf,png}  (text; from results/hexagonal_granular.json)
    analysis/figures/selfconsistent_fcc.{pdf,png}       (supplement; from results/selfconsistent_fcc.json)

The results files are written by the scripts of the same names (selfconsistent_fcc.py
takes several minutes); tests/test_main_results.py, tests/test_application.py and
tests/test_figures.py check that they are current. Style: analysis/figstyle.py.

Run: python analysis/make_figures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import cubic_results  # noqa: E402
import hexagonal_granular  # noqa: E402
import selfconsistent_fcc  # noqa: E402

FIGURES = {
    "cubic_averages": (cubic_results.figure, "cubic_results.json"),
    "hexagonal_extra_modulus": (hexagonal_granular.figure, "hexagonal_granular.json"),
    "selfconsistent_fcc": (selfconsistent_fcc.figure, "selfconsistent_fcc.json"),
}


def main():
    for name, (draw, src) in FIGURES.items():
        draw(json.loads((ROOT / "analysis" / "results" / src).read_text()))
        print(f"wrote analysis/figures/{name}.pdf/.png")


if __name__ == "__main__":
    main()
