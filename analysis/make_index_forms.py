"""Index forms of the basis units for the supplement, section S1.

Writes analysis/tables/index_forms.tex from gradinv.index_forms.latex_index_forms()
(exact sympy expansions; about a minute). tests/test_index_forms.py checks that the
file matches the generator and that every printed expansion rebuilds its unit.

Run: python analysis/make_index_forms.py
"""
from pathlib import Path

from gradinv.index_forms import latex_index_forms

OUT = Path(__file__).resolve().parent / "tables" / "index_forms.tex"


def main():
    OUT.write_text(latex_index_forms())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
