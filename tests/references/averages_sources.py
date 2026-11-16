"""References for the orientation averages: Lazar, Agiasofitou and Boehlke (2022),
eqs. (158a)-(158e) and Table 6."""
import json
from pathlib import Path

import sympy as sp

_PUBLISHED = json.loads((Path(__file__).resolve().parents[2] / "analysis" / "data"
                         / "published_reference_values.json").read_text())

R = sp.Rational


def lazar2022_eq158(a):
    """Voigt-type averaged constants a1..a5, Lazar et al. (2022), eqs. (158a)-(158e), as printed."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = a
    sym = any(isinstance(x, sp.Basic) for x in a)
    q = (lambda n, d: R(n, d)) if sym else (lambda n, d: n / d)
    return [a1 + q(2, 5) * (a7 + a9 + a10) + q(2, 35) * a11,
            a2 + q(1, 10) * (a8 + 2 * a9) + q(1, 70) * a11,
            a3 + q(2, 5) * (a6 + 2 * a10) + q(2, 35) * a11,
            a4 + q(1, 5) * (2 * a6 + a8) + q(1, 35) * a11,
            a5 + q(2, 5) * (a6 + 2 * a7) + q(2, 35) * a11]


# Lazar et al. (2022), Table 6 (= Eremeyev and Lazar 2022, Table 1), a1..a5 in eV/A
# Single source: analysis/data/published_reference_values.json.
TABLE6 = _PUBLISHED["lazar2022_table6"]["abar"]
