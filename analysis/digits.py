"""How many digits a computed number may show.

Shared by the tables (make_tables.py) and the macros of the text
(make_numbers.py through claim_audit.py), so both follow one rule.

Rule: with u the rounding uncertainty of a number x (worst-case linear effect of
rounding the published inputs to their printed precision; see
cubic_results.rounding_uncertainty and claim_audit.uncertainties), the last
digit shown may be uncertain by at most five units:

    unit of the last digit >= u/5,   i.e.   figures <= floor(e + 1 - log10(u/5)),

with e = floor(log10 |x|), and at least one figure. A number whose default
format shows more figures is rounded to the allowed ones. A number with a
relative uncertainty above FLAG_REL is flagged, so that the table or the text
says that it is fixed only roughly by the data.
"""
import math
import re

FLAG_REL = 0.01


def significant_figures(s):
    """Significant figures of a formatted number (trailing zeros of an integer do not count)."""
    mant = s.split("\\times")[0]
    digits = re.sub(r"[^0-9]", "", mant).lstrip("0")
    if "." not in mant:
        digits = digits.rstrip("0")
    return max(len(digits), 1)


def allowed_figures(x, u):
    if u <= 0 or x == 0:
        return 99
    e = math.floor(math.log10(abs(x)))
    return max(1, math.floor(e + 1 - math.log10(u / 5)))


def sci(x, sig):
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, sig - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{sig - 1}f}\\times10^{{{e}}}"


def round_sig(x, sig):
    """x with `sig` significant figures: plain decimals below 1e4, scientific from 1e4 on."""
    if abs(x) >= 1e4:
        return sci(x, sig)
    e = int(math.floor(math.log10(abs(x))))
    nd = sig - 1 - e
    if nd > 0:
        return f"{round(x, nd):.{nd}f}"
    return f"{int(round(x, nd)):d}"


def limit(x, default, u):
    """(string, flagged): `default` (the usual format of x) or, if it shows more figures than
    the rounding uncertainty u allows, x rounded to the allowed figures."""
    s = default
    allowed = allowed_figures(x, u)
    if significant_figures(s) > allowed:
        s = round_sig(x, allowed)
    return s, (u > FLAG_REL * abs(x))
