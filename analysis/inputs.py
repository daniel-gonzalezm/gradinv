"""Inputs of the cubic computations, with the rounding unit of each published number.

For each dataset: the elastic constants (C11, C12, C44) and the eleven gradient
constants D in the Voigt order of Lazar et al. (2022), Table 1, together with half a
unit of the last printed digit of each number (the rounding uncertainty used by
analysis/digits.py through cubic_results.rounding_uncertainty and
claim_audit.uncertainties).

Al and Cu (2NN MEAM) are printed twice: by Lazar et al. (2022), Table 1, with five
decimals, and by Po et al. (2019), Table 1, with five significant figures. The two
printings agree to rounding (tests/test_source_data.py). Entry by entry, the input is
the printing with the finer last digit: five significant figures carry more digits than
five decimals for the entries below 0.1 in magnitude (D2,5, D3,5, D16,16 and D16,17 of
Cu; D2,5 of Al), fewer for those above 1 (C11 of Cu, D1,1 of Al). Where the two
digits are equal the value of Lazar et al. is kept. Fe and W are printed only by
Lazar et al., Cu EAM only by Po et al.

Source of every number: analysis/data/cubic_gradient_constants.json (transcriptions)
and analysis/data/provenance.csv.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LZ, PO = DATA["datasets"]["lazar2022"], DATA["datasets"]["po2019"]


def half_unit(x, rule):
    """Half a unit of the last printed digit: five decimals or five significant figures."""
    if rule == "decimals5":
        return 0.5e-5
    if rule == "significant5":
        return 0.5 * 10.0 ** (math.floor(math.log10(abs(x))) - 4)
    raise ValueError(rule)


def _printing(values, rule, source):
    return [(float(x), half_unit(x, rule), source) for x in values]


def _finer(first, second):
    """Entry by entry, the printing with the smaller rounding unit (`first` on ties)."""
    out = []
    for a, b in zip(first, second, strict=True):
        if abs(a[0] - b[0]) > a[1] + b[1] + 1e-12:
            raise ValueError(f"the two printings disagree beyond rounding: {a} {b}")
        out.append(b if b[1] < a[1] else a)
    return out


LAZAR_TABLE1 = "Lazar et al. 2022, Table 1"
PO_TABLE1 = "Po et al. 2019, Table 1"


def _lazar(mat):
    return (_printing(LZ["C"][mat], "decimals5", LAZAR_TABLE1),
            _printing(LZ["D_voigt"][mat], "decimals5", LAZAR_TABLE1))


def _po(mat):
    return (_printing(PO["C"][mat], "significant5", PO_TABLE1),
            _printing(PO["D_voigt"][mat], "significant5", PO_TABLE1))


def _both(lz_mat, po_mat):
    (lc, ld), (pc, pd) = _lazar(lz_mat), _po(po_mat)
    return _finer(lc, pc), _finer(ld, pd)


# key: (label, [(value, half unit, source)] for C, same for D)
ENTRIES = {
    "Al": ("Al MEAM", *_both("Al", "Al_MEAM")),
    "CuMEAM": ("Cu MEAM", *_both("Cu", "Cu_MEAM")),
    "CuEAM": ("Cu EAM", *_po("Cu_EAM")),
    "Fe": ("Fe MEAM", *_lazar("Fe")),
    "W": ("W MEAM", *_lazar("W")),
}


def label(key):
    return ENTRIES[key][0]


def C(key):
    return [v for v, _h, _s in ENTRIES[key][1]]


def D(key):
    return [v for v, _h, _s in ENTRIES[key][2]]


def half_units(key):
    """(half units of C, half units of D) of the inputs of `key`."""
    return [h for _v, h, _s in ENTRIES[key][1]], [h for _v, h, _s in ENTRIES[key][2]]


def sources(key):
    """The printings the inputs of `key` are taken from, in order of first use."""
    out = []
    for _v, _h, s in ENTRIES[key][1] + ENTRIES[key][2]:
        if s not in out:
            out.append(s)
    return out


if __name__ == "__main__":
    for k in ENTRIES:
        hc, hd = half_units(k)
        print(k, label(k), sources(k))
        print("   C", C(k), hc)
        print("   D", D(k), hd)
