"""Audit of the figures behind the main results and the application.

Two parts.

1. Rounding of the inputs. `figures(tag, C, Dv)` recomputes, from the package
   and by a route separate from make_numbers.py, every macro that is a function
   of one cubic dataset (block moduli, Voigt and Reuss moduli and ratios,
   condition number, block lengths, the application row, the neighborhood
   radius). `uncertainties()` gives, for each such macro, the worst-case linear
   effect of rounding the published inputs to their printed precision
   (analysis/inputs.py). make_numbers.py calls `limit_macros` so that no
   macro shows more figures than analysis/digits.py allows;
   tests/test_claim_figures.py checks the values, the digits and the rows below.

2. CLAIM_ROWS: for each result, the figure or exact statement, the macro (if
   any), the independent check (a test of the suite) and the level of
   validation (V verification, CC cross-comparison with published results,
   C consistency; as in analysis/audit_map.py).

Run: python analysis/claim_audit.py   (prints the rounding table)
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.linalg as sl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import cubic, isotropic  # noqa: E402
from gradinv import lengths as L  # noqa: E402

import application as ap  # noqa: E402
import cubic_results as cr  # noqa: E402
import digits  # noqa: E402
import inputs  # noqa: E402

CUB, ISO = cubic.algebra(), isotropic.algebra()
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LATTICE = DATA["datasets"]["lazarpo2024"]["lattice_a_table2"]
TAGS = {"Al": "Al", "CuMEAM": "CuMEAM", "CuEAM": "CuEAM", "Fe": "Fe", "W": "W"}   # macro tag -> cr key
PD = ("Al", "CuMEAM", "CuEAM")


def figures(tag, C, Dv):
    """{macro name: value} of the macros that depend only on one cubic dataset (C, Dv)."""
    b = cubic.from_lazar2022(cubic.a_from_voigt(Dv))
    out = {}
    if tag in ("Fe", "W"):
        Rn, _p = L.neighborhood_radius(b, L.helmholtz_cubic(*C))
        out[f"Num{tag}NthreethreeT"] = b["T2"][0, 0]
        out[f"Num{tag}NthreetwoT"] = b["T2"][0, 1]
        out[f"Num{tag}DetNT"] = np.linalg.det(b["T2"])
        out[f"Num{tag}NeighborhoodRadius"] = Rn
        out[f"Num{tag}NeighborhoodRatio"] = Rn / (LATTICE[tag] * np.sqrt(3) / 2)
        return out
    out[f"Num{tag}mE"] = b["E"][0, 0]
    out[f"Num{tag}mA"] = b["A2"][0, 0]
    V, R = av.voigt(CUB, b), av.reuss(CUB, b)
    for est, X in (("Voigt", V), ("Reuss", R)):
        out[f"Num{tag}{est}mthree"] = X["H3"][0, 0]
        out[f"Num{tag}{est}mtwo"] = X["H2"][0, 0]
        out[f"Num{tag}{est}Mss"] = X["H1"][0, 0]
        out[f"Num{tag}{est}Msr"] = X["H1"][0, 1]
        out[f"Num{tag}{est}Mrr"] = X["H1"][1, 1]
    out[f"Num{tag}ReussRatio"] = R["H2"][0, 0] / b["E"][0, 0]
    spec = np.linalg.eigvalsh(cr._basis_V().T @ CUB.assemble(b) @ cr._basis_V())
    out[f"Num{tag}Cond"] = spec.max() / np.abs(spec).min()
    out[f"Num{tag}RVmthree"] = R["H3"][0, 0] / V["H3"][0, 0]
    out[f"Num{tag}RVmtwo"] = R["H2"][0, 0] / V["H2"][0, 0]
    out[f"Num{tag}RVHone"] = sl.eigh(R["H1"], V["H1"], eigvals_only=True)[0]
    cl = L.cubic_classical_estimates(*C)
    for est, X, key in (("Voigt", V, "voigt"), ("Hill", av.hill(CUB, b), "hill")):
        r = L.helmholtz_isotropic(*cl[key])
        lens = L.block_lengths(X, r)
        flat = np.concatenate([lens[p] for p in ("H3", "H2", "H1")])
        if est == "Voigt":
            out[f"Num{tag}VoigtLengthSpread"] = flat.max() / flat.min()
        else:
            out[f"Num{tag}HillLengthLP"] = L.lazar_po_length(ISO, X, r)
    s = ap.summary(tag, C, b)
    l7, l8 = s["limit"]["7"]["ell"], s["limit"]["8"]["ell"]
    lamK, muK = s["lame_kroner"]
    out.update({
        f"Num{tag}LimEllSeven": l7, f"Num{tag}LimEllEight": l8,
        f"Num{tag}LimResidualSeven": s["limit"]["7"]["residual"],
        f"Num{tag}LimResidualEight": s["limit"]["8"]["residual"],
        f"Num{tag}KronerNu": s["nu_kroner"],
        f"Num{tag}LengthCrystal": L.lazar_po_length(CUB, b, L.helmholtz_cubic(*C)),
        f"Num{tag}LengthVoigt": s["voigt_kroner_lazar_po"],
        f"Num{tag}LengthReuss": L.lazar_po_length(ISO, R, L.helmholtz_isotropic(lamK, muK)),
        f"Num{tag}VoigtKronerSpread": s["voigt_kroner_spread"],
        f"Num{tag}LimOverLengthPercent": 100 * (l7 / s["voigt_kroner_lazar_po"] - 1),
        f"Num{tag}LimShiftPercent": 100 * (l7 - l8) / l7,
    })
    return out


def _inputs(tag):
    _label, C, Dv, _src = cr.DATASETS[TAGS[tag]]
    return list(C), list(Dv)


@lru_cache(maxsize=None)
def values():
    return {k: v for tag in TAGS for k, v in figures(tag, *_inputs(tag)).items()}


@lru_cache(maxsize=None)
def uncertainties():
    """{macro: worst-case linear effect of the rounding of the 14 inputs of its dataset}."""
    out = {}
    for tag in TAGS:
        C, Dv = _inputs(tag)
        hC, hD = inputs.half_units(TAGS[tag])
        base = figures(tag, C, Dv)
        unc = dict.fromkeys(base, 0.0)
        for kind in ("C", "D"):
            vals = C if kind == "C" else Dv
            for i, x in enumerate(vals):
                pert = list(vals)
                pert[i] = x + (hC if kind == "C" else hD)[i]
                new = figures(tag, pert if kind == "C" else C, pert if kind == "D" else Dv)
                for k in base:
                    unc[k] += abs(new[k] - base[k])
        out.update(unc)
    return out


def limit_macros(nums):
    """Apply analysis/digits.py to the audited macros of make_numbers.numbers().
    nums: list of (name, value string, provenance); returns the new list."""
    vals, unc = values(), uncertainties()
    out = []
    for name, s, prov in nums:
        if name in vals:
            new, flagged = digits.limit(vals[name], s, unc[name])
            if new != s or flagged:
                prov += f"; rounding of the inputs +-{unc[name]:.1g}" + (" (rough)" if flagged else "")
            s = new
        out.append((name, s, prov))
    return out


# ------------------------------------------------------------------ claim rows
# (result, figure or statement, macro or None, independent check, level)
CLAIM_ROWS = [
    ("1", "algebra dimensions 6/15/32 and symmetric 5/11/22", None,
     "tests/test_tables.py::test_structure_table_matches_brute_force_commutant", "V"),
    ("1", "same counts as Poorsolhjouy (2026), Table B.2", None,
     "tests/test_tables.py::test_structure_table_matches_brute_force_commutant", "CC"),
    ("1", "multiplication tables (225 and 1024 products)", None,
     "tests/test_hexagonal.py::test_multiplication_table_all_1024_products", "V"),
    ("1", "extra hexagonal modulus = 4 eta of Auffray et al. (2013)", None,
     "tests/test_hexagonal.py::test_auffray2013_exact", "CC"),
    ("1", "extra modulus = m=+-3 splitting of H3", None,
     "tests/test_hexagonal_extra_modulus.py::test_extra_direction_is_the_m3_splitting_of_H3", "V"),
    ("1", "inverse of odd elements closes in the class", None,
     "tests/test_odd_inverse.py", "V"),
    ("2", "block inverse = 18 x 18 inverse on the data", None,
     "tests/test_cubic.py::test_data_blocks_inverse_and_positive_definiteness", "V"),
    ("2", "condition number of Cu MEAM (set by m_E)", "NumCuMEAMCond",
     "tests/test_claim_figures.py::test_values_by_a_separate_route", "V"),
    ("2", "condition numbers of Al and Cu EAM", "NumAlCond",
     "tests/test_claim_figures.py::test_values_by_a_separate_route", "V"),
    ("2", "m_E of Cu MEAM (one significant figure)", "NumCuMEAMmE",
     "tests/test_claim_figures.py::test_values_by_a_separate_route", "V"),
    ("3", "Voigt reproduces Lazar et al. (2022), Table 6", "NumTableSixMaxDiff",
     "tests/test_averages.py::test_voigt_reproduces_lazar2022_eq158_exact", "CC"),
    ("3", "weighted-mean rule = Frobenius projection", None,
     "tests/test_averages.py::test_weighted_rule_is_the_frobenius_projection", "V"),
    ("3", "Reuss by blocks = Monte Carlo average of rotated compliances", None,
     "tests/test_averages.py::test_monte_carlo_average_of_rotated_compliances", "V"),
    ("3", "D_R <= D_V block by block (proof of section 6(b), identity in exact arithmetic)", None,
     "tests/test_averages.py::test_reuss_below_voigt_proof_identity_exact", "V"),
    ("3", "M^V - M^R rank one (cubic), not rank one (hexagonal)", None,
     "tests/test_main_results.py::test_cubic_voigt_minus_reuss_on_H1_is_rank_one", "V"),
    ("3", "Reuss/Voigt ratios per piece", "NumAlRVmthree",
     "tests/test_claim_figures.py::test_values_by_a_separate_route", "V"),
    ("3", "factor m_2^R/m_E of Cu MEAM (limit 5/2)", "NumCuMEAMReussRatio",
     "tests/test_averages.py::test_cu_meam_factor", "V"),
    ("3", "Voigt independent of the extra modulus, Reuss even in it", "NumHexIsoReussAtSigma",
     "tests/test_main_results.py::test_hexagonal_isotropic_base_closed_value", "V"),
    ("3", "Lazar-Po length = weighted RMS of block lengths", "NumAlLengthCrystal",
     "tests/test_main_results.py::test_lazar_po_length_is_their_eq107_and_the_published_value", "V, CC"),
    ("3", "Fe and W outside Reuss; neighborhood radius below b", "NumFeNeighborhoodRatio",
     "tests/test_neighborhood.py::test_radius_by_brute_force", "V"),
    ("app", "large-grain limit = full minimization at R = 1000 A", "NumAlLimEllSeven",
     "tests/test_application.py", "V"),
    ("app", "excess of l_inf over l_LP of Voigt", "NumCuEAMLimOverLengthPercent",
     "tests/test_claim_figures.py::test_values_by_a_separate_route", "V"),
]


if __name__ == "__main__":
    vals, unc = values(), uncertainties()
    for k in sorted(vals):
        rel = unc[k] / abs(vals[k]) if vals[k] else 0.0
        flag = "  ROUGH" if rel > digits.FLAG_REL else ""
        print(f"{k:34s} {vals[k]: .6g}  +-{unc[k]:.2g}  ({100 * rel:.3f} %){flag}")
