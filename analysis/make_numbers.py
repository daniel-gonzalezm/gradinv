"""Generate paper/paper_numbers.tex: every computed figure quoted in the paper is
a LaTeX macro written here from the package. Do not edit the .tex by
hand; run `python analysis/make_numbers.py` and commit both files.

Macro names: \\Num<Case><Quantity>, letters only. Each macro carries its
provenance (module or file) in a comment. The macros that depend on
one cubic dataset never show more figures than the rounding of the published
inputs supports (analysis/claim_audit.py, analysis/digits.py); a value in
scientific notation (\\times) must be used in math mode. Input data quoted as they
are (published constants) are not generated here.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import concentrator as cc  # noqa: E402
from gradinv import cubic  # noqa: E402
from gradinv import lengths  # noqa: E402
from gradinv import selfconsistent as sc  # noqa: E402
from gradinv.algebra import BlockAlgebra as BA  # noqa: E402

import digits  # noqa: E402
import inputs  # noqa: E402

OUT = ROOT / "paper" / "paper_numbers.tex"
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
VOIGT = {k: inputs.D(k) for k in ("Al", "CuMEAM", "CuEAM", "Fe", "W")}   # analysis/inputs.py
CUB = cubic.algebra()
# Assumed evaluation points (provenance: analysis/data/provenance.csv)
FE_POINT = {"nu": 0.348, "ell": 1.9, "R": 5.0}   # near Solyaev's printed solution at 2R/l0 = 10
FE_TWO_R = ((10, "Ten"), (50, "Fifty"))           # 2R/l0 of Solyaev's Fig. 1 minima
SIXTH_ELL = (1.0, 1.5, 2.0)                       # A; fixed grid for the sixth condition
SIXTH_R = (5.0, 10.0, 20.0)                       # A


def fmt(x, nd):
    return f"{x:.{nd}f}"


def sci_tex(x):
    """One significant figure in math-mode TeX: 1e-4 -> 10^{-4}, 4e-4 -> 4\\times10^{-4}."""
    m, e = f"{x:.0e}".split("e")
    return f"10^{{{int(e)}}}" if m == "1" else f"{m}\\times10^{{{int(e)}}}"


def sci_tex_up(x):
    """Upper bound with one significant figure, rounded up (for text after a "<=" or
    "below"): 1.25e-4 -> 2\\times10^{-4}, 1e-4 -> 10^{-4}, 9.2e-5 -> 10^{-4}."""
    e = math.floor(math.log10(x))
    m = math.ceil(round(x / 10 ** e, 9))
    if m == 10:
        m, e = 1, e + 1
    return f"10^{{{e}}}" if m == 1 else f"{m}\\times10^{{{e}}}"


def power_up(x):
    """The least power of ten not below x, in math-mode TeX: 3e-4 -> 10^{-3}."""
    return f"10^{{{math.ceil(round(math.log10(x), 9))}}}"


def numbers():
    """Ordered list of (name, value string, provenance)."""
    out = []
    blocks = {k: cubic.from_lazar2022(cubic.a_from_voigt(v)) for k, v in VOIGT.items()}
    # Block moduli of the crystal data, eV/A
    for k in ("Al", "CuMEAM", "CuEAM"):
        b = blocks[k]
        out += [(f"Num{k}mE", fmt(b["E"][0, 0], 5), "cubic.from_lazar2022"),
                (f"Num{k}mA", fmt(b["A2"][0, 0], 5), "cubic.from_lazar2022")]
    # Middle eigenvalue of the T1 block of Cu MEAM: the lambda_7 of Lazar et al. (2022),
    # table 5, misprinted there as a repetition of lambda_4
    out.append(("NumCuMEAMLambdaSeven", fmt(float(np.linalg.eigvalsh(blocks["CuMEAM"]["T1"])[1]), 4),
                "cubic.from_lazar2022 (eigenvalues of the T1 block)"))
    # Non-positive T2 block of Fe and W (rejection of Reuss and Hill)
    for k in ("Fe", "W"):
        out.append((f"Num{k}NthreethreeT", fmt(blocks[k]["T2"][0, 0], 3), "cubic.from_lazar2022"))
        # coupling and determinant of the T2 block
        out.append((f"Num{k}NthreetwoT", fmt(blocks[k]["T2"][0, 1], 3), "cubic.from_lazar2022"))
        out.append((f"Num{k}DetNT", fmt(np.linalg.det(blocks[k]["T2"]), 3), "cubic.from_lazar2022"))
    # Neighborhood radius of Admal et al. (2017, eq. 3.11) for Fe and W, and its ratio to the
    # bcc nearest-neighbour distance b = a sqrt(3)/2 with a of Lazar and Po (2024, Table 2)
    # (tests/test_neighborhood.py)
    lattice = DATA["datasets"]["lazarpo2024"]["lattice_a_table2"]
    for k in ("Fe", "W"):
        Rn, _p = lengths.neighborhood_radius(blocks[k], lengths.helmholtz_cubic(*inputs.C(k)))
        out += [(f"Num{k}NeighborhoodRadius", fmt(Rn, 2), "lengths.neighborhood_radius (A)"),
                (f"Num{k}NeighborhoodRatio", fmt(Rn / (lattice[k] * np.sqrt(3) / 2), 2),
                 "lengths.neighborhood_radius / b")]
    # Voigt and Reuss isotropic block moduli, eV/A
    for k in ("Al", "CuMEAM", "CuEAM"):
        for tag, f in (("Voigt", av.voigt), ("Reuss", av.reuss)):
            iso = f(CUB, blocks[k])
            out += [(f"Num{k}{tag}mthree", fmt(iso["H3"][0, 0], 5), f"averages.{f.__name__}"),
                    (f"Num{k}{tag}mtwo", fmt(iso["H2"][0, 0], 6), f"averages.{f.__name__}"),
                    (f"Num{k}{tag}Mss", fmt(iso["H1"][0, 0], 5), f"averages.{f.__name__}"),
                    (f"Num{k}{tag}Msr", fmt(iso["H1"][0, 1], 5), f"averages.{f.__name__}"),
                    (f"Num{k}{tag}Mrr", fmt(iso["H1"][1, 1], 5), f"averages.{f.__name__}")]
        out.append((f"Num{k}ReussRatio", fmt(av.reuss_mE_ratio(blocks[k]), 4 if k == "CuMEAM" else 3),
                    "averages.reuss_mE_ratio"))
    # Solyaev's Fe example
    fe = json.loads((ROOT / "analysis" / "data" / "published_reference_values.json").read_text())["solyaev2023_fe"]
    Cfe = (fe["C11"], fe["C12"], fe["C44"])
    Gfe = sc.helmholtz_cubic_grain(*Cfe)
    kfe = (Cfe[0] + 2 * Cfe[1]) / 3
    for kind, tag in (("printed", "Printed"), ("ma", "Ma"), ("exact", "Exact")):
        inv = sc.invariants(Cfe, Gfe, kfe, FE_POINT["nu"], FE_POINT["ell"], FE_POINT["R"], kind)
        out.append((f"NumFeBone{tag}", fmt(inv["B1"], 2), "selfconsistent.invariants"))
    inv = sc.invariants(Cfe, Gfe, kfe, FE_POINT["nu"], FE_POINT["ell"], FE_POINT["R"], "exact")
    out.append(("NumFeSixthExact", fmt(inv["B2"] - inv["B2'"], 2), "selfconsistent.invariants"))
    for twoR, tag in FE_TWO_R:
        D, nu, ell = sc.solve(Cfe, Gfe, twoR / 2, "exact")
        out += [(f"NumFeCoherentEll{tag}", fmt(ell, 3), "selfconsistent.solve"),
                (f"NumFeCoherentNu{tag}", fmt(nu, 4), "selfconsistent.solve"),
                (f"NumFeCoherentResidual{tag}", sci_tex(D), "selfconsistent.solve")]
    # Weight of the sixth condition with full cubic grains, exact <S6>
    diffs = []
    for name, (C, v) in {"Al": (inputs.C("Al"), VOIGT["Al"]), "CuEAM": (inputs.C("CuEAM"), VOIGT["CuEAM"])}.items():
        C11, C12, C44 = C
        lamV, muV = (C11 + 4 * C12 - 2 * C44) / 5, (C11 - C12 + 3 * C44) / 5
        grain = cc.to_form_I(CUB, cubic.from_lazar2022(cubic.a_from_voigt(v)), "stiffness")
        for ell in SIXTH_ELL:
            for R_ in SIXTH_R:
                G = sc.helmholtz_medium(lamV, muV, ell ** 2)
                P = BA.multiply(sc.averaged_S6("exact", lamV, muV, ell, R_), BA.inverse(G))
                inv = cc.invariants(CUB, cc.concentrator(CUB, grain, G, P))
                diffs.append(inv["B2"] - inv["B2'"])
    diffs = np.array(diffs)
    out += [("NumSixthMin", fmt(diffs.min(), 3), "test_selfconsistent"),
            ("NumSixthMax", fmt(diffs.max(), 3), "test_selfconsistent"),
            ("NumSixthMaxPercent", fmt(100 * np.abs(diffs).max() / 3, 0), "test_selfconsistent")]
    # Full cubic grains: large-grain limit of the scheme and Lazar-Po lengths
    sys.path.insert(0, str(ROOT / "analysis"))
    import selfconsistent_fcc as fcc  # noqa: E402
    for key, tag in (("Al MEAM", "Al"), ("Cu MEAM", "CuMEAM"), ("Cu EAM", "CuEAM")):
        s = fcc.material_summary(key)
        out += [(f"Num{tag}LimEllSeven", fmt(s["limit"]["7"]["ell"], 3), "selfconsistent.large_grain_limit"),
                (f"Num{tag}LimEllEight", fmt(s["limit"]["8"]["ell"], 3), "selfconsistent.large_grain_limit"),
                (f"Num{tag}LimResidualSeven", fmt(s["limit"]["7"]["residual"], 2), "selfconsistent.large_grain_limit"),
                (f"Num{tag}KronerNu", fmt(s["nu_kroner"], 4), "selfconsistent.kroner_nu"),
                (f"Num{tag}LengthCrystal", fmt(s["lazar_po_length"]["crystal"], 3), "Lazar-Po eq. 107"),
                (f"Num{tag}LengthVoigt", fmt(s["lazar_po_length"]["voigt_with_kroner_S"], 3), "Lazar-Po eq. 107"),
                (f"Num{tag}LengthReuss", fmt(s["lazar_po_length"]["reuss_with_kroner_S"], 3), "Lazar-Po eq. 107")]
    # Main results
    import cubic_results as cr  # noqa: E402
    import hexagonal_granular as hx  # noqa: E402
    res = cr.results()["datasets"]
    t6 = max(res[k]["averages"]["voigt"]["table6_max_abs_diff"] for k in ("Al", "W"))
    # printed in math mode as m\times10^{e}
    _m, _e = f"{t6:.0e}".split("e")
    out.append(("NumTableSixMaxDiff", f"{_m}\\times10^{{{int(_e)}}}",
                "cubic_results; Lazar et al. 2022 Table 6"))
    for k in ("Al", "CuMEAM", "CuEAM"):
        e = res[k]
        rv = e["averages"]["reuss_over_voigt"]
        cond = e["crystal"]["condition_number"]
        out += [(f"Num{k}Cond", f"{round(cond, -2):.0f}" if cond > 1000 else fmt(cond, 1), "cubic_results"),
                (f"Num{k}RVmthree", fmt(rv["m3"], 4), "cubic_results (Reuss/Voigt)"),
                (f"Num{k}RVmtwo", fmt(rv["m2"], 4), "cubic_results (Reuss/Voigt)"),
                (f"Num{k}RVHone", fmt(rv["H1"][0], 4), "cubic_results (Reuss/Voigt, H1)"),
                (f"Num{k}VoigtLengthSpread", fmt(e["block_lengths"]["voigt"]["max_over_min"], 2),
                 "lengths.block_lengths"),
                (f"Num{k}HillLengthLP", fmt(e["block_lengths"]["hill"]["lazar_po"], 3),
                 "lengths.lazar_po_length")]
    # Application table: live from analysis/application.py
    import application as ap  # noqa: E402
    arows = ap.results()["rows"]
    for k in ("Al", "CuMEAM", "CuEAM"):
        r = arows[k]
        l7, l8 = r["limit"]["7"]["ell"], r["limit"]["8"]["ell"]
        out += [(f"Num{k}LimResidualEight", fmt(r["limit"]["8"]["residual"], 2),
                 "application.row (selfconsistent.large_grain_limit)"),
                (f"Num{k}VoigtKronerSpread", fmt(r["voigt_kroner_spread"], 2),
                 "application.row (lengths.block_lengths, Kroner C)"),
                (f"Num{k}LimOverLengthPercent", fmt(100 * (l7 / r["voigt_kroner_lazar_po"] - 1), 0),
                 "application.row (l_inf, 7 terms, over l_LP of Voigt with Kroner C)"),
                (f"Num{k}LimShiftPercent", fmt(100 * (l7 - l8) / l7, 1),
                 "application.row (l_inf, 7 to 8 terms)")]
    # Self-consistent estimate against grain size (section 7 and supplement S5).
    # Read from analysis/results/selfconsistent_fcc.json (minimizations;
    # tests/test_application.py checks it against the closed limit).
    fr = json.loads((ROOT / "analysis" / "results" / "selfconsistent_fcc.json").read_text())
    prov = "selfconsistent_fcc.json (analysis/selfconsistent_fcc.py)"
    near, sixth, ncase = 0.0, 0.0, 0
    for name, by_terms in fr["cases"].items():
        for t, rows_ in by_terms.items():
            linf = fr["materials"][name]["limit"][t]["ell"]
            for r_ in rows_:
                ncase += 1
                best = r_["minima"][0]
                if r_["R"] >= 10.0:                       # 2R >= 2 nm
                    near = max(near, abs(best["ell"] - linf) / linf)
                if t == "7":
                    # relative miss of the omitted condition B2' = 3 (not B2 - B2')
                    sixth = max(sixth, abs(best["terms"]["B2'"]))
    # absolute residual of the seven-term minimisers at 2R = 2 nm and 2R = 100 nm
    rho_at = {R: [r_["minima"][0]["rho"] for by_terms in fr["cases"].values()
                  for r_ in by_terms["7"] if r_["R"] == R] for R in (10.0, 500.0)}
    sens = fr["cu_meam_sensitivity"]
    base = [x for x in sens if x["dD16_16"] == 0 and x["dD16_17"] == 0][0]
    dl = max(max(abs(x["ell_inf"] - base["ell_inf"]), abs(x["ell_R5"] - base["ell_R5"])) for x in sens)
    out += [("NumSCCases", str(ncase), prov),
            ("NumSCNearLimitPercent", fmt(100 * near, 0), prov),
            ("NumSCSixthSolutionMaxPercent", fmt(100 * sixth, 0), prov),
            ("NumSCResidualTwoNmMin", digits.round_sig(min(rho_at[10.0]), 2), prov),
            ("NumSCResidualTwoNmMax", digits.round_sig(max(rho_at[10.0]), 2), prov),
            ("NumSCResidualHundredNmMin", digits.sci(min(rho_at[500.0]), 2), prov),
            ("NumSCResidualHundredNmMax", digits.sci(max(rho_at[500.0]), 2), prov),
            ("NumCuMEAMSensitivityEll", power_up(dl), prov)]
    # Moduli induced by the heterogeneity of C (Malyi 2021) against the Voigt estimate of D:
    # grain diameter at which their norms are equal, and their ratio at 2R = 100 nm
    import induced_moduli as im
    prov = "induced_moduli.summary (Malyi 2021, eqs. 10.21-10.22)"
    ind = im.summary()
    out += [(f"Num{k}InducedCrossoverNm", fmt(v["crossover_diameter_nm"], 1), prov) for k, v in ind.items()]
    out += [(f"Num{k}InducedRatioHundredNm", sci_tex(v["ratio_at_diameter_100nm"]), prov) for k, v in ind.items()]
    # Solyaev's Fe example, section 7 and supplement S6.
    # Read from analysis/results/fe_case.json, written by analysis/fe_case.py (minimizations;
    # tests/test_fe_case.py checks the file against the script).
    fc = json.loads((ROOT / "analysis" / "results" / "fe_case.json").read_text())
    prov = "fe_case.json (analysis/fe_case.py)"
    ex = fc["coherent"]["exact"]
    every = [r for rows in fc["coherent"].values() for r in rows]
    an = fc["anisotropic"]
    rms = [m["rms_ell_vs_fig2"] for m in fc["mixed"]]
    out += [("NumFeZener", fmt(fc["inputs"]["zener"], 2), prov),
            ("NumFeMuVoigt", fmt(fc["inputs"]["mu_voigt"], 2), prov),
            ("NumFeMuReuss", fmt(fc["inputs"]["mu_reuss"], 2), prov),
            ("NumFeKronerNu", fmt(fc["kroner"]["nu"], 4), prov),
            ("NumFeLimEll", fmt(fc["kroner"]["limit_7"]["ell"], 4), prov),
            ("NumFeExactEllMin", fmt(min(r["ell"] for r in ex), 4), prov),
            ("NumFeExactEllMax", fmt(max(r["ell"] for r in ex), 4), prov),
            ("NumFeExactNuMin", fmt(min(r["nu"] for r in ex), 4), prov),
            ("NumFeExactNuMax", fmt(max(r["nu"] for r in ex), 4), prov),
            ("NumFeExactResidualMax", sci_tex_up(max(r["rho"] for r in ex)), prov),
            ("NumFeAllEllMin", fmt(min(r["ell"] for r in every), 4), prov),
            ("NumFeAllEllMax", fmt(max(r["ell"] for r in every), 4), prov),
            ("NumFeAllResidualMax", sci_tex_up(max(r["rho"] for r in every)), prov),
            ("NumFeAnisoZener", fmt(an["zener"], 2), prov),
            ("NumFeAnisoEllMin", fmt(min(r["ell"] for r in an["rows"]), 3), prov),
            ("NumFeAnisoEllMax", fmt(max(r["ell"] for r in an["rows"]), 3), prov),
            ("NumFeMixedCount", str(len(fc["mixed"])), prov),
            ("NumFeMixedRmsMin", fmt(min(rms), 2), prov),
            ("NumFeMixedRmsMax", fmt(max(rms), 2), prov)]
    hres = hx.results()["sweeps"]
    for e, tag in (("iso", "Iso"), ("ti", "TI")):
        rows = hres[e]["rows"]
        mid = rows[len(rows) // 2]["reuss"][0]
        one = [r for r in rows if abs(r["delta_over_sigma"] - 1.0) < 1e-9][0]["reuss"][0]
        out.append((f"NumHex{tag}ReussAtSigma", fmt(one / mid, 4), "hexagonal_granular"))
    # granular TI fabric: reachable |Delta|/sigma (density >= 0) and the Reuss change there
    gti = hres["ti"]
    out.append(("NumHexGranDeltaMax", fmt(gti["granular_delta_max_over_sigma"], 2),
                "hexagonal_granular"))
    out.append(("NumHexGranReussChangePercent", fmt(100 * (1 - gti["granular_reuss_m3_ratio_at_max"]), 1),
                "hexagonal_granular"))
    # No macro shows more figures than the rounding of the published inputs supports
    # (analysis/digits.py, analysis/claim_audit.py)
    import claim_audit  # noqa: E402
    out = claim_audit.limit_macros(out)
    # Relative rounding uncertainty of m_E of Cu MEAM, for the caption and the text
    mE_rel = claim_audit.uncertainties()["NumCuMEAMmE"] / claim_audit.values()["NumCuMEAMmE"]
    out.append(("NumCuMEAMmERoundingPercent", fmt(100 * mE_rel, 0), "claim_audit.uncertainties"))
    return out


def render(nums):
    lines = ["% Generated by analysis/make_numbers.py -- do not edit by hand.",
             "% Each macro: value, then its provenance."]
    for name, val, prov in nums:
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}} % {prov}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(numbers()))
    print(f"wrote {OUT.relative_to(ROOT)}")
