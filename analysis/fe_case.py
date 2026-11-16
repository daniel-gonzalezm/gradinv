"""Solyaev's (2023) Fe example, for one sentence of section 7 and the
supplementary material (section S6).

Writes analysis/results/fe_case.json; analysis/make_numbers.py reads it for the
\\NumFe... macros of the supplement and of the neutral sentence. The minimizations
take about two minutes on two cores, which is why the figures go through a file;
tests/test_fe_case.py checks the file against this script (fast: the stored
parameters and conclusions; slow: four cases recomputed).

Contents:
  inputs     the grain constants (published, P12), Zener ratio and the shear
             moduli of the Voigt and Reuss estimates (derived, GPa);
  kroner     Kroner's nu and the closed-form large-grain limit (7 terms);
  coherent   minima of the 7-term error function for the three <S6> tensors
             ("printed", "ma", "exact") at the eleven grain sizes of Solyaev's
             Fig. 2, grain and medium both read as form II Helmholtz models;
  anisotropic the same with the exact tensor for a strongly anisotropic Fe-like
             grain (assumed constants ANISO_C, Zener 2.46), 2R/l0 = 10, 20, 50, 100;
  mixed      the twelve mixed readings (3 tensors x grain or medium read literally
             in form I x <S4> correct or as printed) at 2R/l0 = 10, 20, 50, 100,
             with the RMS difference from the digitized Fig. 2.
Lengths in units of l0 (the grain is l0^2 C (x) delta with l0 = 1).

The mixed readings use the self-contained implementation in
analysis/solyaev_variants.py: the package implements only the coherent reading.

Run: python analysis/fe_case.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from gradinv import lengths as L  # noqa: E402
from gradinv import selfconsistent as sc  # noqa: E402
from references import solyaev_sources as ss  # noqa: E402

FE = json.loads((ROOT / "analysis" / "data" / "published_reference_values.json").read_text())["solyaev2023_fe"]
FE_C = (FE["C11"], FE["C12"], FE["C44"])
FIG2_SIZES = sorted(ss.FIG2_ELL)                  # 2R/l0 of Solyaev's Fig. 2 (published, P14)
KINDS = ("printed", "ma", "exact")
# Assumed (provenance P19): a strongly anisotropic Fe-like grain, Zener ratio 2.46
ANISO_C = (230.0, 135.0, 117.0)
SIZES_SHORT = (10, 20, 50, 100)                   # 2R/l0 for the anisotropic grain and the mixed readings
OUT = ROOT / "analysis" / "results" / "fe_case.json"


def zener(C):
    return 2 * C[2] / (C[0] - C[1])


def coherent(C, kind, twoR):
    """(rho, nu, ell) of the coherent reading (package implementation)."""
    D, nu, ell = sc.solve(C, sc.helmholtz_cubic_grain(*C), twoR / 2, kind)
    return {"two_R": twoR, "rho": D, "nu": nu, "ell": ell}


def mixed(kind, grain_literal, medium_literal, s4_printed, twoR):
    from solyaev_variants import FeCaseVariant, minimize_error
    D, nu, ell = minimize_error(FeCaseVariant(kind, grain_literal, medium_literal, s4_printed, C=FE_C), twoR / 2)
    return {"two_R": twoR, "rho": D, "nu": nu, "ell": ell}


MIXED_VARIANTS = [(k, gl, not gl, s4p) for k in KINDS for gl in (True, False) for s4p in (False, True)]


def _job(args):
    tag, payload = args
    if tag == "coherent":
        kind, twoR = payload
        return tag, payload, coherent(FE_C, kind, twoR)
    if tag == "aniso":
        return tag, payload, coherent(ANISO_C, "exact", payload)
    return tag, payload, mixed(*payload)


def results(processes=2):
    jobs = ([("coherent", (k, s)) for k in KINDS for s in FIG2_SIZES]
            + [("aniso", s) for s in SIZES_SHORT]
            + [("mixed", (*v, s)) for v in MIXED_VARIANTS for s in SIZES_SHORT])
    with Pool(processes) as pool:
        done = pool.map(_job, jobs)
    coh = {k: [] for k in KINDS}
    aniso, mix = [], {v: [] for v in MIXED_VARIANTS}
    for tag, payload, r in done:
        if tag == "coherent":
            coh[payload[0]].append(r)
        elif tag == "aniso":
            aniso.append(r)
        else:
            mix[payload[:4]].append(r)
    mixed_rows = []
    for (k, gl, ml, s4p), rows in mix.items():
        rows.sort(key=lambda r: r["two_R"])
        rms = float(np.sqrt(np.mean([(r["ell"] - ss.FIG2_ELL[r["two_R"]]) ** 2 for r in rows])))
        mixed_rows.append({"kind": k, "grain_literal": gl, "medium_literal": ml, "s4_printed": s4p,
                           "rows": rows, "rms_ell_vs_fig2": rms})
    cl = L.cubic_classical_estimates(*FE_C)
    lim = sc.large_grain_limit(FE_C, sc.helmholtz_cubic_grain(*FE_C))
    return {"description": __doc__.split("\n\n")[0],
            "units": "C and shear moduli in GPa; lengths in units of l0; sizes 2R/l0",
            "inputs": {"C": list(FE_C), "zener": zener(FE_C), "mu_voigt": cl["voigt"][1],
                       "mu_reuss": cl["reuss"][1], "fig2_sizes": FIG2_SIZES},
            "kroner": {"nu": sc.kroner_nu(FE_C), "limit_7": lim},
            "coherent": {k: sorted(v, key=lambda r: r["two_R"]) for k, v in coh.items()},
            "anisotropic": {"C": list(ANISO_C), "zener": zener(ANISO_C),
                            "rows": sorted(aniso, key=lambda r: r["two_R"])},
            "mixed": mixed_rows}


if __name__ == "__main__":
    res = results()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for k, rows in res["coherent"].items():
        e = [r["ell"] for r in rows]
        print(f"coherent {k:7s}: l = {min(e):.4f}-{max(e):.4f}, max rho = {max(r['rho'] for r in rows):.1e}")
    e = [r["ell"] for r in res["anisotropic"]["rows"]]
    print(f"anisotropic (Zener {res['anisotropic']['zener']:.2f}): l = {min(e):.4f}-{max(e):.4f}")
    rms = [m["rms_ell_vs_fig2"] for m in res["mixed"]]
    print(f"mixed readings: RMS vs Fig. 2 = {min(rms):.3f}-{max(rms):.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
