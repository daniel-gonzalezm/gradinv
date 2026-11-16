"""The illustrative application: one row per positive definite cubic dataset
(Al MEAM, Cu MEAM, Cu EAM) for Solyaev's (2023) self-consistent scheme with the
full cubic gradient tensor of the grain, in the large-grain limit.

Columns of the application table (analysis/tables/application.tex, written by
analysis/make_tables.py):
  nu_K        Poisson's ratio of the classical Kroner estimate (k fixed to the grain's);
  ell_inf     effective length of the scheme for R/l -> infinity, closed form
              (selfconsistent.large_grain_limit), 7 terms and, in brackets, 8;
  res_inf     the normalized residual rho/|Phi| in the same limit, 7 (8) terms;
  spread      max/min of the block lengths of the Voigt estimate, measured
              against C_K (x) delta with Kroner's C_K (form II; gradinv.lengths);
  ell_LP      Lazar and Po's (2024, eq. 107) length of the same Voigt estimate with
              Kroner's compliance (= dimension-weighted RMS of the block lengths).

Everything is computed live from the package (a second or two); no minimization
is involved. The closed-form limit is checked against the full minimization at
R = 1000 A in tests/test_application.py (slow), and ell_LP against the direct
index contraction of analysis/selfconsistent_fcc.py.

Units: C in eV/A^3, D in eV/A, lengths in A.
Run: python analysis/application.py   (prints the rows)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import concentrator as cc  # noqa: E402
from gradinv import cubic, isotropic  # noqa: E402
from gradinv import lengths as L  # noqa: E402
from gradinv import selfconsistent as sc  # noqa: E402

import inputs  # noqa: E402

# key: (label, C, D_voigt); same inputs and labels as analysis/cubic_results.py
MATERIALS = {k: (inputs.label(k), inputs.C(k), inputs.D(k)) for k in ("Al", "CuMEAM", "CuEAM")}
CUB, ISO = cubic.algebra(), isotropic.algebra()
TERMS = {"7": sc.TERMS_7, "8": sc.TERMS_8}


def grain(key):
    """(C, form II blocks, form I blocks) of the grain."""
    _label, C, Dv = MATERIALS[key]
    blocks = cubic.from_lazar2022(cubic.a_from_voigt(Dv))
    return C, blocks, cc.to_form_I(CUB, blocks, "stiffness")


def row(key):
    label = MATERIALS[key][0]
    C, blocks, _G = grain(key)
    return summary(label, C, blocks)


def summary(label, C, blocks):
    """Row of the table for a cubic grain (C11, C12, C44) with form II gradient blocks."""
    G = cc.to_form_I(CUB, blocks, "stiffness")
    k = (C[0] + 2 * C[1]) / 3
    nuK = sc.kroner_nu(C)
    lamK, muK = sc.lame_from_k_nu(k, nuK)
    lim = {t: sc.large_grain_limit(C, G, T) for t, T in TERMS.items()}
    V = av.voigt(CUB, blocks)
    ref = L.helmholtz_isotropic(lamK, muK)
    lens = L.block_lengths(V, ref)
    flat = np.concatenate([lens[p] for p in ("H3", "H2", "H1")])
    return {"label": label, "C": list(C), "k": k, "nu_kroner": nuK, "lame_kroner": [lamK, muK],
            "limit": lim,
            "voigt_kroner_lengths": {p: list(map(float, v)) for p, v in lens.items()},
            "voigt_kroner_spread": float(flat.max() / flat.min()),
            "voigt_kroner_lazar_po": L.lazar_po_length(ISO, V, ref)}


def results():
    return {"description": __doc__.split("\n\n")[0], "units": "C eV/A^3; D eV/A; lengths A",
            "rows": {key: row(key) for key in MATERIALS}}


if __name__ == "__main__":
    for key, r in results()["rows"].items():
        lim = r["limit"]
        print(f"{r['label']:8s} nu_K={r['nu_kroner']:.4f}  l_inf={lim['7']['ell']:.4f} ({lim['8']['ell']:.4f})  "
              f"res={lim['7']['residual']:.4f} ({lim['8']['residual']:.4f})  "
              f"spread={r['voigt_kroner_spread']:.3f}  l_LP={r['voigt_kroner_lazar_po']:.4f}  "
              f"lengths=" + ", ".join(f"{p}:" + "/".join(f"{x:.3f}" for x in v)
                                      for p, v in r["voigt_kroner_lengths"].items()))
