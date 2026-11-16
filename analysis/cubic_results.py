"""Main results for the cubic data.

For the five published datasets (Al, Cu MEAM: Lazar et al. 2022, Table 1, by
their eq. 60; Cu EAM: Po et al. 2019, Table 1; Fe and W: Lazar et al. 2022,
stiffness and Voigt only):

  - constants a1..a11 and the 11 block moduli (T1 = M, T2 = N, m_E, m_A);
  - eigenvalues of each block, positive definiteness, condition number on V;
  - gradient compliance S6 = D^-1 in blocks and in the a-constants of Lazar et al.
    (formed for the positive definite sets), checked against the 18 x 18 inverse;
  - Voigt, Reuss and Hill isotropic block moduli and their a1..a5;
    Voigt checked against Lazar et al. (2022), Table 6 (Al and W);
  - Reuss/Voigt ratios per isotropic block (m3, m2 and the generalized
    eigenvalues of (M^R, M^V); one of the latter is exactly 1 in the cubic class,
    because M^V - M^R = m_h m_h^T / M_hh has rank one);
  - block lengths of the crystal (relative to C (x) delta) and of the three
    estimates (Voigt with Voigt C, Reuss with Reuss C, Hill with Hill C), and the
    Lazar-Po length (their dimension-weighted RMS).

Units: C in eV/A^3, D in eV/A, compliance in A/eV, lengths in A.
Run: python analysis/cubic_results.py  (a few seconds). Writes
analysis/results/cubic_results.json and the draft figure
analysis/figures/cubic_averages.{pdf,png}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import cubic, isotropic  # noqa: E402
from gradinv import lengths as L  # noqa: E402
from gradinv.algebra import BlockAlgebra as BA  # noqa: E402

import inputs  # noqa: E402

# key: (label, C, D_voigt, source); inputs and their rounding units in analysis/inputs.py
DATASETS = {k: (inputs.label(k), inputs.C(k), inputs.D(k), "; ".join(inputs.sources(k)))
            for k in ("Al", "CuMEAM", "CuEAM", "Fe", "W")}
PD_SETS = ("Al", "CuMEAM", "CuEAM")
# Lazar et al. (2022), Table 6, a1..a5 (eV/A), as printed; single source in
# analysis/data/published_reference_values.json (provenance: analysis/data/provenance.csv)
PUBLISHED = json.loads((ROOT / "analysis" / "data" / "published_reference_values.json").read_text())
TABLE6 = PUBLISHED["lazar2022_table6"]["abar"]
CUB, ISO = cubic.algebra(), isotropic.algebra()
OUT = ROOT / "analysis" / "results" / "cubic_results.json"
MODULI = [n for n, *_ in cubic.MODULI]
ISO_MODULI = [n for n, *_ in isotropic.MODULI]


def _basis_V():
    from gradinv.skeleton import identity_V
    w, U = np.linalg.eigh(identity_V())
    return U[:, w > 0.5]


def crystal(key, C=None, Dv=None):
    """Block moduli, eigenvalues and compliance of one dataset (C, Dv override the data)."""
    label, C0, Dv0, src = DATASETS[key]
    C = C0 if C is None else C
    Dv = Dv0 if Dv is None else Dv
    a = cubic.a_from_voigt(Dv)
    b = cubic.from_lazar2022(a)
    eig = {p: np.linalg.eigvalsh(B) for p, B in b.items()}
    spec = CUB.spectrum(b)
    out = {"label": label, "source": src, "C": list(C), "a": list(map(float, a)),
           "moduli": dict(zip(MODULI, map(float, cubic.blocks_to_vector(b)))),
           "block_eigenvalues": {p: list(map(float, v)) for p, v in eig.items()},
           "non_positive_blocks": BA.non_positive_blocks(b),
           "spectrum_min": float(spec.min()), "spectrum_max": float(spec.max()),
           "condition_number": float(spec.max() / np.abs(spec).min())}
    S = BA.inverse(b)
    # check: block inverse = 18 x 18 inverse
    Bv = _basis_V()
    S18 = np.linalg.inv(Bv.T @ CUB.assemble(b) @ Bv)
    out["compliance_check_18x18"] = float(np.abs(Bv.T @ CUB.assemble(S) @ Bv - S18).max() / np.abs(S18).max())
    if key in PD_SETS:
        out["compliance_moduli"] = dict(zip(MODULI, map(float, cubic.blocks_to_vector(S))))
        out["compliance_a"] = list(map(float, cubic.to_lazar2022(S)))
    return a, b, out


def estimates(key, b):
    """Voigt (all sets), Reuss and Hill (positive definite sets)."""
    label, C, _Dv, _src = DATASETS[key]
    est = {"voigt": av.voigt(CUB, b)}
    if key in PD_SETS:
        est["reuss"] = av.reuss(CUB, b)
        est["hill"] = av.hill(CUB, b)
    out = {name: {"moduli": dict(zip(ISO_MODULI, map(float, isotropic.blocks_to_vector(X)))),
                  "a": list(map(float, isotropic.to_lazar2022(X)))} for name, X in est.items()}
    if key in TABLE6:
        out["voigt"]["table6_max_abs_diff"] = float(np.abs(np.array(out["voigt"]["a"]) - TABLE6[key]).max())
    if key in PD_SETS:
        V, Rr = est["voigt"], est["reuss"]
        gen = sl.eigh(Rr["H1"], V["H1"], eigvals_only=True)
        out["reuss_over_voigt"] = {"m3": float(Rr["H3"][0, 0] / V["H3"][0, 0]),
                                   "m2": float(Rr["H2"][0, 0] / V["H2"][0, 0]),
                                   "H1": list(map(float, gen))}
        M = b["T1"]
        mh = M[1:, 0:1]
        out["voigt_minus_reuss_H1_rank_one_residual"] = float(
            np.abs((V["H1"] - Rr["H1"]) - mh @ mh.T / M[0, 0]).max())
        out["reuss_mE_ratio"] = float(av.reuss_mE_ratio(b))
    return est, out


def block_length_summary(key, b, est, C=None):
    """Block lengths of the crystal and of the estimates, and the Lazar-Po length."""
    _label, C0, _Dv, _src = DATASETS[key]
    C = C0 if C is None else C
    out = {}
    ref = L.helmholtz_cubic(*C)
    out["crystal"] = {"lengths": {p: list(map(float, v)) for p, v in L.block_lengths(b, ref).items()},
                      "lazar_po": L.lazar_po_length(CUB, b, ref)}
    cl = L.cubic_classical_estimates(*C)
    for name, X in est.items():
        r = L.helmholtz_isotropic(*cl[name])
        lens = L.block_lengths(X, r)
        flat = np.concatenate([lens[p] for p in ("H3", "H2", "H1")])
        out[name] = {"lame": list(cl[name]), "lengths": {p: list(map(float, v)) for p, v in lens.items()},
                     "max_over_min": float(np.nanmax(flat) / np.nanmin(flat)),
                     "lazar_po": L.lazar_po_length(ISO, X, r)}
    return out


def tabulated(key, C=None, Dv=None):
    """Flat {name: value} of every number of the tables for one dataset."""
    _a, b, cr_ = crystal(key, C, Dv)
    est, avg = estimates(key, b)
    out = {f"moduli/{m}": x for m, x in cr_["moduli"].items()}
    out["condition_number"] = cr_["condition_number"]
    out.update({f"compliance/{m}": x for m, x in cr_.get("compliance_moduli", {}).items()})
    for name in ("voigt", "reuss", "hill"):
        if name in avg:
            out.update({f"{name}/{m}": x for m, x in avg[name]["moduli"].items()})
    if key in PD_SETS:
        bl = block_length_summary(key, b, est, C)
        out["lengths/crystal/lazar_po"] = bl["crystal"]["lazar_po"]
        for name in ("voigt", "reuss", "hill"):
            L_ = bl[name]["lengths"]
            out.update({f"lengths/{name}/l3": L_["H3"][0], f"lengths/{name}/l2": L_["H2"][0],
                        f"lengths/{name}/l1m": L_["H1"][0], f"lengths/{name}/l1p": L_["H1"][1],
                        f"lengths/{name}/max_over_min": bl[name]["max_over_min"],
                        f"lengths/{name}/lazar_po": bl[name]["lazar_po"]})
    return out


def rounding_uncertainty(key):
    """Worst-case linear effect, on every tabulated number, of rounding each published input
    (3 elastic and 11 gradient constants) to its printed precision: sum over inputs of
    |f(x + h e_i) - f(x)| with h half a unit of the last printed digit (analysis/inputs.py).
    Decides how many digits the tables show (make_tables.py)."""
    _label, C, Dv, _src = DATASETS[key]
    hC, hD = inputs.half_units(key)
    base = tabulated(key)
    unc = dict.fromkeys(base, 0.0)
    for kind in ("C", "D"):
        vals = list(C) if kind == "C" else list(Dv)
        for i, x in enumerate(vals):
            pert = list(vals)
            pert[i] = x + (hC if kind == "C" else hD)[i]
            new = tabulated(key, C=pert if kind == "C" else list(C), Dv=pert if kind == "D" else list(Dv))
            for k in base:
                unc[k] += abs(new[k] - base[k])
    return unc


def results():
    res = {"description": __doc__.split("\n\n")[0], "units": "C eV/A^3; D eV/A; compliance A/eV; lengths A",
           "datasets": {}}
    for key in DATASETS:
        _a, b, cr = crystal(key)
        est, avg = estimates(key, b)
        entry = {"crystal": cr, "averages": avg}
        if key in PD_SETS:
            entry["block_lengths"] = block_length_summary(key, b, est)
        entry["rounding_uncertainty"] = rounding_uncertainty(key)
        res["datasets"][key] = entry
    return res


def main():
    res = results()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for key, e in res["datasets"].items():
        cr = e["crystal"]
        print(f"{key:7s} PD={not cr['non_positive_blocks']!s:5s} cond={cr['condition_number']:.3g} "
              f"18x18 check={cr['compliance_check_18x18']:.1e}")
        if "reuss_over_voigt" in e["averages"]:
            rv = e["averages"]["reuss_over_voigt"]
            bl = e["block_lengths"]
            print(f"        R/V m3={rv['m3']:.4f} m2={rv['m2']:.4f} H1={np.round(rv['H1'], 4)}  "
                  f"lengths V={ {p: np.round(v, 3) for p, v in bl['voigt']['lengths'].items()} } "
                  f"lLP crystal={bl['crystal']['lazar_po']:.4f}")
    figure(res)
    print(f"wrote {OUT.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}.pdf/.png")


# ------------------------------------------------------------------ figure
OUT_FIG = ROOT / "analysis" / "figures" / "cubic_averages"
PIECES = (("H3", 0, "$H_3$"), ("H2", 0, "$H_2$"), ("H1", 0, "$H_1^{-}$"), ("H1", 1, "$H_1^{+}$"))


def figure(res):
    """Figure cubic_averages: (a) Reuss/Voigt ratio per isotropic piece;
    (b) block lengths of Voigt (filled) and Reuss (open), each with the C of the same
    estimate, and the Lazar-Po length of Voigt (dotted)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import figstyle as fs
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH_IN, 2.35), constrained_layout=True,
                             gridspec_kw={"wspace": 0.04})
    y = np.arange(len(PIECES))
    off = dict(zip(PD_SETS, (-0.22, 0.0, 0.22)))
    for key in PD_SETS:
        e = res["datasets"][key]
        label = e["crystal"]["label"]
        col, mk = fs.MATERIAL_STYLE[label]
        rv = e["averages"]["reuss_over_voigt"]
        vals = [rv["m3"], rv["m2"], rv["H1"][0], rv["H1"][1]]
        axes[0].plot(vals, y + off[key], mk, color=col, ms=4.5, ls="none", label=label)
        bl = e["block_lengths"]
        for name, fill in (("voigt", col), ("reuss", "white")):
            lv = [bl[name]["lengths"][p][i] for p, i, _ in PIECES]
            axes[1].plot(lv, y + off[key], mk, ms=4.5, color=col, mfc=fill, mew=1.0, ls="none")
        axes[1].axvline(bl["voigt"]["lazar_po"], color=col, lw=0.8, ls=(0, (1, 1.5)))
    axes[0].set_xscale("log")
    axes[0].axvline(1.0, color=fs.INK_SECONDARY, lw=0.6)
    axes[0].annotate(r"$=1$ (rank-one $M^{\mathrm{V}}-M^{\mathrm{R}}$)", (1.0, 3), xytext=(0.03, 3.0), fontsize=7,
                     color=fs.INK_SECONDARY, va="center",
                     arrowprops=dict(arrowstyle="-", color=fs.INK_SECONDARY, lw=0.5))
    axes[0].set_xlabel("Reuss / Voigt")
    axes[1].set_xlabel(r"block length $\ell_p$ (Å)")
    for ax, letter in zip(axes, "ab"):
        ax.set_yticks(y)
        ax.set_yticklabels([lab for *_x, lab in PIECES])
        ax.set_ylim(len(PIECES) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        fs.grid(ax, "x")
        fs.panel_label(ax, letter, x=0.0)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3)
    proxies = [Line2D([], [], marker="o", ls="none", color=fs.INK_SECONDARY, ms=4.5, label="Voigt"),
               Line2D([], [], marker="o", ls="none", color=fs.INK_SECONDARY, mfc="white", mew=1.0, ms=4.5,
                      label="Reuss"),
               Line2D([], [], color=fs.INK_SECONDARY, lw=0.8, ls=(0, (1, 1.5)),
                      label=r"$\ell_{\mathrm{LP}}$ of Voigt")]
    axes[1].legend(handles=proxies, loc="lower left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.4)
    fs.save(fig, OUT_FIG)


if __name__ == "__main__":
    main()
