"""Solyaev's (2023) self-consistent scheme with the full cubic gradient
tensor of the grain (Al MEAM, Cu MEAM, Cu EAM), exact <S6>, 7 and 8 terms.

Writes analysis/results/selfconsistent_fcc.json and the figure
analysis/figures/selfconsistent_fcc.{pdf,png} (supplement, section S5).

Units: C in eV/A^3 and D in eV/A (cubic_gradient_constants.json), so the medium
G = l^2 C (x) delta has l in A and R is in A. Unknowns (nu, l); k is the grain's.

For every (material, terms, R): grid of rho(nu, l), every grid local minimum
polished with Nelder-Mead, distinct minima reported (existence and uniqueness);
residual rho and rho/|Phi(R/l)| (rho vanishes like Phi for large grains, so
the normalized residual is the size-independent measure of misfit); per-term
residuals; B2 - B2'. Also: the closed-form large-grain limit
(selfconsistent.large_grain_limit), Kroner's nu, the Lazar-Po length
l^2 = S_ijkl D_ijmklm / 18 (their eq. 107) of the crystal, of its Voigt and Reuss
averages (with Kroner's S), and the sensitivity of Cu MEAM to the rounding of
D16,16 and D16,17, which fix m_E ~ 5e-5.

Run: python analysis/selfconsistent_fcc.py   (about 3 min on 2 cores)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import concentrator as cc  # noqa: E402
from gradinv import cubic, isotropic  # noqa: E402
from gradinv import selfconsistent as sc  # noqa: E402

import inputs  # noqa: E402

DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
MATERIALS = {inputs.label(k): (inputs.C(k), inputs.D(k)) for k in ("Al", "CuMEAM", "CuEAM")}
R_GRID = [2.5, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0]
NU_GRID = np.linspace(-0.9, 0.49, 50)
ELL_GRID = np.linspace(0.05, 6.0, 80)
TERMS = {"7": sc.TERMS_7, "8": sc.TERMS_8}
CUB, ISO = cubic.algebra(), isotropic.algebra()
OUT_JSON = ROOT / "analysis" / "results" / "selfconsistent_fcc.json"
OUT_FIG = ROOT / "analysis" / "figures" / "selfconsistent_fcc"
# Lattice parameters of Al and Cu (Lazar and Po 2024, Table 2; provenance P09): shaded zone of the
# figure, R < a with the larger of the two (Al), which covers the same grid points for every row
LATTICE = {k: DATA["datasets"]["lazarpo2024"]["lattice_a_table2"][k] for k in ("Al", "Cu")}


def grain_blocks(D_voigt, da=None):
    a = np.array(cubic.a_from_voigt(D_voigt), dtype=float)
    if da is not None:
        a = a + np.asarray(da)
    return a, cubic.from_lazar2022(a)


def objective(C, G, k, R, terms):
    def f(p):
        nu, ell = p
        if not (-1 < nu < 0.5 and ell > 0):
            return np.inf
        with np.errstate(all="ignore"):
            v = sc.error(sc.invariants(C, G, k, nu, ell, R), terms)
        return v if np.isfinite(v) else np.inf
    return f


def all_minima(C, G, R, terms):
    """Every grid local minimum (8 neighbours), polished; distinct ones sorted by rho."""
    k = (C[0] + 2 * C[1]) / 3
    f = objective(C, G, k, R, terms)
    F = np.array([[f((nu, ell)) for ell in ELL_GRID] for nu in NU_GRID])
    seeds = []
    n, m = F.shape
    for i in range(n):
        for j in range(m):
            if not np.isfinite(F[i, j]):
                continue
            nb = F[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            if F[i, j] <= np.nanmin(np.where(np.isfinite(nb), nb, np.inf)):
                seeds.append((NU_GRID[i], ELL_GRID[j]))
    found = []
    for s in seeds:
        r = minimize(f, s, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-11, maxiter=4000))
        nu, ell = map(float, r.x)
        if not np.isfinite(r.fun):
            continue
        if all(abs(nu - q["nu"]) > 1e-4 or abs(ell - q["ell"]) > 1e-3 for q in found):
            found.append({"rho": float(r.fun), "nu": nu, "ell": ell})
    found.sort(key=lambda q: q["rho"])
    for q in found:
        inv = sc.invariants(C, G, k, q["nu"], q["ell"], R)
        q["rho_over_Phi"] = q["rho"] / abs(float(cc.solyaev_phi(R / q["ell"])))
        q["terms"] = {t: float((inv[t] - sc.TARGETS[t]) / sc.TARGETS[t]) for t in sc.TERMS_8}
        q["B2_minus_B2p"] = float(inv["B2"] - inv["B2'"])
        q["at_grid_edge"] = bool(q["ell"] >= ELL_GRID[-1] - 1e-6 or q["nu"] <= NU_GRID[0] + 1e-6)
    return found


def _C4(C11, C12, C44):
    d = np.eye(3)
    C = np.zeros((3, 3, 3, 3))
    for i, j, k, l in np.ndindex(3, 3, 3, 3):
        C[i, j, k, l] = (C12 * d[i, j] * d[k, l] + C44 * (d[i, k] * d[j, l] + d[i, l] * d[j, k])
                         + (C11 - C12 - 2 * C44) * (i == j == k == l))
    return C


def _S4(C4):
    d = np.eye(3)
    I4 = 0.5 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
    w, v = np.linalg.eigh(I4.reshape(9, 9))
    Q = v[:, w > 0.5]
    return (Q @ np.linalg.inv(Q.T @ C4.reshape(9, 9) @ Q) @ Q.T).reshape(3, 3, 3, 3)


def lazar_po_length(D27, C4):
    """l^2 = S_ijkl D_ijmklm / 18 (Lazar and Po 2024, eq. 107), form II, 27x27 storage."""
    D = np.asarray(D27, dtype=float).reshape(3, 3, 3, 3, 3, 3)
    return float(np.sqrt(np.einsum("ijkl,ijmklm->", _S4(C4), D) / 18))


def material_summary(name):
    C, Dv = MATERIALS[name]
    a, blocks = grain_blocks(Dv)
    G = cc.to_form_I(CUB, blocks, "stiffness")
    k = (C[0] + 2 * C[1]) / 3
    nuK = sc.kroner_nu(C)
    lamK, muK = sc.lame_from_k_nu(k, nuK)
    CK = lamK * np.einsum("ij,kl->ijkl", np.eye(3), np.eye(3)) + 2 * muK * 0.5 * (
        np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3)) + np.einsum("il,jk->ijkl", np.eye(3), np.eye(3)))
    out = {"C": C, "k": k, "nu_kroner": nuK, "lame_kroner": [lamK, muK],
           "limit": {t: sc.large_grain_limit(C, G, T) for t, T in TERMS.items()},
           "lazar_po_length": {
               "crystal": lazar_po_length(cubic.lazar2022_tensor(a), _C4(*C)),
               "voigt_with_kroner_S": lazar_po_length(ISO.assemble(av.voigt(CUB, blocks)), CK),
               "reuss_with_kroner_S": lazar_po_length(ISO.assemble(av.reuss(CUB, blocks)), CK)}}
    return out


def run_case(args):
    name, t, R = args
    C, Dv = MATERIALS[name]
    G = cc.to_form_I(CUB, grain_blocks(Dv)[1], "stiffness")
    return name, t, R, all_minima(C, G, R, TERMS[t])


def cu_meam_sensitivity():
    """D16,16 and D16,17 of Cu MEAM (which fix m_E = 2(D16,16 - D16,17)) moved by their
    rounding unit (analysis/inputs.py): l_inf and the solve at R = 5 A."""
    C, Dv = MATERIALS["Cu MEAM"]
    _hC, hD = inputs.half_units("CuMEAM")
    rows = []
    for s16 in (-1, 0, 1):
        for s17 in (-1, 0, 1):
            Dp = list(Dv)
            Dp[9] += s16 * hD[9]
            Dp[10] += s17 * hD[10]
            _, blocks = grain_blocks(Dp)
            G = cc.to_form_I(CUB, blocks, "stiffness")
            lim = sc.large_grain_limit(C, G, sc.TERMS_7)
            D, nu, ell = sc.solve(C, G, 5.0, terms=sc.TERMS_7, nu_grid=NU_GRID, ell_grid=ELL_GRID)
            rows.append({"dD16_16": s16 * hD[9], "dD16_17": s17 * hD[10], "mE": float(blocks["E"][0, 0]),
                         "ell_inf": lim["ell"], "ell_R5": ell, "nu_R5": nu, "rho_R5": D})
    return rows


def figure(res):
    """Figure selfconsistent_fcc (supplement): effective length, Poisson's
    ratio and normalized residual against the grain diameter, 7 terms (solid) and 8 (dashed);
    dotted: large-grain limit and Kroner. Shaded: grains whose radius is below the lattice
    parameter (2R < 2a with the largest a of the three materials, Al)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import figstyle as fs
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT_WIDTH_IN, 2.1), constrained_layout=True)
    shade_nm = 2 * max(LATTICE.values()) / 10
    for name in MATERIALS:
        col, mk = fs.MATERIAL_STYLE[name]
        for t, ls in (("7", "-"), ("8", "--")):
            rows = res["cases"][name][t]
            x = [2 * r["R"] / 10 for r in rows]  # 2R in nm
            best = [r["minima"][0] for r in rows]
            kw = dict(color=col, ls=ls, marker=mk, ms=3, mfc=col if t == "7" else "white", mew=0.8)
            axes[0].plot(x, [b["ell"] for b in best], **kw)
            axes[1].plot(x, [b["nu"] for b in best], **kw)
            axes[2].plot(x, [b["rho_over_Phi"] for b in best], **kw)
        axes[0].axhline(res["materials"][name]["limit"]["7"]["ell"], color=col, lw=0.8, ls=(0, (1, 1.5)))
        axes[1].axhline(res["materials"][name]["nu_kroner"], color=col, lw=0.8, ls=(0, (1, 1.5)))
    ylabels = (r"effective length $\ell$ (Å)", r"Poisson's ratio $\nu$", r"residual $\rho/|\Phi|$")
    for ax, ylab, letter in zip(axes, ylabels, "abc"):
        ax.axvspan(0.0, shade_nm, color=fs.SHADE, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.4, 120)
        ax.set_xlabel("grain diameter $2R$ (nm)")
        ax.set_ylabel(ylab)
        fs.grid(ax)
        fs.panel_label(ax, letter, x=0.0)
    axes[2].text(0.62, 0.30, "$R<a$", transform=axes[2].get_xaxis_transform(), fontsize=7,
                 color=fs.INK_SECONDARY, ha="center", va="center", rotation=90)
    handles = [Line2D([], [], color=fs.MATERIAL_STYLE[n][0], marker=fs.MATERIAL_STYLE[n][1], ms=3, label=n)
               for n in MATERIALS]
    handles += [Line2D([], [], color=fs.INK_SECONDARY, ls="-", label="7 terms"),
                Line2D([], [], color=fs.INK_SECONDARY, ls="--", label="8 terms")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, handletextpad=0.4, columnspacing=1.2)
    fs.save(fig, OUT_FIG)


def main():
    res = {"description": __doc__.split("\n\n")[0], "R_grid_A": R_GRID,
           "grid": {"nu": [NU_GRID[0], NU_GRID[-1], len(NU_GRID)], "ell": [ELL_GRID[0], ELL_GRID[-1], len(ELL_GRID)]},
           "materials": {n: material_summary(n) for n in MATERIALS}, "cases": {}}
    jobs = [(n, t, R) for n in MATERIALS for t in TERMS for R in R_GRID]
    with Pool(2) as pool:
        for n, t, R, minima in pool.imap_unordered(run_case, jobs):
            res["cases"].setdefault(n, {}).setdefault(t, []).append({"R": R, "minima": minima})
            print(f"{n:8s} {t} R={R:6.1f}  minima={len(minima)}  best: nu={minima[0]['nu']:.4f} "
                  f"ell={minima[0]['ell']:.4f} rho={minima[0]['rho']:.3e} "
                  f"D/Phi={minima[0]['rho_over_Phi']:.3f}", flush=True)
    for n in res["cases"]:
        for t in res["cases"][n]:
            res["cases"][n][t].sort(key=lambda r: r["R"])
    res["cu_meam_sensitivity"] = cu_meam_sensitivity()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    figure(res)
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}.pdf/.png")


if __name__ == "__main__":
    main()
