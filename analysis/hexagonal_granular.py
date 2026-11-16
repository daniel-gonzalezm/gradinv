"""Hexagonal gradient tensors of a granular packing and the extra modulus.

The hexagonal tensors come from the granular micromechanics of Pirmoradi, Suiker and
Poorsolhjouy (2024, J. Elast. 156): equal spheres, contact stiffness
K(n) = k_n n n + k_s (I - n n) (their eq. 52 with k_s = k_t), and, in their form I
(kappa_ijk = u_i,jk, the form I of gradinv.forms), the sixth-order tensor of their eq. (70),

    A_ijklmn = (rho_c l^4 / 4) * int K_il(n) n_j n_k n_m n_n xi(n) dS,

with the directional density of contacts of their eq. (71) truncated to

    xi = (1 + a20 P2(x) + a40 P4(x) + a60 P6(x) + a66 P6^6(x) cos 6 phi) / (4 pi),
    x = cos theta,  P6^6(x) = 10395 (1 - x^2)^3.

The density is even in n, so the fifth-order tensor vanishes, and it is invariant under the
hexagonal group with the sixfold axis e3 and a twofold axis e1, so the tensor lies in the
6/mmm algebra of gradinv.hexagonal. The integrand is a polynomial of degree 12 in n:
the Gauss-Legendre (16 nodes in x) x trapezoidal (32 nodes in phi) rule is exact.
Units: rho_c l^4 k_n = 1 (the ratios below do not depend on it).

Checks (no fit, no free parameter):
  - uniform density: their eq. (74), a1 = a2 = a11 = (k_n - k_s)/420, a7 = a8 = (k_n + 6 k_s)/420;
  - a66 = 0: transversely isotropic (invariant under any rotation about e3);
  - a66 moves only m_B1 and m_B2, with Delta = (1440/13) (k_n - k_s) a66 and
    sigma = (m_B1 + m_B2)/2 unchanged;
  - xi > 0 bounds a66, hence Delta/sigma, for each fabric.

The hexagonal statements of section 6 on two bases, with Delta then swept algebraically
(m_B1 = sigma + Delta/2, m_B2 = sigma - Delta/2, |Delta| < 2 sigma):
  - isotropic fabric (a20 = a40 = a60 = 0): the curve m3^R(Delta)/m3^R(0) is the same for
    every isotropic base, since its H3 blocks all equal m3 (c = 5/m3, sigma = m3);
  - transversely isotropic fabric: a20 = 0.2, k_s = k_n/2 (the largest a20 and the stiffness
    ratio of figure 3 of Pirmoradi et al.).
For every Delta: Voigt, Reuss and Hill isotropic moduli; Delta read back without the basis,
Delta = D :: H6 / 4; eta of Auffray et al. (2013) = Delta/4; the gradient compliance (block
inverse checked against the 18 x 18 inverse). Voigt does not depend on Delta; Reuss is even
in Delta and only its H3 modulus moves, with the closed form
    m3^R(Delta) = 7 / (c + 8 sigma / (4 sigma^2 - Delta^2)),
    c = (A2^-1)_33 + 2 (E1^-1)_33 + 2 (E2^-1)_33.

Run: python analysis/hexagonal_granular.py  (a few seconds). Writes
analysis/results/hexagonal_granular.json and the figure
analysis/figures/hexagonal_extra_modulus.{pdf,png}.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from gradinv import averages as av  # noqa: E402
from gradinv import forms, hexagonal, isotropic  # noqa: E402
from gradinv._tensor import rotation_27  # noqa: E402
from gradinv.algebra import BlockAlgebra as BA  # noqa: E402
from gradinv.skeleton import identity_V  # noqa: E402

HEX, ISO = hexagonal.algebra(), isotropic.algebra()
KN, KS = 1.0, 0.5                 # k_s = k_n / 2, as in figure 3 of Pirmoradi et al. (2024)
FABRICS = {"iso": {}, "ti": {"a20": 0.2}}   # a20 = 0.2: largest value of their figure 3
DELTA_COEFF = Fraction(1440, 13)  # Delta = DELTA_COEFF (k_n - k_s) a66 (rho_c l^4 = 1)
DELTA_FRACTIONS = np.round(np.linspace(-1.9, 1.9, 39), 10)   # Delta / sigma
OUT = ROOT / "analysis" / "results" / "hexagonal_granular.json"
ISO_MODULI = [n for n, *_ in isotropic.MODULI]


# ------------------------------------------------------------------ granular tensors
def _quadrature(nx=16, nphi=32):
    x, w = np.polynomial.legendre.leggauss(nx)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    X, P = np.meshgrid(x, phi, indexing="ij")
    W = np.outer(w, np.full(nphi, 2 * np.pi / nphi))
    s = np.sqrt(1 - X ** 2)
    n = np.stack([s * np.cos(P), s * np.sin(P), X], -1).reshape(-1, 3)
    return n, W.ravel(), X.ravel(), P.ravel()


def _legendre(x):
    P2 = (3 * x ** 2 - 1) / 2
    P4 = (35 * x ** 4 - 30 * x ** 2 + 3) / 8
    P6 = (231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5) / 16
    P66 = 10395 * (1 - x ** 2) ** 3
    return P2, P4, P6, P66


def density(x, phi, a20=0.0, a40=0.0, a60=0.0, a66=0.0):
    """Directional density of contacts, eq. (71) of Pirmoradi et al. (2024), truncated."""
    P2, P4, P6, P66 = _legendre(x)
    return (1 + a20 * P2 + a40 * P4 + a60 * P6 + a66 * P66 * np.cos(6 * phi)) / (4 * np.pi)


def granular_form1(kn=KN, ks=KS, **fabric):
    """(C, A): fourth- and sixth-order stiffness of eq. (70), rho_c l^2 = rho_c l^4 = 1."""
    n, w, x, phi = _quadrature()
    f = w * density(x, phi, **fabric)
    K = ks * np.eye(3)[None] + (kn - ks) * np.einsum("ai,al->ail", n, n)
    C = np.einsum("a,aik,aj,al->ijkl", f, K, n, n)
    A = np.einsum("a,ail,aj,ak,am,an->ijklmn", f, K, n, n, n, n) / 4
    return C, A


def _form2_matrix():
    """27 x 27 matrix of kappa = form2_to_form1(eta)."""
    M = np.zeros((27, 27))
    for c in range(27):
        e = np.zeros(27)
        e[c] = 1.0
        M[:, c] = forms.form2_to_form1(e.reshape(3, 3, 3)).ravel()
    return M


def to_form2(A):
    """Form-II tensor D with kappa . A kappa = eta . D eta (kappa = u_i,jk)."""
    M, Pv = _form2_matrix(), identity_V()
    return Pv @ M.T @ np.asarray(A).reshape(27, 27) @ M @ Pv


def granular_blocks(kn=KN, ks=KS, **fabric):
    """Hexagonal blocks of the granular tensor and its distance to the algebra."""
    D = to_form2(granular_form1(kn, ks, **fabric)[1])
    b = HEX.moduli(D)
    return b, float(np.abs(HEX.assemble(b) - D).max())


def max_a66(a20=0.0, a40=0.0, a60=0.0, nx=2001, nphi=1441):
    """Largest |a66| with xi >= 0 (the extremes of cos 6 phi are +-1 at the same x)."""
    x = np.linspace(-1, 1, nx)
    P2, P4, P6, P66 = _legendre(x)
    base = 1 + a20 * P2 + a40 * P4 + a60 * P6
    ok = P66 > 1e-12
    return float(np.min(base[ok] / P66[ok]))


def eq74(kn=KN, ks=KS):
    """Isotropic sixth-order tensor of eqs. (39) and (74) of Pirmoradi et al. (2024)."""
    d = np.eye(3)
    E = lambda s: np.einsum(s, d, d, d)  # noqa: E731
    a1 = a2 = a11 = (kn - ks) / 420
    a7 = a8 = (kn + 6 * ks) / 420
    return (a1 * (E("ij,kl,mn->ijklmn") + E("ik,jl,mn->ijklmn") + E("im,jk,ln->ijklmn") + E("in,jk,lm->ijklmn"))
            + a2 * (E("ij,km,ln->ijklmn") + E("ij,kn,lm->ijklmn") + E("ik,jm,ln->ijklmn") + E("ik,jn,lm->ijklmn"))
            + a7 * E("il,jk,mn->ijklmn")
            + a8 * (E("il,jm,kn->ijklmn") + E("il,jn,km->ijklmn"))
            + a11 * (E("im,jl,kn->ijklmn") + E("im,jn,kl->ijklmn") + E("in,jl,km->ijklmn") + E("in,jm,kl->ijklmn")))


def granular_checks():
    """Checks of the granular construction against Pirmoradi et al. and the algebra."""
    _C, A0 = granular_form1()
    out = {"eq74_max_abs_diff": float(np.abs(A0 - eq74()).max())}
    t = 0.7319
    Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    for key, fab in FABRICS.items():
        a66 = 0.9 * max_a66(**fab)
        C, A = granular_form1(**fab)
        D0 = to_form2(A)
        b0, r0 = granular_blocks(**fab)
        b1, r1 = granular_blocks(**fab, a66=a66)
        C_rot = np.einsum("ip,jq,kr,ls,pqrs->ijkl", Rz, Rz, Rz, Rz, C)
        R27 = rotation_27(Rz)
        out[key] = {
            "a66_max": max_a66(**fab),
            "distance_to_algebra": max(r0, r1),
            "ti_residual_D_a66_0": float(np.abs(R27 @ D0 @ R27.T - D0).max()),
            "ti_residual_C": float(np.abs(C_rot - C).max()),
            "delta_a66_0": float(hexagonal.extra_modulus(b0)),
            "delta_coeff_error": float(abs(hexagonal.extra_modulus(b1) / ((KN - KS) * a66) - float(DELTA_COEFF))),
            "other_blocks_change": float(max(np.abs(b1[p] - b0[p]).max() for p in b0 if p not in ("B1", "B2"))),
            "sigma_change": float(abs((b1["B1"] + b1["B2"] - b0["B1"] - b0["B2"])[0, 0]) / 2),
            "positive_definite": bool(HEX.is_positive_definite(b0) and HEX.is_positive_definite(b1)),
            "delta_max_over_sigma": float(DELTA_COEFF) * (KN - KS) * max_a66(**fab) / float(b0["B1"][0, 0]),
        }
    # the coefficient does not depend on k_n, k_s or the TI fabric
    for kn, ks in ((1.0, 0.0), (0.0, 1.0)):
        b, _ = granular_blocks(kn, ks, a20=0.1, a40=-0.05, a60=0.02, a66=1e-5)
        out[f"delta_coeff_error_kn{kn:g}_ks{ks:g}"] = float(
            abs(hexagonal.extra_modulus(b) / ((kn - ks) * 1e-5) - float(DELTA_COEFF)))
    return out


# ------------------------------------------------------------------ proposition 3
def base_blocks(key):
    return granular_blocks(**FABRICS[key])[0]


def with_delta(blocks, delta):
    sig = blocks["B1"][0, 0]
    out = {k: v.copy() for k, v in blocks.items()}
    out["B1"], out["B2"] = np.array([[sig + delta / 2]]), np.array([[sig - delta / 2]])
    return out


def reuss_m3_closed(blocks, delta):
    sig = blocks["B1"][0, 0]
    Xi = BA.inverse(blocks)
    c = Xi["A2"][0, 0] + 2 * Xi["E1"][0, 0] + 2 * Xi["E2"][0, 0]
    return 7 / (c + 8 * sig / (4 * sig ** 2 - delta ** 2))


def _basis_V():
    w, U = np.linalg.eigh(identity_V())
    return U[:, w > 0.5]


def checks(base):
    """Structural checks on one base (TI at Delta = 0; Delta from H6; compliance)."""
    _h, _k, H6 = hexagonal.structure_tensors()
    t = 0.7319
    Rz = rotation_27(np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]))
    D0 = HEX.assemble(base)
    sig = base["B1"][0, 0]
    D1 = HEX.assemble(with_delta(base, 0.8 * sig))
    Bv = _basis_V()
    S18 = np.linalg.inv(Bv.T @ D1 @ Bv)
    S = HEX.assemble(BA.inverse(with_delta(base, 0.8 * sig)))
    return {"ti_rotation_residual_delta0": float(np.abs(Rz @ D0 @ Rz.T - D0).max()),
            "ti_rotation_residual_delta08": float(np.abs(Rz @ D1 @ Rz.T - D1).max()),
            "delta_from_H6_error": float(abs(np.sum(D1 * H6) / 4 - 0.8 * sig)),
            "eta_auffray_over_delta": float(hexagonal.to_auffray2013(with_delta(base, 0.8 * sig))[11]
                                            / (0.8 * sig)),
            "compliance_check_18x18": float(np.abs(Bv.T @ S @ Bv - S18).max() / np.abs(S18).max()),
            "positive_definite_delta0": HEX.is_positive_definite(base)}


def sweep(key):
    base = base_blocks(key)
    sig = float(base["B1"][0, 0])
    rows = []
    for f in DELTA_FRACTIONS:
        b = with_delta(base, f * sig)
        V, Rr, H = av.voigt(HEX, b), av.reuss(HEX, b), av.hill(HEX, b)
        rows.append({"delta_over_sigma": float(f), "delta": float(f * sig),
                     "voigt": list(map(float, isotropic.blocks_to_vector(V))),
                     "reuss": list(map(float, isotropic.blocks_to_vector(Rr))),
                     "hill": list(map(float, isotropic.blocks_to_vector(H))),
                     "reuss_m3_closed": float(reuss_m3_closed(base, f * sig))})
    V0 = np.array(rows[len(rows) // 2]["voigt"])
    R = np.array([r["reuss"] for r in rows])
    # granular end of the admissible range: the actual granular tensor, not the algebraic sweep
    a66 = max_a66(**FABRICS[key])
    bg, _ = granular_blocks(**FABRICS[key], a66=a66)
    Rg, R0 = av.reuss(HEX, bg), av.reuss(HEX, base)
    return {"fabric": FABRICS[key], "sigma": sig, "rows": rows, "checks": checks(base),
            "base_moduli": dict(zip([m for m, *_ in hexagonal.MODULI],
                                    map(float, hexagonal.blocks_to_vector(base)))),
            "voigt_max_variation": float(max(np.abs(np.array(r["voigt"]) - V0).max() for r in rows)),
            "reuss_evenness": float(np.abs(R - R[::-1]).max()),
            "reuss_non_H3_variation": float(np.abs(R[:, 1:] - R[len(rows) // 2, 1:]).max()),
            "reuss_closed_form_error": float(max(abs(r["reuss"][0] - r["reuss_m3_closed"]) for r in rows)),
            "granular_delta_max_over_sigma": float(hexagonal.extra_modulus(bg)) / sig,
            "granular_reuss_m3_ratio_at_max": float(isotropic.blocks_to_vector(Rg)[0]
                                                    / isotropic.blocks_to_vector(R0)[0])}


def results():
    return {"description": __doc__.split("\n\n")[0], "units": "rho_c l^4 k_n = 1",
            "k_n": KN, "k_s": KS, "delta_coefficient": str(DELTA_COEFF),
            "granular_checks": granular_checks(),
            "sweeps": {key: sweep(key) for key in FABRICS}}


def main():
    res = results()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("granular checks:", {k: (f"{v:.1e}" if isinstance(v, float) else v)
                               for k, v in res["granular_checks"].items() if not isinstance(v, dict)})
    for key, s in res["sweeps"].items():
        g = res["granular_checks"][key]
        mid = s["rows"][len(s["rows"]) // 2]
        one = [r for r in s["rows"] if abs(r["delta_over_sigma"] - 1.0) < 1e-9][0]
        print(f"{key}: sigma={s['sigma']:.5f}  Voigt var={s['voigt_max_variation']:.1e}  "
              f"Reuss even={s['reuss_evenness']:.1e}  non-H3={s['reuss_non_H3_variation']:.1e}  "
              f"closed={s['reuss_closed_form_error']:.1e}  m3R(sigma)/m3R(0)={one['reuss'][0] / mid['reuss'][0]:.4f}  "
              f"granular |Delta|/sigma <= {s['granular_delta_max_over_sigma']:.4f}, "
              f"m3R ratio there {s['granular_reuss_m3_ratio_at_max']:.5f}")
        print("   granular:", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in g.items()})
        print("   checks:", {k: (f"{v:.1e}" if isinstance(v, float) else v) for k, v in s["checks"].items()})
    figure(res)
    print(f"wrote {OUT.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}.pdf/.png")


# ------------------------------------------------------------------ figure
OUT_FIG = ROOT / "analysis" / "figures" / "hexagonal_extra_modulus"


def figure(res):
    """Figure hexagonal_extra_modulus: m3 of Reuss and Hill, over m3 of Voigt, against
    Delta/sigma for the two granular bases; shaded: |Delta| reachable by the TI fabric."""
    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.SINGLE_WIDTH_IN, 2.35), constrained_layout=True)
    styles = {"iso": ("#2a78d6", "isotropic base"), "ti": ("#eb6834", "TI base")}
    g = res["sweeps"]["ti"]["granular_delta_max_over_sigma"]
    ax.axvspan(-g, g, color="#d9d9d9", lw=0, label="granular, TI fabric")
    for key, s_ in res["sweeps"].items():
        col, lab = styles[key]
        x = [r["delta_over_sigma"] for r in s_["rows"]]
        v = s_["rows"][0]["voigt"][0]
        lw = 2.2 if key == "iso" else 1.1   # the curves nearly coincide: the wider one stays visible below
        ax.plot(x, [r["reuss"][0] / v for r in s_["rows"]], "-", color=col, lw=lw, label=f"Reuss, {lab}")
        ax.plot(x, [r["hill"][0] / v for r in s_["rows"]], "--", color=col, lw=lw, label=f"Hill, {lab}")
    ax.axhline(1.0, color=fs.INK, lw=1.1, label="Voigt (both bases)")
    # capital Greek upright, as stix.sty sets it in the text
    ax.set_xlabel(r"extra modulus $\Delta/\sigma$")
    ax.set_ylabel(r"$m_3\,/\,m_3^{\mathrm{V}}$")
    ax.set_ylim(0, 1.08)
    fs.grid(ax)
    ax.legend(loc="lower center", ncol=1, handlelength=2.2)
    fs.save(fig, OUT_FIG)


if __name__ == "__main__":
    main()
