"""Solyaev's (2023) concentration tensor in block form.

Independent routes: index-form transcriptions of the sources
(tests/references/solyaev_sources.py), inversion on V by 18x18 matrices and
index contractions of the full 27x27 concentrator.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.spatial.transform import Rotation

from gradinv import averages as av
from gradinv import concentrator as cc
from gradinv import cubic, hexagonal, isotropic
from gradinv._tensor import rotation_27
from gradinv.algebra import BlockAlgebra
from gradinv.forms import form2_to_form1
from gradinv.skeleton import identity_V, sym_projector
from helpers import basis_V, is_zero
from references import solyaev_sources as ss

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
LZ, PO = DATA["datasets"]["lazar2022"], DATA["datasets"]["po2019"]
GRAINS = {"Al": (LZ["C"]["Al"], cubic.a_from_voigt(LZ["D_voigt"]["Al"])),
          "Cu_EAM": (PO["C"]["Cu_EAM"], cubic.a_from_voigt(PO["D_voigt"]["Cu_EAM"]))}

CUB, HEX, ISO = cubic.algebra(), hexagonal.algebra(), isotropic.algebra()
IV = identity_V()
BV = basis_V()
T = cc.form_I_map()
Ti = BV @ np.linalg.inv(BV.T @ T @ BV) @ BV.T
BA = BlockAlgebra


def inv_V(A):
    return BV @ np.linalg.inv(BV.T @ A @ BV) @ BV.T


def iso_close(X, Y, tol=1e-13):
    return all(np.abs(np.asarray(X[k], float) - np.asarray(Y[k], float)).max() < tol for k in X)


def grain_form_I(name):
    return cc.to_form_I(CUB, cubic.from_lazar2022(GRAINS[name][1]), "stiffness")


def helmholtz_form_I(lam, mu, l2):
    """The Helmholtz model l^2 C (x) delta (form II) in Solyaev's form I."""
    return cc.to_form_I(ISO, isotropic.from_lazar2022([0, lam * l2 / 2, 0, mu * l2, 0]), "stiffness")


# ------------------------------------------------------------------ A kinematics
def test_T_from_its_index_definition():
    d = np.eye(3)
    Tdef = ss.ten(lambda i, j, k, a, b, c: d[k, a] * d[i, b] * d[j, c] + d[k, a] * d[j, b] * d[i, c]
                  - d[i, a] * d[j, b] * d[k, c]) @ IV
    assert np.abs(Tdef - T).max() < 1e-14
    assert np.abs(T - (3 * sym_projector() - 2 * IV)).max() < 1e-14


def test_T_is_isotropic_and_diagonal_in_GL3():
    m = ISO.moduli(T)
    assert np.abs(ISO.residual(T)).max() < 1e-14
    assert np.isclose(m["H3"][0, 0], 1) and np.isclose(m["H2"][0, 0], -2)
    assert np.allclose(m["H1"], np.diag([1, -2]), atol=1e-14)
    for alg in (CUB, HEX):                # T written in the crystal blocks = diag(t_c)
        emb = av.embed_isotropic(alg, m)
        tf = cc.form_factors(alg)
        assert all(np.allclose(emb[p], np.diag(tf[p]), atol=1e-14) for p in emb)
    assert cc.form_factors(CUB) == {"T1": (1, 1, -2), "T2": (1, -2), "E": (-2,), "A2": (1,)}
    assert cc.form_factors(HEX) == {"A1": (-2,), "A2": (1, 1, -2), "B1": (1,), "B2": (1,),
                                    "E1": (1, -2, 1, -2), "E2": (1, -2)}


@pytest.mark.exact
def test_T_exact():
    Tx = cc.form_I_map(exact=True)
    m = isotropic.algebra(exact=True).moduli(Tx)
    assert m["H3"][0, 0] == 1 and m["H2"][0, 0] == -2 and is_zero(m["H1"] - sp.diag(1, -2))


def test_T_agrees_with_gradinv_forms_in_solyaev_order():
    rng = np.random.default_rng(26)
    eta = rng.normal(size=(3, 3, 3))
    eta = 0.5 * (eta + eta.transpose(1, 0, 2))
    kap_g = form2_to_form1(eta)                      # gradinv.forms: kappa_ijk = d_j d_k u_i
    kap_s = np.einsum("kij->ijk", kap_g)             # Solyaev: kappa_ijk = d_i d_j u_k
    assert np.abs(T @ eta.ravel() - kap_s.ravel()).max() < 1e-14


def test_energy_and_block_conversions():
    rng = np.random.default_rng(27)
    eta = rng.normal(size=(3, 3, 3))
    eta = 0.5 * (eta + eta.transpose(1, 0, 2))
    kap = T @ eta.ravel()
    bl = cubic.from_lazar2022(GRAINS["Al"][1])
    D = CUB.assemble(bl)
    G = Ti.T @ D @ Ti
    assert abs(eta.ravel() @ D @ eta.ravel() - kap @ G @ kap) < 1e-12
    blkI = cc.to_form_I(CUB, bl, "stiffness")
    assert all(np.abs(blkI[p] - CUB.moduli(G)[p]).max() < 1e-13 for p in blkI)
    Brand = {p: rng.normal(size=(len(c), len(c))) for p, c, _d in cubic.IRREPS}
    B27 = CUB.assemble(Brand)
    assert all(np.abs(cc.to_form_I(CUB, Brand, "strain")[p] - CUB.moduli(T @ B27 @ Ti)[p]).max() < 1e-13
               for p in Brand)
    assert all(np.abs(cc.to_form_I(CUB, Brand, "compliance")[p] - CUB.moduli(T @ B27 @ T)[p]).max() < 1e-13
               for p in Brand)
    for kind in ("stiffness", "strain", "compliance"):
        back = cc.to_form_II(CUB, cc.to_form_I(CUB, Brand, kind), kind)
        assert all(np.allclose(back[p], Brand[p], atol=1e-14) for p in Brand)


# ------------------------------------------------------------------ B identity, invariants
def test_solyaev_identity_and_its_invariants():
    I6 = ss.solyaev_identity()
    assert np.abs(I6 - IV).max() == 0
    vals = ss.solyaev_contractions(I6)
    assert all(abs(vals[k] - cc.IDENTITY_VALUES[k]) < 1e-14 for k in vals)
    iso_id = {"H3": np.array([[1.0]]), "H2": np.array([[1.0]]), "H1": np.eye(2)}
    inv = cc.invariants_from_isotropic(iso_id)
    assert all(abs(inv[k] - cc.IDENTITY_VALUES[k]) < 1e-14 for k in inv)


@pytest.mark.exact
def test_invariants_closed_form_exact():
    """B1..B5, B2' as linear forms in (y3, y2, Y): exact coefficients, rank 6, B2 - B2'."""
    alg = isotropic.algebra(exact=True)
    y3, y2 = sp.symbols("y3 y2")
    Y = sp.Matrix(2, 2, sp.symbols("Yss Ysr Yrs Yrr"))
    iso = {"H3": sp.Matrix([[y3]]), "H2": sp.Matrix([[y2]]), "H1": Y}
    Biso = alg.assemble(iso)
    closed = cc.invariants_from_isotropic(iso)
    J = ss.solyaev_contraction_tensors()
    rows = []
    for k in cc.INVARIANT_NAMES:
        direct = sum(Biso[q] * int(J[k].flat[q]) for q in range(729) if J[k].flat[q])
        assert is_zero(sp.expand(direct) - closed[k]), k
        rows.append([sp.diff(closed[k], s) for s in (y3, y2, *Y)])
    assert sp.Matrix(rows).rank() == 6
    assert is_zero(closed["B2"] - closed["B2'"] - 3 * sp.sqrt(5) * (Y[1, 0] - Y[0, 1]))


def test_invariants_closed_form_numeric():
    rng = np.random.default_rng(28)
    iso = {"H3": np.array([[0.4]]), "H2": np.array([[-1.1]]), "H1": rng.normal(size=(2, 2))}
    cl = cc.invariants_from_isotropic(iso)
    ct = ss.solyaev_contractions(ISO.assemble(iso))
    assert all(abs(cl[k] - ct[k]) < 1e-13 for k in cl)


# ------------------------------------------------------------------ C media
LAM, MU, L2 = 0.62, 0.21, 1.4                     # eV/A^3 and A^2 (arbitrary positive values)


def test_eq19_read_with_form_I_indices():
    G19 = ss.solyaev_eq19(LAM, MU, L2)
    gm = ISO.moduli(G19)
    assert np.abs(ISO.residual(G19)).max() < 1e-13
    assert np.isclose(gm["H3"][0, 0], 2 * MU * L2) and np.isclose(gm["H2"][0, 0], 2 * MU * L2)
    assert iso_close(cc.solyaev_medium_isotropic(LAM, MU, L2), gm, 1e-14)
    # the Helmholtz model l^2 C (x) delta of form II, moved to form I, is not eq. (19)
    hI = helmholtz_form_I(LAM, MU, L2)
    assert abs(hI["H2"][0, 0] - gm["H2"][0, 0]) > 1e-3
    assert np.isclose(hI["H2"][0, 0], MU * L2 / 2)
    hI27 = Ti.T @ ss.helmholtz_form_II(LAM, MU, L2) @ Ti
    assert iso_close(ISO.moduli(hI27), hI, 1e-13)


def test_solyaev_eq3_closed_moduli():
    A = (0.3, -0.2, 0.15, 0.8, 0.4)
    assert iso_close(cc.solyaev_G_isotropic(*A), ISO.moduli(ss.solyaev_eq3(*A)), 1e-14)
    assert np.abs(ISO.residual(ss.solyaev_eq3(*A))).max() < 1e-14


# ------------------------------------------------------------------ D concentrator
PHI = cc.solyaev_phi(5.0 / np.sqrt(L2))
S6_27 = ss.solyaev_eq21_form(*cc.solyaev_n(LAM, MU, PHI))
G19_27 = ss.solyaev_eq19(LAM, MU, L2)
P_SOL = S6_27 @ inv_V(G19_27)


def _cases():
    rng = np.random.default_rng(26)
    A = rng.normal(size=(2, 2))
    Grnd = ISO.assemble({"H3": np.array([[1.0]]), "H2": np.array([[0.7]]), "H1": A @ A.T + np.eye(2)})
    Prnd = ISO.assemble({"H3": np.array([[0.3]]), "H2": np.array([[-0.2]]),
                         "H1": rng.normal(size=(2, 2)) * 0.3})
    return {"solyaev": (G19_27, P_SOL), "random": (Grnd, Prnd)}


CASES = _cases()


def test_eq21_tensor_and_P():
    assert np.abs(ISO.residual(S6_27)).max() < 1e-14 and np.abs(IV @ S6_27 @ IV - S6_27).max() == 0
    s6c = cc.solyaev_s6_isotropic(*cc.solyaev_n(LAM, MU, PHI))
    assert iso_close(s6c, ISO.moduli(S6_27), 1e-15)
    pcl = BA.multiply(s6c, BA.inverse(cc.solyaev_medium_isotropic(LAM, MU, L2)))
    pm = ISO.moduli(P_SOL)
    assert iso_close(pcl, pm, 1e-13)
    assert abs(pm["H1"][0, 1] - pm["H1"][1, 0]) > 1e-6       # P is not symmetric


@pytest.mark.core
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("grain", list(GRAINS))
def test_block_concentrator_equals_18x18_inversion(case, grain):
    G, P = CASES[case]
    Ggr = Ti.T @ cubic.lazar2022_tensor(GRAINS[grain][1]) @ Ti
    B_direct = inv_V(IV + P @ (Ggr - G))
    blocks = cc.concentrator(CUB, grain_form_I(grain), ISO.moduli(G), ISO.moduli(P))
    assert np.abs(CUB.assemble(blocks) - B_direct).max() / np.abs(B_direct).max() < 1e-12
    # the same in form II, moved by T B T^-1
    blocksII = cc.concentrator(CUB, cubic.from_lazar2022(GRAINS[grain][1]), ISO.moduli(T @ G @ T),
                               ISO.moduli(Ti @ P @ Ti))
    back = cc.to_form_I(CUB, blocksII, "strain")
    assert max(np.abs(back[p] - blocks[p]).max() for p in back) < 1e-12
    # E: closed invariants = index contractions; F: rotating the grain changes nothing
    cl = cc.invariants(CUB, blocks)
    ct = ss.solyaev_contractions(B_direct)
    assert max(abs(cl[k] - ct[k]) for k in cl) < 1e-12
    for Q in Rotation.random(3, random_state=np.random.default_rng(3)).as_matrix():
        Q3 = rotation_27(Q)
        ctr = ss.solyaev_contractions(inv_V(IV + P @ (Q3 @ Ggr @ Q3.T - G)))
        assert max(abs(cl[k] - ctr[k]) for k in cl) < 1e-12


def test_hexagonal_concentrator_equals_18x18_inversion():
    rng = np.random.default_rng(29)
    G, P = CASES["random"]
    gr = {}
    for p, c, _d in hexagonal.IRREPS:
        A = rng.normal(size=(len(c), len(c)))
        gr[p] = 0.5 * (A @ A.T) + np.eye(len(c))
    B_direct = inv_V(IV + P @ (HEX.assemble(gr) - G))
    blocks = cc.concentrator(HEX, gr, ISO.moduli(G), ISO.moduli(P))
    assert np.abs(HEX.assemble(blocks) - B_direct).max() / np.abs(B_direct).max() < 1e-12
    cl, ct = cc.invariants(HEX, blocks), ss.solyaev_contractions(B_direct)
    assert max(abs(cl[k] - ct[k]) for k in cl) < 1e-12


# ------------------------------------------------------------------ G limits
def test_limits_isotropic_grain_medium_grain_cauchy_grain():
    G, P = CASES["solyaev"]
    gm, pm = ISO.moduli(G), ISO.moduli(P)
    iso_grain = cc.to_form_I(CUB, cubic.from_lazar2022(np.r_[0.05, 0.2, 0.1, 0.3, 0.2, np.zeros(6)]),
                             "stiffness")
    Bt = CUB.assemble(cc.concentrator(CUB, iso_grain, gm, pm))
    assert np.abs(ISO.residual(Bt)).max() < 1e-13
    Bb = cc.concentrator(CUB, av.embed_isotropic(CUB, gm), gm, pm)
    assert np.abs(CUB.assemble(Bb) - IV).max() < 1e-13
    inv = cc.invariants(CUB, Bb)
    assert all(abs(inv[k] - cc.IDENTITY_VALUES[k]) < 1e-12 for k in inv)
    zero = {p: np.zeros((len(c), len(c))) for p, c, _d in cubic.IRREPS}
    Bz = cc.concentrator(CUB, zero, gm, pm)
    assert np.abs(CUB.assemble(Bz) - inv_V(IV - S6_27)).max() < 1e-12


@pytest.mark.exact
def test_cauchy_grain_exact():
    """G_gr = 0: B = (I - <S6>)^-1, since P G = <S6> (Solyaev, sec. 4), on symbolic blocks."""
    alg = cubic.algebra(exact=True)
    g3, g2, s3, s2 = sp.symbols("g3 g2 s3 s2", positive=True)
    Gm = sp.Matrix(2, 2, sp.symbols("G11 G12 G12 G22"))
    Sm = sp.Matrix(2, 2, sp.symbols("S11 S12 S21 S22"))
    G = {"H3": sp.Matrix([[g3]]), "H2": sp.Matrix([[g2]]), "H1": Gm}
    S = {"H3": sp.Matrix([[s3]]), "H2": sp.Matrix([[s2]]), "H1": Sm}
    P = BA.multiply(S, BA.inverse(G))
    zero = {p: sp.zeros(len(c), len(c)) for p, c, _d in cubic.IRREPS}
    B = cc.concentrator(alg, zero, G, P)
    target = av.embed_isotropic(alg, BA.inverse({k: sp.eye(v.shape[0]) - v for k, v in S.items()}))
    assert all(is_zero(B[p] - target[p]) for p in B)


# ------------------------------------------------------------------ H sixth invariant
def test_sixth_invariant_is_not_fixed_by_the_five():
    G, P = CASES["random"]
    inv = cc.invariants(CUB, cc.concentrator(CUB, grain_form_I("Al"), ISO.moduli(G), ISO.moduli(P)))
    assert abs(inv["B2"] - inv["B2'"]) > 1e-6


def _fe_case(twoR, which, ell=1.9):
    """Solyaev's Fe example: simplified grain l0^2 C_gr (x) delta and Helmholtz medium,
    both read as form II, with the printed <S6> (eq. 21) or Ma's."""
    fe = ss.FE_EXAMPLE
    k_b, nu = fe["k"], fe["nu"]
    mu, lam = 3 * k_b * (1 - 2 * nu) / (2 * (1 + nu)), 3 * k_b * nu / (1 + nu)
    a = [0, fe["C12"] / 2, 0, fe["C44"], 0, 0, 0, fe["C11"] - fe["C12"] - 2 * fe["C44"], 0, 0, 0]
    grain = cc.to_form_I(CUB, cubic.from_lazar2022(a), "stiffness")
    G = helmholtz_form_I(lam, mu, ell ** 2)
    R_ = twoR / 2
    if which == "printed":
        S6 = cc.solyaev_s6_isotropic(*cc.solyaev_n(lam, mu, cc.solyaev_phi(R_ / ell)))
    else:
        S6 = cc.ma2018_s6_isotropic(cc.ma2018_nhat(lam, mu, cc.ma2018_aifantis_constants(lam, mu, ell ** 2),
                                                   ell, ell, R_))
    return cc.invariants(CUB, cc.concentrator(CUB, grain, G, BA.multiply(S6, BA.inverse(G))))


# ------------------------------------------------------------------ I sources of eqs. 19-21
def test_ma2018_aifantis_tensor_is_the_form_II_helmholtz_model():
    Tma = ss.ma2018_eq26(*cc.ma2018_aifantis_constants(LAM, MU, L2))
    hI27 = Ti.T @ ss.helmholtz_form_II(LAM, MU, L2) @ Ti
    assert np.abs(IV @ Tma @ IV - hI27).max() < 1e-14
    assert np.abs(IV @ Tma @ IV - G19_27).max() > 0.1           # and not eq. (19) literally


@pytest.mark.parametrize("Rr", [2.0, 5.0, 30.0])
def test_solyaev_coefficients_against_ma2018(Rr):
    ell = np.sqrt(L2)
    nh = cc.ma2018_nhat(LAM, MU, cc.ma2018_aifantis_constants(LAM, MU, L2), ell, ell, Rr)
    ns = cc.solyaev_n(LAM, MU, cc.solyaev_phi(Rr / ell))
    # n1, n2, n4, n5 printed = Ma's; n3 printed = n^3 / 2
    assert np.allclose([nh[0], nh[1], nh[3], nh[4]], [ns[0], ns[1], ns[3], ns[4]], rtol=1e-12)
    assert np.isclose(nh[2], 2 * ns[2], rtol=1e-12)
    # Ma's <N^> = the form of eq. (21) with (n1, 2n2, 4n3, 2n4, 2n5), not the printed one
    N27 = ss.ma2018_eq318(nh)
    assert np.abs(N27 - IV @ N27 @ IV).max() < 1e-15
    assert np.abs(ss.solyaev_eq21_form(ns[0], 2 * ns[1], 4 * ns[2], 2 * ns[3], 2 * ns[4]) - N27).max() < 1e-14
    assert iso_close(cc.ma2018_s6_isotropic(nh), ISO.moduli(N27), 1e-14)
    assert not iso_close(cc.solyaev_s6_isotropic(*ns), ISO.moduli(N27), 1e-6)
    # fourth order: Ma's s1, s2 are Solyaev's; Ma's term s2 (dd + dd) = 2 s2 I
    s1, s2, gx = ss.ma2018_s1_s2(LAM, MU, ell, Rr)
    phi = cc.solyaev_phi(Rr / ell)
    assert np.isclose(gx, phi)
    assert np.allclose(cc.solyaev_s4(LAM, MU, phi), (s1, s2))


# ------------------------------------------------------------------ Eshelby limits
@pytest.mark.parametrize("lam,mu", [(1.3, 0.7), (62.0, 26.16), (2.0, 1.0)])
def test_classical_limit_of_eq20_needs_2_s2(lam, mu):
    """With Phi -> 0, s1 d (x) d + s2 I must be the classical Eshelby tensor of a
    sphere; with I = (dd + dd)/2 that needs 2 s2."""
    nu = lam / (2 * (lam + mu))
    hyd, dev = (1 + nu) / (3 * (1 - nu)), 2 * (4 - 5 * nu) / (15 * (1 - nu))
    s1, s2 = cc.solyaev_s4(lam, mu, 0.0)
    assert np.isclose(3 * s1 + 2 * s2, hyd) and np.isclose(2 * s2, dev)
    assert not np.isclose(s2, dev)


def test_limits_of_phi():
    """Phi -> -1/3 as R/l -> 0 (then 1 + 3 Phi -> 0) and Phi ~ -1/(2 R/l) for large R/l."""
    assert abs(cc.solyaev_phi(1e-3) + 1 / 3) < 1e-6
    assert abs(2 * 300.0 * cc.solyaev_phi(300.0) + 1) < 1e-2
    assert round(2 * 50.0 * cc.solyaev_phi(50.0), 5) == -0.99960
    assert round(cc.solyaev_phi(1.0), 8) == -0.27067057
