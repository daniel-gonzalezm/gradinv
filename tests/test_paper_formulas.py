"""Formulas printed in the manuscript, checked against the package.

Every closed formula typed in paper/sections/*.tex that is not already covered
literally by another test is restated here, in the form in which it is printed,
and compared exactly (sympy) with the package:
- section 2: eq. (iso-lazar), isotropic block moduli in a1..a5 of Lazar et al. (2022);
  eq. (iso-units) and the trace sign convention of the H1 copies;
  the couple-stress remark (a tensor supported on R has m3 = Mss = Msr = 0);
- section 3: eq. (cub-lazar), cubic block moduli in a1..a11 as isotropic part plus
  increments; P^E = J_2 O J_2^T; eq. (cub-iso-units); eq. (cub-isotropy);
- appendix A: eq. (lazar-block) entry by entry, the reflection R_theta, and the
  Helmholtz-type example (M_sr = 2 sqrt5/3 lambda l^2);
- section 4: eq. (hex-structure) and the rotations about n; the embeddings
  eq. (hex-embeddings) and the text, rebuilt independently; eq. (hex-iso-units);
  the seventeen isotropy conditions eq. (hex-isotropy); eq. (hex-extra) and the
  statements around it (commuting units, pi/6 rotation, separable tensor);
  eq. (hex-auffray), the orthogonal matrices W1, W3, W4 against the printed matrix of
  Auffray et al. (2013), with m_A1 = j22 - j23 and G_32;
- section 5: block inverses (symmetric and odd), the seven conditions
  eq. (cub-pd) = positive definiteness, eq. (hex-extra-compliance) and the extra modulus
  of the compliance, eq. (cub-lazar-inverse) (exact and numeric) and the Cauchy-type
  relations a3 = 4 a2, a1 = a5 of the Cu EAM data;
- section 6: eqs. (cub-voigt), (cub-reuss) and (hex-voigt-reuss) as printed;
  eq. (hex-reuss-delta), <H6> = 0 and the value 21/23; the rank of V - R on H1 (cubic
  rank one is in test_main_results; hexagonal rank two here); eq. (reuss-factor) and the
  classical facts quoted from Kube and de Jong (2016). The proof of section 6(b) is
  checked in test_averages (test_reuss_below_voigt_proof_identity_exact).
- section 7: the form-I variable u_k,ij = (3 P^S - 2 I) eta; eq. (sc-blocks) with
  the printed embedding of isotropic elements, y3, y2, Y, eqs. (sc-six) and (sc-sixth); the
  first-order large-grain expansion eq. (sc-limit) (remainder O(Phi^2)); eq. (sc-linf) is
  checked against the full minimization in test_application;
- appendix B: eq. (esh-polarization) in indices with exact sphere moments, S = P D,
  eq. (esh-radial), l^2 F = -3 Phi, eq. (esh-helmholtz) as printed; the statements on Ma et
  al. (2018) and Solyaev (2023) are in test_eshelby and test_eshelby_real_space;
- section 8: the extra modulus in the acoustic tensor,
  Q_il = D_ijmlkn xi_j xi_m xi_k xi_n: Delta adds (Delta/8) Re((z.xi)^4 z z), whose
  longitudinal part is (Delta/8) sin^6(theta) cos(6 phi); the rest is transversely isotropic;
- supplement S7: the six class coefficients of the Helmholtz average (the moment, T and
  real-space formulas are literal in test_eshelby and test_eshelby_real_space).
"""
import numpy as np
import pytest
import sympy as sp

from gradinv import cubic, isotropic
from gradinv._tensor import N, TRIPLES, index
from gradinv.skeleton import embeddings, identity_V, sym_projector
from helpers import is_zero
from references.isotropic_sources import lazar2016_b

a = sp.symbols("a1:12")
s2, s5, s6, s30 = sp.sqrt(2), sp.sqrt(5), sp.sqrt(6), sp.sqrt(30)
R = sp.Rational


def _iso_printed(a1, a2, a3, a4, a5):
    m3, m2 = 2 * (a4 + a5), 2 * a4 - a5
    return {"m3": m3, "m2": m2,
            "Msr": s5 / 3 * (a1 + 4 * a2 - 2 * a3),
            "Mss": R(10, 3) * (a1 + a2 + a3) + m3,
            "Mrr": -R(4, 3) * a1 + R(8, 3) * a2 + R(2, 3) * a3 + m2}


@pytest.mark.exact
def test_section2_isotropic_moduli_in_lazar_constants():
    blk = isotropic.from_lazar2022(list(a[:5]), exact=True)
    p = _iso_printed(*a[:5])
    got = {"m3": blk["H3"][0, 0], "m2": blk["H2"][0, 0], "Mss": blk["H1"][0, 0],
           "Msr": blk["H1"][0, 1], "Mrr": blk["H1"][1, 1]}
    assert all(sp.simplify(got[k] - p[k]) == 0 for k in p)


def test_section2_isotropic_units_and_trace_signs():
    E = embeddings()
    I, S = identity_V(), sym_projector()
    alg = isotropic.algebra()
    U = alg.units()
    assert np.allclose(U[("H3", "3", "3")], S - E["s"] @ E["s"].T)
    assert np.allclose(U[("H2", "2", "2")], I - S - E["r"] @ E["r"].T)
    # trace eta_ppk of J_s v and J_r w: positive multiples (sqrt(5/3), 2/sqrt3) of v, w
    tr = np.zeros((3, N))
    for (i, j, k) in TRIPLES:
        if i == j:
            tr[k, index(i, j, k)] = 1.0
    assert np.allclose(tr @ E["s"], np.sqrt(5 / 3) * np.eye(3))
    assert np.allclose(tr @ E["r"], 2 / np.sqrt(3) * np.eye(3))


def test_section2_couple_stress_type_tensor_is_uncoupled():
    rng = np.random.default_rng(3)
    I, S = identity_V(), sym_projector()
    PR = I - S
    A = rng.standard_normal((N, N))
    D = PR @ (A + A.T) @ PR
    blk = isotropic.algebra().moduli(D)
    assert abs(blk["H3"][0, 0]) < 1e-12
    assert np.allclose(blk["H1"][0, :], 0, atol=1e-12)


def _cub_printed():
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = a
    iso = _iso_printed(a1, a2, a3, a4, a5)
    m3, m2 = iso["m3"], iso["m2"]
    return {
        "Mhh": m3 + R(12, 5) * (a6 + a7) + R(3, 5) * a8 + R(2, 5) * a11,
        "N33": m3 + R(4, 3) * (a6 + a7) + R(1, 3) * a8,
        "Mss": iso["Mss"] + R(44, 15) * (a6 + a7) + R(11, 15) * a8 + 2 * a9 + 4 * a10 + R(3, 5) * a11,
        "N22": m2 + R(2, 3) * a6 - R(4, 3) * a7 + R(2, 3) * a8,
        "Mrr": iso["Mrr"] + R(2, 3) * a6 - R(4, 3) * a7 + R(2, 3) * a8,
        "N32": s2 / 3 * (-2 * a6 + a7 + a8),
        "Msr": iso["Msr"] + s5 * (-R(4, 15) * a6 + R(2, 15) * (a7 + a8) + R(2, 5) * (a9 - a10)),
        "Mhs": s6 * (R(8, 15) * (a6 + a7) + R(2, 15) * a8 + R(1, 3) * a9 + R(2, 3) * a10 + R(1, 5) * a11),
        "Mhr": s30 / 15 * (2 * a6 - a7 - a8 + 2 * a9 - 2 * a10),
        "mE": m2, "mA": m3}


@pytest.mark.exact
def test_section3_cubic_moduli_in_lazar_constants():
    blk = cubic.from_lazar2022(list(a), exact=True)
    got = dict(zip([n for n, *_ in cubic.MODULI], cubic.blocks_to_vector(blk)))
    p = _cub_printed()
    assert set(got) == set(p)
    assert all(sp.simplify(sp.expand(got[k] - p[k])) == 0 for k in p), \
        [k for k in p if sp.simplify(sp.expand(got[k] - p[k])) != 0]


def test_section3_cubic_units_tied_to_isotropic_ones():
    CUB, ISO = cubic.algebra(), isotropic.algebra()
    U, V = CUB.units(), ISO.units()
    E = embeddings()
    assert np.allclose(V[("H3", "3", "3")], U[("T1", "h", "h")] + U[("T2", "3", "3")] + U[("A2", "a", "a")])
    assert np.allclose(V[("H2", "2", "2")], U[("T2", "2", "2")] + U[("E", "e", "e")])
    for x in "sr":
        for y in "sr":
            assert np.allclose(V[("H1", x, y)], U[("T1", x, y)])
    # P^E = J_2 O J_2^T, with O acting on the traceless symmetric model space of H2
    O = cubic.structure_tensor().reshape(9, 9)
    from gradinv.skeleton import h2_basis
    B = np.array([np.array(X, dtype=float).reshape(9) for X in h2_basis()]).T  # 9 x 5
    PO = B.T @ O @ B                                                             # 5 x 5
    assert np.allclose(U[("E", "e", "e")], E["2"] @ PO @ E["2"].T)


def test_section3_isotropy_conditions():
    rng = np.random.default_rng(5)
    CUB = cubic.algebra()
    ISO = isotropic.algebra()
    for _ in range(5):
        x = rng.standard_normal(11)
        iso_like = x.copy()
        names = [n for n, *_ in cubic.MODULI]
        v = dict(zip(names, iso_like))
        v["N33"] = v["mA"] = v["Mhh"]
        v["N22"] = v["mE"]
        v["Mhs"] = v["Mhr"] = v["N32"] = 0.0
        D = CUB.assemble(cubic.vector_to_blocks([v[n] for n in names]))
        assert np.allclose(ISO.residual(D), 0, atol=1e-12)
        D2 = CUB.assemble(cubic.vector_to_blocks(list(x)))
        assert not np.allclose(ISO.residual(D2), 0, atol=1e-6)


@pytest.mark.exact
def test_appendix_A_lazar2016_block_entries():
    A5 = a[:5]
    blk = isotropic.from_lazar2022(list(A5), exact=True)
    b1, b2, b3, b4, b5 = lazar2016_b(A5)
    c, s = s5 / 3, R(2, 3)
    Rt = sp.Matrix([[c, s], [s, -c]])
    assert Rt.det() == -1 and is_zero(Rt * Rt.T - sp.eye(2))
    L = sp.Matrix([[b4, s5 / 3 * b5], [s5 / 3 * b5, b3]])
    M = blk["H1"]
    assert is_zero(M - Rt * L * Rt)
    printed = sp.Matrix([[(5 * b4 + 4 * b3) / 9 + R(20, 27) * b5,
                          2 * s5 / 9 * (b4 - b3) - s5 / 27 * b5],
                         [2 * s5 / 9 * (b4 - b3) - s5 / 27 * b5,
                          (4 * b4 + 5 * b3) / 9 - R(20, 27) * b5]])
    assert is_zero(M - printed)
    assert sp.simplify(b1 - blk["H3"][0, 0]) == 0 and sp.simplify(b2 - blk["H2"][0, 0]) == 0
    assert abs(float(sp.acos(c)) * 180 / np.pi - 41.81) < 0.005


def test_appendix_A_helmholtz_example():
    lam, mu, ell2 = 1.3, 0.7, 2.0
    C = np.zeros((3, 3, 3, 3))
    d = np.eye(3)
    C += lam * np.einsum("ij,kl->ijkl", d, d) + mu * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d))
    D = ell2 * np.einsum("ijkl,mn->ijmkln", C, d).reshape(27, 27)
    blk = isotropic.algebra().moduli(D)
    assert np.isclose(blk["H1"][0, 1], 2 * np.sqrt(5) / 3 * lam * ell2)
    c, s = np.sqrt(5) / 3, 2 / 3
    Rt = np.array([[c, s], [s, -c]])
    L = Rt @ blk["H1"] @ Rt
    assert np.allclose(L, np.diag([(3 * lam + 2 * mu) * ell2, 2 * mu * ell2]))


# ======================================================================
# Section 4 (hexagonal algebra) and section 5 (compliance), as printed.
from itertools import permutations

from gradinv import hexagonal
from gradinv.skeleton import embed_h2, embed_vector_r, embed_vector_s

_EPS = np.zeros((3, 3, 3))
for _p in permutations(range(3)):
    _EPS[_p] = np.linalg.det(np.eye(3)[list(_p)])
_D3 = np.eye(3)
_n, _a = np.array([0.0, 0, 1]), np.array([1.0, 0, 0])
_b = np.cross(_n, _a)
_z = _a + 1j * _b


def _sym3(T):
    return sum(T.transpose(p) for p in permutations(range(3))) / 6


def _harm(T):
    S = _sym3(T)
    t = np.einsum("ppk->k", S)
    return S - (np.einsum("ij,k->ijk", _D3, t) + np.einsum("ik,j->ijk", _D3, t)
                + np.einsum("jk,i->ijk", _D3, t)) / 5


def _col(T):
    return np.asarray(T, float).reshape(27, 1)


def _rot(axis, t):
    a = np.asarray(axis, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _act3(Q, T):
    return np.einsum("ip,jq,kr,pqr->ijk", Q, Q, Q, T)


def _printed_structure():
    z3 = np.einsum("i,j,k->ijk", _z, _z, _z)
    z6 = np.einsum("ijk,lmn->ijklmn", z3, z3).reshape(27, 27)
    return z3.real, z3.imag, z6.real


def test_section4_structure_tensors_as_printed():
    h, k, H6 = _printed_structure()
    hp, kp, H6p = hexagonal.structure_tensors()
    assert np.allclose(h, hp) and np.allclose(k, kp) and np.allclose(H6, H6p)
    for T in (h, k):
        assert np.allclose(T, _sym3(T)) and np.allclose(np.einsum("ppk->k", T), 0)
        assert np.isclose(np.linalg.norm(T), 2)
    H = H6.reshape([3] * 6)
    for p in permutations(range(6)):
        assert np.allclose(H, H.transpose(p))
    assert np.allclose(np.einsum("iiklmn->klmn", H), 0)
    # rotation by phi about n maps z to exp(-i phi) z; by pi/3 it reverses h and k,
    # by pi/6 it maps h to k and k to -h
    phi = 0.37
    Q = _rot(_n, phi)
    assert np.allclose(Q @ _z, np.exp(-1j * phi) * _z)
    Q3, Q6 = _rot(_n, np.pi / 3), _rot(_n, np.pi / 6)
    assert np.allclose(_act3(Q3, h), -h) and np.allclose(_act3(Q3, k), -k)
    assert np.allclose(_act3(Q6, h), k) and np.allclose(_act3(Q6, k), -h)
    # B1 is even under the half-turn about a
    Qa = _rot(_a, np.pi)
    assert np.allclose(_act3(Qa, h), h) and np.allclose(_act3(Qa, k), -k)


def _printed_hex_embeddings():
    """Embeddings of eq. (hex-embeddings) and the text, with v in (a, b) and
    X in ((aa - bb)/sqrt2, (ab + ba)/sqrt2), the model bases of the package."""
    f = lambda M: np.array(M.evalf(), dtype=float)
    h, k, _ = _printed_structure()
    vb = [_a, _b]
    Xb = [(np.outer(_a, _a) - np.outer(_b, _b)) / np.sqrt(2),
          (np.outer(_a, _b) + np.outer(_b, _a)) / np.sqrt(2)]
    nn = np.outer(_n, _n)
    sx = lambda v: [sp.Float(x) for x in v]
    J2 = lambda X: f(embed_h2(sp.Matrix(X)))
    Js = lambda v: f(embed_vector_s(sx(v)))
    Jr = lambda v: f(embed_vector_r(sx(v)))
    cross_mat = lambda X: np.einsum("jpq,p,qk->jk", _EPS, _n, X)   # (n x X)_jk = e_jpq n_p X_qk
    O_n = np.einsum("i,j,k,l->ijkl", _n, _n, _n, _n)
    Jh_n = np.sqrt(5 / 2) * (np.einsum("ijkl,l->ijk", O_n, _n)
                             - (np.einsum("i,jk->ijk", _n, _D3) + np.einsum("j,ik->ijk", _n, _D3)
                                + np.einsum("k,ij->ijk", _n, _D3)) / 5)   # J_h of eq. (cub-embeddings)
    E1_3 = np.hstack([_col(np.sqrt(15) / 2 * _harm(np.einsum("i,j,k->ijk", _n, _n, v))) for v in vb])
    E1_2 = np.hstack([J2(np.sqrt(2) * (np.outer(_n, np.cross(_n, v)) + np.outer(np.cross(_n, v), _n)) / 2)
                      for v in vb])
    E2_3 = np.hstack([_col(np.sqrt(3) * _sym3(np.einsum("i,jk->ijk", _n, cross_mat(X)))) for X in Xb])
    out = {"A1": {"2": J2(np.sqrt(3 / 2) * (nn - _D3 / 3))},
           "A2": {"3": _col(np.sqrt(5 / 2) * _harm(np.einsum("i,j,k->ijk", _n, _n, _n))),
                  "s": Js(_n), "r": Jr(_n)},
           "B1": {"3": _col(h / 2)}, "B2": {"3": _col(k / 2)},
           "E1": {"3": E1_3, "2": E1_2, "s": np.hstack([Js(v) for v in vb]),
                  "r": np.hstack([Jr(v) for v in vb])},
           "E2": {"3": E2_3, "2": np.hstack([J2(X) for X in Xb])}}
    assert np.allclose(out["A2"]["3"], _col(Jh_n))
    return out


def test_section4_embeddings_as_printed():
    alg = hexagonal.algebra()
    emb = _printed_hex_embeddings()
    for p, copies, _d in hexagonal.IRREPS:
        for c in copies:
            assert np.allclose(emb[p][c], alg.embedding(p, c), atol=1e-12), (p, c)
    cols = np.hstack([emb[p][c] for p, copies, _d in hexagonal.IRREPS for c in copies])
    assert cols.shape == (27, 18) and np.allclose(cols.T @ cols, np.eye(18))


def test_section4_units_tied_to_isotropic_ones():
    U, V = hexagonal.algebra().units(), isotropic.algebra().units()
    assert np.allclose(V[("H3", "3", "3")], U[("A2", "3", "3")] + U[("B1", "3", "3")] + U[("B2", "3", "3")]
                       + U[("E1", "3", "3")] + U[("E2", "3", "3")])
    assert np.allclose(V[("H2", "2", "2")], U[("A1", "2", "2")] + U[("E1", "2", "2")] + U[("E2", "2", "2")])
    for x in "sr":
        for y in "sr":
            assert np.allclose(V[("H1", x, y)], U[("A2", x, y)] + U[("E1", x, y)])


def _hex_iso_conditions(v):
    """The seventeen conditions of eq. (hex-isotropy) as residuals."""
    A, F, G = v["A2"], v["E1"], v["E2"]
    i3, is_, ir = 0, 1, 2          # A2 = (3, s, r)
    f3, f2, fs, fr = 0, 1, 2, 3    # E1 = (3, 2, s, r)
    mB1, mB2, mA1 = v["B1"][0, 0], v["B2"][0, 0], v["A1"][0, 0]
    return [mB1 - mB2, mB2 - A[i3, i3], A[i3, i3] - F[f3, f3], F[f3, f3] - G[0, 0],
            mA1 - F[f2, f2], F[f2, f2] - G[1, 1],
            A[is_, is_] - F[fs, fs], A[is_, ir] - F[fs, fr], A[ir, ir] - F[fr, fr],
            A[i3, is_], A[i3, ir], F[f3, f2], F[f3, fs], F[f3, fr], F[f2, fs], F[f2, fr], G[0, 1]]


def test_section4_isotropy_conditions():
    rng = np.random.default_rng(64)
    HEX, ISO = hexagonal.algebra(), isotropic.algebra()
    for _ in range(3):
        X = rng.standard_normal((27, 27))
        Diso = ISO.assemble(ISO.moduli(X + X.T))
        b = HEX.moduli(Diso)
        assert np.allclose(_hex_iso_conditions(b), 0, atol=1e-12)
        iso = ISO.moduli(Diso)
        assert np.isclose(b["B1"][0, 0], iso["H3"][0, 0]) and np.isclose(b["A1"][0, 0], iso["H2"][0, 0])
        assert np.allclose(b["A2"][1:, 1:], iso["H1"]) and np.allclose(b["E1"][2:, 2:], iso["H1"])
    # the 17 conditions are independent and, with them, the tensor is isotropic
    Jac = np.array([_hex_iso_conditions(hexagonal.vector_to_blocks(e)) for e in np.eye(22)]).T
    assert np.linalg.matrix_rank(Jac) == 17 == 22 - 5
    x = rng.standard_normal(22)
    x -= np.linalg.pinv(Jac) @ (Jac @ x)                    # project onto the conditions
    D = HEX.assemble(hexagonal.vector_to_blocks(x))
    assert np.allclose(ISO.residual(D), 0, atol=1e-10)


def test_section4_extra_modulus_as_printed():
    U = hexagonal.algebra().units()
    h, k, H6 = _printed_structure()
    PB1, PB2 = U[("B1", "3", "3")], U[("B2", "3", "3")]
    assert np.allclose(PB1, np.outer(h, h) / 4) and np.allclose(PB2, np.outer(k, k) / 4)
    assert np.allclose(PB1 - PB2, H6 / 4)
    # sum = projector onto span(h, k), the m = +-3 part of H3
    S = PB1 + PB2
    assert np.allclose(S @ S, S) and np.isclose(np.trace(S), 2)
    rng = np.random.default_rng(65)
    v = rng.standard_normal(22)
    blk = hexagonal.vector_to_blocks(v)
    D = hexagonal.algebra().assemble(blk)
    assert np.isclose(hexagonal.extra_modulus(blk), np.sum(D * H6) / 4)
    # the rotation by pi/6 about n exchanges P^B1 and P^B2 and reverses Delta
    from gradinv._tensor import rotation_27
    R6 = rotation_27(_rot(_n, np.pi / 6))
    assert np.allclose(R6 @ PB1 @ R6.T, PB2)
    Drot = R6 @ D @ R6.T
    assert np.isclose(hexagonal.extra_modulus(hexagonal.algebra().moduli(Drot)), -hexagonal.extra_modulus(blk))
    # every other unit commutes with all rotations about n
    Rz = rotation_27(_rot(_n, 0.41))
    for key, Uk in U.items():
        moves = not np.allclose(Rz @ Uk @ Rz.T, Uk)
        assert moves == (key[0] in ("B1", "B2")), key
    assert np.allclose(Rz @ S @ Rz.T, S)


def test_section4_separable_tensor_has_no_extra_modulus():
    rng = np.random.default_rng(66)
    c11, c12, c13, c33, c44 = 2.0, 0.8, 0.6, 1.7, 0.5
    C = np.zeros((3, 3, 3, 3))
    Cv = np.array([[c11, c12, c13, 0, 0, 0], [c12, c11, c13, 0, 0, 0], [c13, c13, c33, 0, 0, 0],
                   [0, 0, 0, c44, 0, 0], [0, 0, 0, 0, c44, 0], [0, 0, 0, 0, 0, (c11 - c12) / 2]])
    vo = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    for i in range(3):
        for j in range(3):
            for kk in range(3):
                for l in range(3):
                    C[i, j, kk, l] = Cv[vo[(i, j)], vo[(kk, l)]]
    L = 1.3 * np.outer(_n, _n) + 0.4 * (_D3 - np.outer(_n, _n))
    D = np.einsum("ijkl,mn->ijmkln", C, L).reshape(27, 27)
    alg = hexagonal.algebra()
    assert np.allclose(alg.residual(D), 0, atol=1e-12)
    assert abs(hexagonal.extra_modulus(alg.moduli(D))) < 1e-12


_W1 = sp.Matrix([[15, -3 * sp.sqrt(15), 0, 6 * sp.sqrt(15), 0],
                 [-15, -sp.sqrt(15), 10 * sp.sqrt(3), 2 * sp.sqrt(15), 10 * sp.sqrt(3)],
                 [-15 * s2, -s30, -5 * s6, 2 * s30, -5 * s6],
                 [0, 4 * sp.sqrt(15), -10 * sp.sqrt(3), 2 * sp.sqrt(15), 10 * sp.sqrt(3)],
                 [0, 4 * s30, 5 * s6, 2 * s30, -5 * s6]]) / 30
_W3 = sp.Matrix([[6 * sp.sqrt(10), 6 * sp.sqrt(15), 0, 0, 0],
                 [-3 * sp.sqrt(10), 2 * sp.sqrt(15), 10 * sp.sqrt(3), -5 * s6, 10 * sp.sqrt(3)],
                 [-6 * s5, 2 * s30, -5 * s6, -10 * sp.sqrt(3), -5 * s6],
                 [-3 * sp.sqrt(10), 2 * sp.sqrt(15), 10 * sp.sqrt(3), 5 * s6, -10 * sp.sqrt(3)],
                 [-6 * s5, 2 * s30, -5 * s6, 10 * sp.sqrt(3), 5 * s6]]) / 30
_W4 = sp.Matrix([[0, 2 * sp.sqrt(3), -2 * s6], [-3 * s2, 2 * sp.sqrt(3), s6],
                 [3 * s2, 2 * sp.sqrt(3), s6]]) / 6


@pytest.mark.core
@pytest.mark.exact
def test_section4_change_of_basis_to_auffray2013_as_printed():
    W1p = _W1 * sp.diag(-1, 1, 1, 1, 1)
    for W in (_W1, W1p, _W3, _W4):
        assert is_zero(W.T * W - sp.eye(W.shape[0]))
    P = hexagonal.AUFFRAY2013_PARAMS
    A11, etaAc, H6m, fJ, J4 = hexagonal.auffray2013_blocks(P)
    blk = hexagonal.from_auffray2013(P, exact=True)
    F, A, G = blk["E1"], blk["A2"], blk["E2"]
    mA1, mB1, mB2 = blk["A1"][0, 0], blk["B1"][0, 0], blk["B2"][0, 0]
    assert is_zero(A11 + etaAc - _W1 * sp.diag(mB1, F) * _W1.T)
    assert is_zero(A11 - W1p * sp.diag(mB2, F) * W1p.T)
    assert is_zero(H6m + fJ - _W3 * sp.diag(A, G) * _W3.T)
    assert is_zero(J4 - _W4 * sp.diag(mA1, G) * _W4.T)
    # rows: their basis vectors in the printed groups (eq. 10, table 2 order)
    assert hexagonal.AUFFRAY2013_TABLE2 == ("111", "221", "122", "331", "133", "222", "112", "121",
                                            "332", "233", "333", "113", "131", "223", "232", "123",
                                            "132", "231")
    j11, j12, j22, j23 = P[18], P[19], P[20], P[21]
    eta = P[11]
    assert sp.simplify(mA1 - (j22 - j23)) == 0
    assert sp.simplify(G[0, 1] - s2 / 3 * (-j11 - j12 + j22 + j23)) == 0
    assert sp.simplify(mB1 - mB2 - 4 * eta) == 0


def test_section4_change_of_basis_numeric_against_their_matrix():
    """The printed W applied to random block moduli rebuild their 18 x 18 matrix."""
    rng = np.random.default_rng(67)
    v = rng.standard_normal(22)
    blk = hexagonal.vector_to_blocks(v)
    params = hexagonal.to_auffray2013(blk)
    A11, etaAc, H6m, fJ, J4 = (to_f(M) for M in hexagonal.auffray2013_blocks(list(params)))
    W1, W3, W4 = (to_f(W) for W in (_W1, _W3, _W4))
    W1p = W1 @ np.diag([-1, 1, 1, 1, 1])
    F, A, G = blk["E1"], blk["A2"], blk["E2"]
    bd = lambda x, M: np.block([[np.array([[x]]), np.zeros((1, M.shape[0]))], [np.zeros((M.shape[0], 1)), M]])
    assert np.allclose(A11 + etaAc, W1 @ bd(blk["B1"][0, 0], F) @ W1.T)
    assert np.allclose(A11, W1p @ bd(blk["B2"][0, 0], F) @ W1p.T)
    bd2 = np.block([[A, np.zeros((3, 2))], [np.zeros((2, 3)), G]])
    assert np.allclose(H6m + fJ, W3 @ bd2 @ W3.T)
    assert np.allclose(J4, W4 @ bd(blk["A1"][0, 0], G) @ W4.T)


def to_f(M):
    return np.array(sp.Matrix(M).evalf(), dtype=float)


# ---------------------------------------------------------------------- section 5
def test_section5_block_inverses_and_conditions():
    rng = np.random.default_rng(68)
    for mod, n in ((cubic, 11), (hexagonal, 22)):
        alg = mod.algebra()
        for _ in range(3):
            X = rng.standard_normal((27, 27))
            D = alg.assemble(alg.moduli(X @ X.T + 0.1 * np.eye(27)))
            blk = alg.moduli(D)
            inv = {p: np.linalg.inv(B) for p, B in blk.items()}
            Dinv = alg.assemble(inv)
            from helpers import basis_V
            Bv = basis_V()
            assert np.allclose(Bv.T @ Dinv @ Bv, np.linalg.inv(Bv.T @ D @ Bv))
        # odd (non-symmetric) elements invert the same way
        blk = {p: rng.standard_normal(B.shape) + 3 * np.eye(B.shape[0]) for p, B in alg.moduli(np.eye(27)).items()}
        K = alg.assemble(blk)
        Kinv = alg.assemble({p: np.linalg.inv(B) for p, B in blk.items()})
        from helpers import basis_V
        Bv = basis_V()
        assert np.allclose(Bv.T @ K @ Bv @ Bv.T @ Kinv @ Bv, np.eye(18))


def _cub_pd_printed(b):
    M, Nb, mE, mA = b["T1"], b["T2"], b["E"][0, 0], b["A2"][0, 0]
    return (M[0, 0] > 0 and M[0, 0] * M[1, 1] - M[0, 1] ** 2 > 0 and np.linalg.det(M) > 0
            and Nb[0, 0] > 0 and np.linalg.det(Nb) > 0 and mE > 0 and mA > 0)


@pytest.mark.core
def test_section5_seven_conditions_equal_positive_definiteness():
    rng = np.random.default_rng(69)
    alg = cubic.algebra()
    from helpers import basis_V
    Bv = basis_V()
    hits = 0
    for _ in range(400):
        x = rng.standard_normal(11) + np.array([2, 2, 2, 0, 0, 0, 2, 2, 0, 1, 1]) * 0.8
        b = cubic.vector_to_blocks(x)
        D = alg.assemble(b)
        pd = np.linalg.eigvalsh(Bv.T @ D @ Bv).min() > 0
        assert pd == _cub_pd_printed(b)
        hits += pd
    assert 20 < hits < 380


def test_section5_hexagonal_extra_modulus_of_compliance():
    rng = np.random.default_rng(70)
    U = hexagonal.algebra().units()
    _h, _k, H6 = _printed_structure()
    m1, m2 = 0.7, 1.9
    lhs = U[("B1", "3", "3")] / m1 + U[("B2", "3", "3")] / m2
    rhs = 0.5 * (1 / m1 + 1 / m2) * (U[("B1", "3", "3")] + U[("B2", "3", "3")]) + (1 / m1 - 1 / m2) / 8 * H6
    assert np.allclose(lhs, rhs)
    v = rng.standard_normal(22)
    blk = hexagonal.vector_to_blocks(v)
    inv = {p: np.linalg.inv(B) for p, B in blk.items()}
    Delta = hexagonal.extra_modulus(blk)
    assert np.isclose(hexagonal.extra_modulus(inv), -Delta / (blk["B1"][0, 0] * blk["B2"][0, 0]))


def _cub_inverse_printed(M):
    Mhh, Mss, Mrr, Mhs, Mhr, Msr, N33, N22, N32, mE, mA = M
    u, x, y = Mhh - mA, N33 - mA, N22 - mE
    X = Mss - s6 * Mhs + R(3, 2) * Mhh - R(5, 2) * N33
    Y = Mrr - N22
    Z = Msr - s6 / 2 * Mhr - sp.sqrt(10) / 2 * N32
    return [(2 * X - 5 * Y + s5 * Z) / 15, (X + 5 * Y + 2 * s5 * Z) / 30, (4 * X + 5 * Y - 4 * s5 * Z) / 30,
            (mA + 2 * mE) / 6, (mA - mE) / 3,
            (2 * x + y - 2 * s2 * N32) / 6, (2 * x - 2 * y + s2 * N32) / 6, (x + 2 * y + 2 * s2 * N32) / 3,
            (x - u) / 2 + (s6 * Mhs + s30 * Mhr + 3 * s2 * N32) / 6,
            (x - u) / 2 + (2 * s6 * Mhs - s30 * Mhr - 3 * s2 * N32) / 12,
            (5 * u - 9 * x) / 2]


@pytest.mark.core
@pytest.mark.exact
def test_section5_inverse_of_the_lazar_map_as_printed():
    names = [n for n, *_ in cubic.MODULI]
    assert names == ["Mhh", "Mss", "Mrr", "Mhs", "Mhr", "Msr", "N33", "N22", "N32", "mE", "mA"]
    Msym = sp.symbols(" ".join(names))
    got = cubic.to_lazar2022(cubic.vector_to_blocks(list(Msym), exact=True), exact=True)
    printed = _cub_inverse_printed(Msym)
    assert all(sp.simplify(sp.expand(g - p)) == 0 for g, p in zip(got, printed))
    # isotropic case: u = x = y = 0 and the first line inverts eq. (iso-lazar)
    m3, m2, Mss, Msr, Mrr = sp.symbols("m3 m2 Mss Msr Mrr")
    iso = _cub_inverse_printed([m3, Mss, Mrr, 0, 0, Msr, m3, m2, 0, m2, m3])
    assert all(sp.simplify(c) == 0 for c in iso[5:])
    back = _iso_printed(*iso[:5])
    assert all(sp.simplify(back[k] - v) == 0 for k, v in
               {"m3": m3, "m2": m2, "Mss": Mss, "Msr": Msr, "Mrr": Mrr}.items())


def test_section5_inverse_of_the_lazar_map_numeric():
    rng = np.random.default_rng(71)
    av = rng.standard_normal(11)
    v = cubic.blocks_to_vector(cubic.from_lazar2022(av))
    printed = [float(e) for e in _cub_inverse_printed([sp.Float(x) for x in v])]
    assert np.allclose(printed, av)


def test_section5_cauchy_relations_of_the_cu_eam_data():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).resolve().parents[1] / "analysis" / "data"
                       / "cubic_gradient_constants.json").read_text())
    keys = data["_voigt_keys"]
    v = data["datasets"]["po2019"]["D_voigt"]["Cu_EAM"]
    Dv = dict(zip(keys, v))
    assert Dv["D2,4"] == Dv["D3,5"] and Dv["D2,5"] == Dv["D16,17"]
    a = cubic.a_from_voigt(v)
    assert np.isclose(a[2], 4 * a[1]) and np.isclose(a[0], a[4])
    # the MEAM sets do not obey them
    vm = data["datasets"]["po2019"]["D_voigt"]["Cu_MEAM"]
    am = cubic.a_from_voigt(vm)
    assert not np.isclose(am[2], 4 * am[1])


# ---------------------------------------------------------------------- section 6
def _sym_block(prefix, n):
    M = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            M[i, j] = M[j, i] = sp.Symbol(f"{prefix}{i}{j}")
    return M


def _rat_pd(rng, n):
    A = sp.Matrix(n, n, lambda i, j: R(int(rng.integers(-4, 5)), 5))
    return A * A.T + sp.eye(n) / 3


@pytest.mark.exact
def test_section6_cubic_voigt_and_reuss_as_printed():
    """eqs. (cub-voigt) and (cub-reuss): symbolic Voigt; Reuss exact on rational tensors."""
    from gradinv import averages as avg
    alg = cubic.algebra(exact=True)
    X = {"T1": _sym_block("M", 3), "T2": _sym_block("N", 2),
         "E": sp.Matrix([[sp.Symbol("mE")]]), "A2": sp.Matrix([[sp.Symbol("mA")]])}
    v = avg.voigt(alg, X)
    M, Nb, mE, mA = X["T1"], X["T2"], X["E"][0, 0], X["A2"][0, 0]
    assert is_zero(sp.Matrix([[v["H3"][0, 0] - (3 * M[0, 0] + 3 * Nb[0, 0] + mA) / 7,
                               v["H2"][0, 0] - (3 * Nb[1, 1] + 2 * mE) / 5]]))
    assert is_zero(v["H1"] - M[1:, 1:])
    rng = np.random.default_rng(606)
    for _ in range(3):
        X = {"T1": _rat_pd(rng, 3), "T2": _rat_pd(rng, 2), "E": _rat_pd(rng, 1), "A2": _rat_pd(rng, 1)}
        M, Nb, mE, mA = X["T1"], X["T2"], X["E"][0, 0], X["A2"][0, 0]
        Mi, Ni = M.inv(), Nb.inv()
        mh = M[1:, 0]
        r = avg.reuss(alg, X, check_pd=False)
        assert r["H3"][0, 0] == 1 / ((3 * Mi[0, 0] + 3 * Ni[0, 0] + 1 / mA) / 7)
        assert r["H2"][0, 0] == 1 / ((3 * Ni[1, 1] + 2 / mE) / 5)
        assert is_zero(r["H1"] - (M[1:, 1:] - mh * mh.T / M[0, 0]))
        assert is_zero(r["H1"] - Mi[1:, 1:].inv())          # Schur complement = inverse of (M^-1)_(sr)


@pytest.mark.exact
def test_section6_hexagonal_voigt_and_reuss_as_printed():
    """eq. (hex-voigt-reuss), with c = (A^-1)_33 + 2 (F^-1)_33 + 2 (G^-1)_33."""
    from gradinv import averages as avg
    alg = hexagonal.algebra(exact=True)
    rng = np.random.default_rng(607)
    for _ in range(2):
        X = {p: _rat_pd(rng, len(c)) for p, c, _d in alg.irreps}
        A, F, G = X["A2"], X["E1"], X["E2"]
        mA1, mB1, mB2 = X["A1"][0, 0], X["B1"][0, 0], X["B2"][0, 0]
        v, r = avg.voigt(alg, X), avg.reuss(alg, X, check_pd=False)
        assert v["H3"][0, 0] == (A[0, 0] + mB1 + mB2 + 2 * F[0, 0] + 2 * G[0, 0]) / 7
        assert v["H2"][0, 0] == (mA1 + 2 * F[1, 1] + 2 * G[1, 1]) / 5
        assert is_zero(v["H1"] - (A[1:, 1:] + 2 * F[2:, 2:]) / 3)
        Ai, Fi, Gi = A.inv(), F.inv(), G.inv()
        c = Ai[0, 0] + 2 * Fi[0, 0] + 2 * Gi[0, 0]
        assert r["H3"][0, 0] == 7 / (c + 1 / mB1 + 1 / mB2)
        assert r["H2"][0, 0] == 5 / (1 / mA1 + 2 * Fi[1, 1] + 2 * Gi[1, 1])
        assert is_zero(r["H1"].inv() - (Ai[1:, 1:] + 2 * Fi[2:, 2:]) / 3)


@pytest.mark.exact
def test_section6_hexagonal_extra_modulus_statements():
    """eq. (hex-reuss-delta) and the text around it: <H6> = 0 (Voigt independent of Delta),
    1/m_B1 + 1/m_B2 = 8 sigma/(4 sigma^2 - Delta^2), and m3^R(sigma)/m3^R(0) = 21/23 for
    an isotropic base."""
    from gradinv import averages as avg
    alg = hexagonal.algebra(exact=True)
    U = alg.units()
    cop = {p: c[0] for p, c, _d in alg.irreps}
    H6 = 4 * (U[("B1", cop["B1"], cop["B1"])] - U[("B2", cop["B2"], cop["B2"])])   # eq. (hex-extra)
    iso = isotropic.algebra(exact=True).moduli(H6)
    assert all(is_zero(B) for B in iso.values())
    sig, Dl, cc = sp.symbols("sigma Delta c", positive=True)
    m3R = 7 / (cc + 1 / (sig + Dl / 2) + 1 / (sig - Dl / 2))
    assert sp.simplify(m3R - 7 / (cc + 8 * sig / (4 * sig ** 2 - Dl ** 2))) == 0
    base = {"H3": sp.Matrix([[sig]]), "H2": sp.Matrix([[sig / 4]]),
            "H1": sp.Matrix([[2 * sig, sig / 3], [sig / 3, sig]])}
    X = avg.embed_isotropic(alg, base)
    X0 = dict(X)
    X["B1"], X["B2"] = sp.Matrix([[sig + sig / 2]]), sp.Matrix([[sig - sig / 2]])
    r1, r0 = avg.reuss(alg, X, check_pd=False), avg.reuss(alg, X0, check_pd=False)
    assert sp.simplify(r1["H3"][0, 0] / r0["H3"][0, 0] - R(21, 23)) == 0


def test_section6_rank_of_voigt_minus_reuss_on_H1():
    """Rank one in the cubic class (any T1 block); rank two in general in the hexagonal one."""
    from gradinv import averages as avg
    rng = np.random.default_rng(608)
    for _ in range(20):
        Xh = {}
        for p, c, _d in hexagonal.IRREPS:
            A = rng.standard_normal((len(c), len(c)))
            Xh[p] = A @ A.T + 0.1 * np.eye(len(c))
        HEXA = hexagonal.algebra()
        G = avg.voigt(HEXA, Xh)["H1"] - avg.reuss(HEXA, Xh)["H1"]
        w = np.linalg.eigvalsh(G)
        assert w.min() > 1e-10 * w.max()                    # positive definite: rank two


@pytest.mark.exact
def test_section6_reuss_factor_and_its_classical_analogue():
    """eq. (reuss-factor) and the classical facts quoted from Kube and de Jong (2016):
    kappa_V = kappa_R (their eq. 27), mu_V - mu_R = (3/5)(c11 - c12 - 2 c44)^2 /
    (3 (c11 - c12) + 4 c44) (eq. 28), mu_R -> (5/2) c' as c' -> 0 (from eq. 30)."""
    from gradinv import averages as avg
    from gradinv.lengths import cubic_classical_estimates
    alg = cubic.algebra(exact=True)
    mE, n33, n32, n22 = sp.symbols("mE n33 n32 n22", positive=True)
    Nb = sp.Matrix([[n33, n32], [n32, n22]])
    X = {"T1": sp.diag(1, 2, 3), "T2": Nb, "E": sp.Matrix([[mE]]), "A2": sp.Matrix([[1]])}
    m2R = avg.reuss(alg, X, check_pd=False)["H2"][0, 0]
    printed = R(5, 2) / (1 + R(3, 2) * mE * Nb.inv()[1, 1])
    assert sp.simplify(m2R / mE - printed) == 0
    assert sp.limit(printed, mE, 0) == R(5, 2)
    assert sp.simplify(avg.reuss_mE_ratio(X) - printed) == 0
    c11, c12, c44, cp = sp.symbols("c11 c12 c44 cp", positive=True)
    est = cubic_classical_estimates(c11, c12, c44)
    kap = {k: lam + 2 * mu / 3 for k, (lam, mu) in est.items()}
    assert sp.simplify(kap["voigt"] - kap["reuss"]) == 0
    muV, muR = est["voigt"][1], est["reuss"][1]
    assert sp.simplify(muV - muR - R(3, 5) * (c11 - c12 - 2 * c44) ** 2
                       / (3 * (c11 - c12) + 4 * c44)) == 0
    muR_cp = muR.subs(c11, c12 + 2 * cp)
    assert sp.limit(muR_cp / cp, cp, 0) == R(5, 2)


# ------------------------------------------------------------------ section 7, appendix B, S7
def _rand_V(rng):
    X = rng.normal(size=(3, 3, 3))
    return (X + X.transpose(1, 0, 2)) / 2


def test_section7_form_I_variable():
    """u_k,ij = eta_kij + eta_kji - eta_ijk = ((3 P^S - 2 I) eta)_ijk, with eta_ijk = eps_ij,k;
    3 P^S - 2 I is 1 on S and -2 on R."""
    rng = np.random.default_rng(67)
    B = rng.normal(size=(3, 3, 3))
    B = (B + B.transpose(0, 2, 1)) / 2              # u_k = B_kij x_i x_j / 2, so u_k,ij = B_kij
    grad_u = B                                      # u_i,jk = B_ijk
    eta = 0.5 * (grad_u + grad_u.transpose(1, 0, 2))  # eps_ij,k = (u_i,jk + u_j,ik)/2
    kappa = np.einsum("kij->ijk", B)                # kappa~_ijk = u_k,ij
    printed = np.einsum("kij->ijk", eta) + np.einsum("kji->ijk", eta) - eta
    assert np.allclose(kappa, printed)
    Tmap = 3 * sym_projector() - 2 * identity_V()
    assert np.allclose(Tmap @ eta.ravel(), kappa.ravel())
    w = np.linalg.eigvalsh((Tmap + Tmap.T) / 2)
    assert np.allclose(np.sort(w), [-2] * 8 + [0] * 9 + [1] * 10)


def test_section7_concentrator_blocks_and_six_conditions():
    """eq. (sc-blocks) with the printed embedding of isotropic elements, the printed
    y3, y2, Y, eq. (sc-six) and eq. (sc-sixth), for a random grain and a non-symmetric L."""
    from gradinv import averages as avg
    from gradinv import concentrator as cc
    rng = np.random.default_rng(7)
    CUB = cubic.algebra()
    grain = {p: rng.normal(size=(len(c), len(c))) + 3 * np.eye(len(c)) for p, c, _d in CUB.irreps}
    medium = {"H3": np.array([[1.4]]), "H2": np.array([[0.8]]), "H1": np.array([[2.0, 0.3], [0.3, 1.1]])}
    L = {"H3": np.array([[0.2]]), "H2": np.array([[-0.1]]), "H1": rng.normal(size=(2, 2)) * 0.3}

    def emb(iso):
        m3, m2, M = iso["H3"][0, 0], iso["H2"][0, 0], iso["H1"]
        T1 = np.zeros((3, 3)); T1[0, 0] = m3; T1[1:, 1:] = M
        return {"T1": T1, "T2": np.diag([m3, m2]), "E": np.array([[m2]]), "A2": np.array([[m3]])}
    Bp = {p: np.linalg.inv(np.eye(len(X)) + emb(L)[p] @ (X - emb(medium)[p])) for p, X in grain.items()}
    pkg = cc.concentrator(CUB, grain, medium, L)
    assert all(np.allclose(Bp[p], pkg[p]) for p in Bp)
    y3 = (3 * Bp["T1"][0, 0] + 3 * Bp["T2"][0, 0] + Bp["A2"][0, 0]) / 7
    y2 = (3 * Bp["T2"][1, 1] + 2 * Bp["E"][0, 0]) / 5
    Y = Bp["T1"][1:, 1:]
    iso = avg.isotropic_part(CUB, Bp)
    assert np.isclose(iso["H3"][0, 0], y3) and np.isclose(iso["H2"][0, 0], y2) and np.allclose(iso["H1"], Y)
    inv = cc.invariants(CUB, Bp)
    assert np.isclose(inv["B2"] - inv["B2'"], 3 * np.sqrt(5) * (Y[1, 0] - Y[0, 1]))
    # <B> = I  <=>  y3 = y2 = 1, Y = I  <=>  the six invariants take the values of I; the five
    # of Solyaev do not suffice: an isotropic element with Y = I + antisymmetric part
    ident = cc.invariants_from_isotropic({"H3": np.eye(1), "H2": np.eye(1), "H1": np.eye(2)})
    assert all(np.isclose(ident[k], v) for k, v in cc.IDENTITY_VALUES.items())
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    skew = cc.invariants_from_isotropic({"H3": np.eye(1), "H2": np.eye(1), "H1": np.eye(2) + 0.1 * J})
    assert all(np.isclose(skew[k], ident[k]) for k in ("B1", "B3", "B4", "B5"))
    assert not np.isclose(skew["B2"], 3) and np.isclose(skew["B2"] + skew["B2'"], 6)
    # five invariants alone: B2 and B2' move together only by the antisymmetric part
    M6 = np.array([[cc.invariants_from_isotropic({"H3": np.array([[float(i == 0)]]),
                                                  "H2": np.array([[float(i == 1)]]),
                                                  "H1": np.eye(2)[[0, 0, 1, 1][i - 2]][:, None]
                                                  @ np.eye(2)[[0, 1, 0, 1][i - 2]][None, :]
                                                  if i >= 2 else np.zeros((2, 2))})[k]
                    for i in range(6)] for k in cc.INVARIANT_NAMES])
    assert np.linalg.matrix_rank(M6) == 6
    assert np.linalg.matrix_rank(np.delete(M6, cc.INVARIANT_NAMES.index("B2'"), axis=0)) == 5


def test_section7_large_grain_first_order():
    """eq. (sc-limit): <B> - I = -Phi S^ D^-1 (l^-2 V - D^) + O(Phi^2), with V the Voigt
    estimate of the grain (form I), for the Al MEAM grain in a Helmholtz medium."""
    import json
    from pathlib import Path
    from gradinv import averages as avg
    from gradinv import concentrator as cc
    from gradinv import selfconsistent as sc
    from gradinv.algebra import BlockAlgebra as BA
    data = json.loads((Path(__file__).resolve().parents[1] / "analysis" / "data"
                       / "cubic_gradient_constants.json").read_text())
    CUB = cubic.algebra()
    G = cc.to_form_I(CUB, cubic.from_lazar2022(cubic.a_from_voigt(data["datasets"]["lazar2022"]["D_voigt"]["Al"])),
                     "stiffness")
    lam, mu, ell = 0.6, 0.25, 1.3
    Dhat = sc.helmholtz_medium(lam, mu, 1.0)
    V = avg.voigt(CUB, G)
    errs = []
    for R_ in (100.0, 200.0, 400.0):
        phi = cc.solyaev_phi(R_ / ell)
        S = sc.averaged_S6("exact", lam, mu, ell, R_)
        Shat = {p: v / phi for p, v in S.items()}
        D = sc.helmholtz_medium(lam, mu, ell ** 2)
        exact = avg.isotropic_part(CUB, cc.concentrator(CUB, G, D, BA.multiply(S, BA.inverse(D))))
        inner = {p: V[p] / ell ** 2 - Dhat[p] for p in V}
        first = BA.multiply(BA.multiply(Shat, BA.inverse(Dhat)), inner)
        e = max(np.abs(exact[p] - np.eye(len(exact[p])) + phi * first[p]).max() for p in exact)
        f = max(np.abs(phi * first[p]).max() for p in first)
        errs.append((abs(phi), e, f))
    for (p1, e1, f1), (p2, e2, _f2) in zip(errs, errs[1:]):
        assert e1 < 0.05 * f1
        assert 0.8 < (e1 / p1 ** 2) / (e2 / p2 ** 2) < 1.25        # remainder is O(Phi^2)


def test_appendixB_polarization_in_indices_and_helmholtz_factor():
    """eq. (esh-polarization) in indices with exact sphere moments equals the package's P,
    S_bar = P D for generic constants, and l^2 F(l) = -3 Phi(R/l) (Solyaev's Phi)."""
    from gradinv import concentrator as cc
    from gradinv import eshelby as es
    ISO = isotropic.algebra()
    IV = identity_V()
    d = np.eye(3)
    n4 = (np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d)) / 15
    n6 = np.zeros((3,) * 6)
    import itertools as it
    for idx in it.product(range(3), repeat=6):
        cnt = np.bincount(np.array(idx), minlength=3)
        if np.all(cnt % 2 == 0):
            # <n_x^2a n_y^2b n_z^2c> = (2a-1)!!(2b-1)!!(2c-1)!!/105 for a+b+c = 3
            df = {0: 1, 2: 1, 4: 3, 6: 15}
            n6[idx] = df[cnt[0]] * df[cnt[1]] * df[cnt[2]] / 105
    lam, mu, R_ = 1.3, 0.7, 0.9
    a = (0.11, -0.05, 0.21, 0.37, 0.13)
    l1s, l2s = es.lengths_squared(lam, mu, a)
    F1, F2 = es.radial_factor(np.sqrt(l1s), R_), es.radial_factor(np.sqrt(l2s), R_)
    P = (F1 / (lam + 2 * mu) * n6
         + F2 / mu * (np.einsum("rtij,sk->rtsijk", n4, d) - n6)).reshape(27, 27)
    P = IV @ P @ IV
    assert np.allclose(P, P.T)
    pk = es.polarization_isotropic(lam, mu, np.sqrt(l1s), np.sqrt(l2s), R_)
    assert np.allclose(P, ISO.assemble(pk))
    Sbar = ISO.assemble(es.sphere_N_isotropic(lam, mu, a, R_))
    T = ISO.assemble(es.ma2018_T_isotropic(a))
    assert np.allclose(Sbar, P @ T)
    assert not np.allclose(Sbar, Sbar.T)
    for R_, ell in ((0.5, 1.0), (3.0, 1.9), (20.0, 2.0)):
        assert np.isclose(ell ** 2 * es.radial_factor(ell, R_), -3 * cc.solyaev_phi(R_ / ell))
    # printed radial factor (esh-radial), with E = exp(-2R/l)
    for R_, l in ((0.5, 1.0), (3.0, 1.9)):
        E = np.exp(-2 * R_ / l)
        assert np.isclose(3 / (2 * R_ ** 3 * l) * (R_ ** 2 * (1 + E) + 2 * l * R_ * E - l ** 2 * (1 - E)),
                          es.radial_factor(l, R_))


def test_appendixB_helmholtz_blocks_as_printed():
    """eq. (esh-helmholtz), with -Phi factored out of M as printed."""
    from gradinv import concentrator as cc
    from gradinv import eshelby as es
    lam, mu = 1.3, 0.7
    c = lam + 2 * mu
    for R_, ell in ((1.0, 1.0), (5.0, 1.9)):
        phi = cc.solyaev_phi(R_ / ell)
        bl = es.helmholtz_sphere_N_isotropic(lam, mu, ell ** 2, R_)
        M = -phi * np.array([[(19 * lam + 26 * mu) / (15 * c), -2 * np.sqrt(5) * (lam - mu) / (15 * c)],
                             [4 * np.sqrt(5) / 15, 1 / 3]])
        assert np.isclose(bl["H3"][0, 0], -4 * (4 * lam + 11 * mu) / (35 * c) * phi)
        assert np.isclose(bl["H2"][0, 0], -phi / 5) and np.allclose(bl["H1"], M)


def test_appendix_B_six_class_coefficients_of_the_helmholtz_average():
    """Appendix B: the exact average in the six classes (K1, K2a, K2b, K3, K4, K5 in the
    order of tests/test_eshelby.py::SIX) and the ratio -2/5 of the two halves of n^_2."""
    from gradinv import concentrator as cc
    from gradinv import eshelby as es
    from test_eshelby import N27, coords6
    lam, mu = 1.3, 0.7
    c = lam + 2 * mu
    for R_, ell in ((1.0, 1.0), (5.0, 1.9)):
        phi = cc.solyaev_phi(R_ / ell)
        coef, res = coords6(N27(es.helmholtz_sphere_N_isotropic(lam, mu, ell ** 2, R_)))
        printed = phi * np.array([-(5 * lam + 12 * mu) / (35 * c), -(lam + mu) / (7 * c),
                                  2 * (lam + mu) / (35 * c), -(3 * lam + 10 * mu) / (70 * c),
                                  -(5 * lam + 12 * mu) / (35 * c), -(3 * lam - 4 * mu) / (70 * c)])
        assert res < 1e-13 and np.allclose(coef, printed)
        assert np.isclose(coef[2] / coef[1], -0.4)


def test_section8_extra_modulus_in_the_acoustic_tensor():
    """Section 8: Q_il(xi) = D_ijmlkn xi_j xi_m xi_k xi_n (Rosi and Auffray 2016, eq. 11).
    For a hexagonal tensor, the part with m_B1 = m_B2 = sigma gives a transversely
    isotropic Q (Q(R xi) = R Q(xi) R^T for every rotation R about n), and the extra modulus
    adds (Delta/8) Re((z.xi)^4 z (x) z), with longitudinal component
    (Delta/8) sin^6(theta) cos(6 phi)."""
    rng = np.random.default_rng(68)
    blk = hexagonal.vector_to_blocks(rng.standard_normal(22))
    sig = 0.5 * (blk["B1"][0, 0] + blk["B2"][0, 0])
    delta = hexagonal.extra_modulus(blk)
    ti = {p: np.array(b, float) for p, b in blk.items()}
    ti["B1"] = np.array([[sig]])
    ti["B2"] = np.array([[sig]])
    alg = hexagonal.algebra()
    D = np.asarray(alg.assemble(blk), float).reshape([3] * 6)
    Dti = np.asarray(alg.assemble(ti), float).reshape([3] * 6)

    def Q(T, x):
        return np.einsum("ijmlkn,j,m,k,n->il", T, x, x, x, x)

    for th, ph in rng.uniform(0, np.pi, size=(12, 2)):
        x = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        extra = delta / 8 * np.real((_z @ x) ** 4 * np.outer(_z, _z))
        assert np.allclose(Q(D, x) - Q(Dti, x), extra)
        assert np.isclose(x @ extra @ x, delta / 8 * np.sin(th) ** 6 * np.cos(6 * ph))
        Rz = _rot(_n, rng.uniform(0, 2 * np.pi))
        assert np.allclose(Q(Dti, Rz @ x), Rz @ Q(Dti, x) @ Rz.T)
    # a nonzero Delta does change Q with the azimuth: the TI test above fails for D
    x = np.array([1.0, 0.0, 0.0])
    Rz = _rot(_n, np.pi / 6)
    assert not np.allclose(Q(D, Rz @ x), Rz @ Q(D, x) @ Rz.T)


# ------------------------------------------------------------------ sections 4(c), 6(d) and S7
def test_section4_projection_on_the_structure_tensor():
    """Section 4(c): |H6|^2 = 32, so the orthogonal projection of D on H6 is (Delta/8) H6."""
    _h, _k, H6 = hexagonal.structure_tensors()
    H6 = np.asarray(H6, dtype=float)
    assert abs(np.sum(H6 * H6) - 32) < 1e-12
    alg = hexagonal.algebra()
    rng = np.random.default_rng(73)
    bl = {p: (lambda A: A @ A.T)(rng.normal(size=(len(c), len(c)))) for p, c, _d in hexagonal.IRREPS}
    D = alg.assemble(bl)
    delta = hexagonal.extra_modulus(bl)
    assert abs(np.sum(D * H6) / np.sum(H6 * H6) - delta / 8) < 1e-12


def test_section6_granular_extra_modulus_as_printed():
    """eq. (hex-granular): Delta = (1440/13)(k_n - k_s) a66 rho_c l^4, printed as such, for
    several stiffnesses and axial fabrics (the granular tensor of Pirmoradi et al. 2024)."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("hexagonal_granular", root / "analysis" / "hexagonal_granular.py")
    hg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hg)
    text = (root / "paper" / "sections" / "06_averages.tex").read_text()
    assert r"\Dhex=\tfrac{1440}{13}\,(k_n-k_s)\,a_{66}\,\rho_cl^4" in text
    for kn, ks, fab in ((1.0, 0.5, {}), (2.0, 0.3, {"a20": 0.2}), (1.0, 0.8, {"a20": -0.1, "a40": 0.05, "a60": 0.02})):
        a66 = 0.5 * hg.max_a66(**fab)
        b, dist = hg.granular_blocks(kn, ks, **fab, a66=a66)
        assert dist < 1e-14
        assert abs(hexagonal.extra_modulus(b) - 1440 / 13 * (kn - ks) * a66) < 1e-12


def test_appendix_B_six_class_coefficients_of_the_general_average():
    """Appendix B, eq. (esh-six-general): the six class coefficients of the exact average for a
    general isotropic medium (two lengths), against the block route of the package."""
    from gradinv import eshelby as es
    from test_eshelby import N27, coords6
    rng = np.random.default_rng(7303)
    for _ in range(4):
        lam, mu = rng.uniform(0.5, 2.0, 2)
        a1, a2, a3, a4, a5 = rng.uniform(-0.2, 0.2, 5) + np.array([0.3, 0.05, 0.1, 0.6, 0.2])
        R_ = rng.uniform(0.5, 4.0)
        l1s, l2s = es.lengths_squared(lam, mu, (a1, a2, a3, a4, a5))
        F1, F2 = es.radial_factor(np.sqrt(l1s), R_), es.radial_factor(np.sqrt(l2s), R_)
        c = lam + 2 * mu
        printed = np.array([
            (7 * a2 + 14 * a3 + 2 * a4 + 2 * a5) * F1 / (105 * c) + 2 * (28 * a3 + 6 * a4 - a5) * F2 / (105 * mu),
            (14 * a1 + 7 * a2 + 4 * a4 + 4 * a5) * F1 / (210 * c) + (14 * a2 - 2 * a4 + 5 * a5) * F2 / (105 * mu),
            (7 * a2 + 14 * a3 + 2 * a4 + 2 * a5) * F1 / (105 * c) - 2 * (7 * a3 + a4 + a5) * F2 / (105 * mu),
            2 * (a4 + a5) * F1 / (105 * c) - (2 * a4 - 5 * a5) * F2 / (105 * mu),
            2 * (a4 + a5) * F1 / (105 * c) + 2 * (6 * a4 - a5) * F2 / (105 * mu),
            (14 * a1 + 7 * a2 + 4 * a4 + 4 * a5) * F1 / (210 * c) - (7 * a2 + 4 * a4 + 4 * a5) * F2 / (210 * mu)])
        coef, res = coords6(N27(es.sphere_N_isotropic(lam, mu, (a1, a2, a3, a4, a5), R_)))
        assert res < 1e-13 and np.allclose(coef, printed, rtol=1e-12, atol=1e-14)
