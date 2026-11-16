"""Isotropic limit against the sources.

Lazar et al. (2022) a1..a5; Auffray (2015) Tables 2 and 3; Lazar (2016)
pieces and b1..b5; Monchiet and Bonnet (2011a) projectors and inverse;
invariance of the block invariants under the choice of the H1 pair.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from gradinv import cubic, isotropic
from gradinv._tensor import R, TRIPLES, index
from gradinv.index_forms import monchiet_bonnet_2011a_tensors
from gradinv.skeleton import THETA_LAZAR, embeddings, h1_pair, identity_V
from helpers import is_zero
from references.isotropic_sources import (AUFFRAY2015_P581_ORDER, AUFFRAY2015_P581_ROT,
                                          AUFFRAY2015_P581_STR, AUFFRAY2015_TABLE2_CORRECTED,
                                          AUFFRAY2015_TABLE2_PRINTED, MB2011A_TABLE2,
                                          auffray2015_table3, lazar2016_b, lazar2016_embeddings,
                                          monchiet_bonnet_JK)

DATA = json.loads((Path(__file__).resolve().parents[1] / "analysis" / "data"
                   / "cubic_gradient_constants.json").read_text())
A = sp.symbols("a1:6")
COS, SIN = sp.sqrt(5) / 3, R(2, 3)


def _exact_blocks():
    return isotropic.algebra(exact=True).moduli(isotropic.lazar2022_tensor(A, exact=True))


# ------------------------------------------------------------------ E Lazar et al. (2022)
CLOSED_FORMS = {  # moduli in Lazar's a1..a5
    "m3": 2 * (A[3] + A[4]), "m2": 2 * A[3] - A[4],
    "Mss": R(10, 3) * (A[0] + A[1] + A[2]) + 2 * A[3] + 2 * A[4],
    "Msr": sp.sqrt(5) / 3 * (A[0] + 4 * A[1] - 2 * A[2]),
    "Mrr": -R(4, 3) * A[0] + R(8, 3) * A[1] + R(2, 3) * A[2] + 2 * A[3] - A[4]}


@pytest.mark.exact
def test_lazar2022_constants_to_moduli_exact():
    D = isotropic.lazar2022_tensor(A, exact=True)
    Ix = identity_V(exact=True)
    assert is_zero(D - D.T) and is_zero(Ix * D * Ix - D)
    alg = isotropic.algebra(exact=True)
    blk = alg.moduli(D)
    assert is_zero(alg.assemble(blk) - D)
    v = isotropic.blocks_to_vector(blk)
    for (name, *_), x in zip(isotropic.MODULI, v):
        assert is_zero(x - CLOSED_FORMS[name]), name
    assert is_zero(blk["H1"][0, 1] - blk["H1"][1, 0])
    J = sp.Matrix(v).jacobian(A)
    assert sp.nsimplify(J.det()) == -120 * sp.sqrt(5)
    assert is_zero(sp.Matrix(isotropic.to_lazar2022(blk, exact=True)) - sp.Matrix(A))


def test_lazar2022_constants_to_moduli_numeric():
    rng = np.random.default_rng(1)
    a = rng.normal(size=5)
    blk = isotropic.from_lazar2022(a)
    f = sp.lambdify(A, [CLOSED_FORMS[n] for n, *_ in isotropic.MODULI])
    assert np.allclose(isotropic.blocks_to_vector(blk), f(*a))
    assert np.allclose(isotropic.algebra().moduli(isotropic.lazar2022_tensor(a))["H1"], blk["H1"])
    assert np.allclose(isotropic.to_lazar2022(blk), a)
    # m3, m2 are the A2 and E eigenvalues of Lazar et al. (2022): 2(a4 + a5), 2 a4 - a5
    assert np.isclose(blk["H3"][0, 0], 2 * (a[3] + a[4])) and np.isclose(blk["H2"][0, 0], 2 * a[3] - a[4])


# ------------------------------------------------------------------ D, F Auffray (2015)
def _auffray_names(U):
    return {"Pss": U[("H1", "s", "s")], "Qsr": U[("H1", "s", "r")], "Qrs": U[("H1", "r", "s")],
            "Prr": U[("H1", "r", "r")]}


def test_auffray2015_table2_printed_fails_corrected_holds():
    """Table 2 as printed is not an inverse pair; with the misprint corrected it is."""
    B = _auffray_names(isotropic.algebra().units())
    printed = all(np.allclose(B[a] @ B[b], B[c]) for (a, b), c in AUFFRAY2015_TABLE2_PRINTED.items())
    corrected = all(np.allclose(B[a] @ B[b], B[c]) for (a, b), c in AUFFRAY2015_TABLE2_CORRECTED.items())
    assert not printed and corrected


@pytest.mark.exact
def test_auffray2015_table3_exact():
    """Table 3 holds for ms3, ms1, mr2, mr1; mc1 = -Msr."""
    D = isotropic.lazar2022_tensor(A, exact=True)
    blk = _exact_blocks()
    ms3, ms1, mr2, mr1, mc1 = auffray2015_table3(D)
    assert is_zero(ms3 - blk["H3"][0, 0]) and is_zero(ms1 - blk["H1"][0, 0])
    assert is_zero(mr2 - blk["H2"][0, 0]) and is_zero(mr1 - blk["H1"][1, 1])
    assert is_zero(mc1 + blk["H1"][0, 1]) and not is_zero(mc1 - blk["H1"][0, 1])


def test_auffray2015_p581_basis_vectors():
    """His H1,rot vector on p. 581 is -E_r e1 (opposite to his p. 572 embedding); H1,str is +E_s e1."""
    E = embeddings(exact=True)
    scale = lambda t: 1 if t[0] == t[1] else sp.sqrt(2)
    ours_r = [E["r"][index(*t), 0] * scale(t) for t in AUFFRAY2015_P581_ORDER]
    ours_s = [E["s"][index(*t), 0] * scale(t) for t in AUFFRAY2015_P581_ORDER]
    assert all(sp.simplify(o + p) == 0 for o, p in zip(ours_r, AUFFRAY2015_P581_ROT))
    assert all(sp.simplify(o - p) == 0 for o, p in zip(ours_s, AUFFRAY2015_P581_STR))


# ------------------------------------------------------------------ G Lazar (2016)
def test_lazar2016_pieces():
    E = embeddings(exact=True)
    E3L, E4 = lazar2016_embeddings()
    assert is_zero(E4 - (COS * E["s"] + SIN * E["r"]))
    assert is_zero(E3L - (SIN * E["s"] - COS * E["r"]))          # det of the change -1
    Ea, Eb = h1_pair(THETA_LAZAR, exact=True)
    assert is_zero(Ea - E4) and is_zero(Eb + E3L)
    assert is_zero(E3L.T * E3L - sp.eye(3)) and is_zero(E4.T * E4 - sp.eye(3)) and is_zero(E3L.T * E4)
    Ef = [np.array(x, dtype=float) for x in h1_pair(float(THETA_LAZAR.evalf()))]
    assert np.allclose(Ef[0], np.array(E4, dtype=float)) and np.allclose(Ef[1], -np.array(E3L, dtype=float))
    # cross term (3)eta_ijk (4)eta_kji = (sqrt5/3) u.c for unit pieces
    u, c = sp.symbols("u0:3"), sp.symbols("w0:3")
    e3v, e4v = E3L * sp.Matrix(u), E4 * sp.Matrix(c)
    cross = sum(e3v[index(i, j, k)] * e4v[index(k, j, i)] for (i, j, k) in TRIPLES)
    assert sp.simplify(cross - COS * sum(u[t] * c[t] for t in range(3))) == 0


@pytest.mark.exact
def test_lazar2016_moduli_exact():
    blk = _exact_blocks()
    b = lazar2016_b(A)
    C2 = sp.Matrix([[SIN, COS], [-COS, SIN]])   # columns: E_(3), E_(4) in (E_s, E_r)
    ML = C2.T * blk["H1"] * C2
    assert is_zero(b[0] - blk["H3"][0, 0]) and is_zero(b[1] - blk["H2"][0, 0])
    assert is_zero(ML - sp.Matrix([[b[2], COS * b[4]], [COS * b[4], b[3]]]))
    # block invariants do not depend on the pair (rotation phi or Lazar's reflection)
    phi = sp.symbols("phi", real=True)
    Rp = sp.Matrix([[sp.cos(phi), -sp.sin(phi)], [sp.sin(phi), sp.cos(phi)]])
    M = blk["H1"]
    Mp = Rp.T * M * Rp
    assert sp.simplify(Mp.trace() - M.trace()) == 0 and sp.simplify(Mp.det() - M.det()) == 0
    assert is_zero(ML.trace() - M.trace()) and is_zero(ML.det() - M.det())


# ------------------------------------------------------------------ H Monchiet and Bonnet (2011a)
@pytest.fixture(scope="module")
def mb():
    T = monchiet_bonnet_2011a_tensors()
    return dict(zip(("J1", "J2", "J3", "J4", "K1", "K2"), monchiet_bonnet_JK(T)))


def test_monchiet_bonnet_2011a_projectors_and_table(mb):
    U = isotropic.algebra(exact=True).units()
    assert is_zero(mb["K1"] - U[("H3", "3", "3")]) and is_zero(mb["K2"] - U[("H2", "2", "2")])
    for a, b in itertools.product(("J1", "J2", "J3", "J4"), repeat=2):
        target = mb[MB2011A_TABLE2[(a, b)]] if (a, b) in MB2011A_TABLE2 else sp.zeros(27, 27)
        assert is_zero(mb[a] * mb[b] - target), (a, b)
    assert not is_zero(mb["J1"] - mb["J1"].T) or not is_zero(mb["J4"] - mb["J4"].T)  # oblique pair


@pytest.mark.core
@pytest.mark.exact
def test_monchiet_bonnet_2011a_inverse_exact(mb):
    """Their eq. (2.13) with one misprint corrected is our inverse; as printed it is not an inverse."""
    D = isotropic.lazar2022_tensor(A, exact=True)
    al = sp.symbols("al1:7")
    Dmb = al[0] * mb["K1"] + al[1] * mb["K2"] + al[2] * mb["J1"] + al[3] * mb["J2"] + al[4] * mb["J3"] \
        + al[5] * mb["J4"]
    sol = sp.solve([e for e in (Dmb - D) if e != 0], al, dict=True)[0]
    A1, A2, A3, A4, A5, A6 = (sol[k] for k in al)
    DJ = A3 * A6 - A4 * A5
    fixed = mb["K1"] / A1 + mb["K2"] / A2 + (A6 * mb["J1"] - A4 * mb["J2"] - A5 * mb["J3"] + A3 * mb["J4"]) / DJ
    printed = mb["K1"] / A1 + mb["K2"] / A2 + (A6 * mb["J1"] - A4 * mb["J2"] + A5 * mb["J3"] - A3 * mb["J4"]) / DJ
    alg = isotropic.algebra(exact=True)
    ours = alg.assemble(alg.inverse(alg.moduli(D)))
    assert is_zero(fixed - ours)
    # printed version: test numerically at a generic point (faster, same conclusion)
    pt = dict(zip(A, (R(1, 3), R(2, 7), R(-1, 5), R(5, 4), R(1, 6))))
    assert not is_zero((printed * D).subs(pt) - identity_V(exact=True))


# ------------------------------------------------------------------ the H1 pair of Lazar (2016)
def test_h1_pair_of_lazar2016_is_a_rotation_of_the_gl3_pair():
    E = embeddings()
    ga, gb = h1_pair(0)
    la, lb = h1_pair(float(THETA_LAZAR.evalf()))
    cos_t = abs(ga[:, 0] @ la[:, 0])
    assert np.isclose(cos_t, np.sqrt(5) / 3, atol=1e-14)
    G, L = np.hstack([ga, gb]), np.hstack([la, lb])
    assert np.allclose(G @ G.T, L @ L.T, atol=1e-14)            # same plane
    # separable C x delta (lambda = mu = 1): diag(3 lambda + 2 mu, 2 mu) in Lazar's pair
    D = isotropic.lazar2022_tensor([0, 0.5, 0, 1, 0])          # = C x delta with lambda = mu = 1
    block = lambda Ea, Eb: np.array([[np.trace(X.T @ D @ Y) / 3 for Y in (Ea, Eb)] for X in (Ea, Eb)])
    assert np.allclose(block(la, -lb), np.diag([5, 2]), atol=1e-13)
    assert np.allclose(block(ga, gb), [[11 / 3, 2 * np.sqrt(5) / 3], [2 * np.sqrt(5) / 3, 10 / 3]])
    # relative off-diagonal coupling of the Voigt H1 block in the pair of Lazar and in the GL(3) pair
    iso = isotropic.algebra()
    expected = {"Al": (0.16, 0.35), "Cu": (0.29, 0.34), "Cu_EAM": (0.80, 0.07)}
    for name, (lz, gl) in expected.items():
        v = (DATA["datasets"]["po2019"]["D_voigt"]["Cu_EAM"] if name == "Cu_EAM"
             else DATA["datasets"]["lazar2022"]["D_voigt"][name])
        M = iso.moduli(cubic.lazar2022_tensor(cubic.a_from_voigt(v)))["H1"]
        Rt = np.array([[np.sqrt(5) / 3, -2 / 3], [2 / 3, np.sqrt(5) / 3]])  # columns: la, lb in (s, r)
        ML = Rt.T @ M @ Rt
        rel = [abs(X[0, 1]) / np.sqrt(abs(X[0, 0] * X[1, 1])) for X in (ML, M)]
        assert np.allclose(rel, (lz, gl), atol=0.005), (name, rel)
        assert np.allclose(np.linalg.eigvalsh(ML), np.linalg.eigvalsh(M))
