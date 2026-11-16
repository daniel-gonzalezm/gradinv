"""Neighborhood radius of Admal, Marian and Po (2017, eqs. 3.10-3.11) for the
cubic datasets.

(R^2/5) C_ijkl delta_mn + D_ijmkln must be positive definite on V. The package
computes the smallest such R block by block (lengths.neighborhood_radius). Here
it is recomputed without the algebra: C (x) delta is built from its indices, D
from Lazar et al.'s (2022) constants, and positive definiteness is tested on an
18-dimensional basis of V just below and just above the radius. The radii and
their ratio to the nearest-neighbour distance b = a sqrt(3)/2 (bcc) are macros.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sl

from gradinv import cubic
from gradinv import lengths as L
from helpers import basis_V

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())["datasets"]
LZ = DATA["lazar2022"]
LATTICE = DATA["lazarpo2024"]["lattice_a_table2"]
NUMBERS = dict(re.findall(r"\\newcommand\{\\(Num[A-Za-z]*)\}\{([^}]*)\}",
                          (ROOT / "paper" / "paper_numbers.tex").read_text()))


def c_tensor(C11, C12, C44):
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    v = C12 * (i == j) * (k == l) + C44 * ((i == k) * (j == l) + (i == l) * (j == k))
                    if i == j == k == l:
                        v = C11
                    C[i, j, k, l] = v
    return C


def c_delta(C11, C12, C44):
    """27 x 27 storage of C_ijkl delta_mn, from the indices."""
    C = c_tensor(C11, C12, C44)
    G = np.zeros((27, 27))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                for k in range(3):
                    for l in range(3):
                        G[9 * i + 3 * j + m, 9 * k + 3 * l + m] = C[i, j, k, l]
    return G


def min_eig_on_V(M):
    Bv = basis_V()
    return np.linalg.eigvalsh(Bv.T @ (0.5 * (M + M.T)) @ Bv)[0]


def radius(k):
    b = cubic.from_lazar2022(cubic.a_from_voigt(LZ["D_voigt"][k]))
    return L.neighborhood_radius(b, L.helmholtz_cubic(*LZ["C"][k]))


@pytest.mark.parametrize("k", ["Fe", "W"])
def test_radius_by_brute_force(k):
    R, p = radius(k)
    assert p == "T2" and R > 0
    D = cubic.lazar2022_tensor(cubic.a_from_voigt(LZ["D_voigt"][k]))
    G = c_delta(*LZ["C"][k])
    assert min_eig_on_V(D) < 0
    assert min_eig_on_V((R * (1 - 1e-6)) ** 2 / 5 * G + D) < 0
    assert min_eig_on_V((R * (1 + 1e-6)) ** 2 / 5 * G + D) > 0
    # same radius from the full 18 x 18 generalized eigenproblem
    Bv = basis_V()
    lam = sl.eigh(Bv.T @ D @ Bv, Bv.T @ G @ Bv, eigvals_only=True)[0]
    assert np.sqrt(-5 * lam) == pytest.approx(R, rel=1e-10)


@pytest.mark.parametrize("k", ["Al", "Cu"])
def test_positive_definite_sets_need_no_radius(k):
    R, p = radius(k)
    assert R == 0.0 and p is None


def test_isotropic_limit():
    """A Helmholtz tensor l^2 C (x) delta with l^2 < 0 needs exactly R^2 = -5 l^2."""
    C = (1.2, 0.7, 0.4)
    ref = L.helmholtz_cubic(*C)
    D = L.helmholtz_cubic(*C, l2=-0.3)
    R, _p = L.neighborhood_radius(D, ref)
    assert R == pytest.approx(np.sqrt(1.5), rel=1e-12)


def test_reference_from_indices_matches_package():
    for k in ("Fe", "W"):
        alg = cubic.algebra()
        G_pkg = alg.assemble(L.helmholtz_cubic(*LZ["C"][k]))
        assert np.abs(G_pkg - c_delta(*LZ["C"][k])).max() < 1e-12


@pytest.mark.parametrize("k", ["Fe", "W"])
def test_macros(k):
    R, _p = radius(k)
    b = LATTICE[k] * np.sqrt(3) / 2
    assert NUMBERS[f"Num{k}NeighborhoodRadius"] == f"{R:.2f}"
    assert NUMBERS[f"Num{k}NeighborhoodRatio"] == f"{R / b:.2f}"
