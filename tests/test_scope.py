"""Statements of sections 2 and 8 about the scope of the two classes.

Fifth-order coupling: of the cubic and hexagonal point groups, only the
centrosymmetric m-3m and 6/mmm force the coupling M: V -> Sym2 to vanish.

Site symmetries of two lattices with a basis (diamond, hcp): no reference point
gives both the full crystal form of D and an inversion centre.
    atom sites (-43m, -6m2)   -> 11 / 22 constants of D, no inversion
    inversion centres (-3m)   -> 34 constants of D (trigonal)
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gradinv._tensor import rotation_27
from helpers import basis_V, commutant_dimension

BV = basis_V()
INV = -np.eye(3)

# orthonormal basis of Sym2 (9 x 6)
_cols = []
for i in range(3):
    for j in range(i, 3):
        e = np.zeros((3, 3))
        e[i, j] = e[j, i] = 1.0
        _cols.append(e.ravel() / np.linalg.norm(e))
BS = np.array(_cols).T


def rot(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * angle).as_matrix()


def mirror(n):
    n = np.asarray(n, float)
    return np.eye(3) - 2 * np.outer(n, n) / (n @ n)


def closure(gens):
    G = [np.eye(3)]
    while True:
        new = []
        for x in (g @ h for g in G for h in gens):
            if not any(np.allclose(x, y) for y in G + new):
                new.append(x)
        if not new:
            return G
        G += new


# ------------------------------------------------------------------ fifth-order coupling
_O = [rot([0, 0, 1], np.pi / 2), rot([1, 1, 1], 2 * np.pi / 3)]
_D6 = [rot([0, 0, 1], np.pi / 3), rot([1, 0, 0], np.pi)]
GROUPS = {
    "432 (O)": (_O, 24, 3, 11),
    "-43m (Td)": ([rot([1, 1, 1], 2 * np.pi / 3), INV @ rot([0, 0, 1], np.pi / 2), mirror([1, -1, 0])], 24, 5, 11),
    "m-3m (Oh)": (_O + [INV], 48, 0, 11),
    "622 (D6)": (_D6, 12, 8, 22),
    "6mm (C6v)": ([rot([0, 0, 1], np.pi / 3), mirror([0, 1, 0])], 12, 12, 22),
    "-6m2 (D3h)": ([rot([0, 0, 1], 2 * np.pi / 3), mirror([0, 0, 1]), mirror([0, 1, 0])], 12, 8, 22),
    "6/mmm (D6h)": (_D6 + [INV], 24, 0, 22),
}


def repV(Q):
    return BV.T @ rotation_27(Q) @ BV


def repS(Q):
    return BS.T @ np.kron(Q, Q) @ BS


@pytest.mark.parametrize("name", list(GROUPS))
def test_fifth_order_coupling_invariants_per_point_group(name):
    """Invariant maps M: V -> Sym2; only the centrosymmetric m-3m and 6/mmm force M = 0
    (coherent with Rosi et al. 2024, Theorems 2.4-2.5, and Le Quang et al. 2021)."""
    gens, order, n_M, n_D = GROUPS[name]
    assert len(closure(gens)) == order
    rows = np.vstack([np.kron(repS(g), repV(g)) - np.eye(108) for g in gens])
    assert 108 - np.linalg.matrix_rank(rows, tol=1e-9) == n_M
    # constants of D: symmetric commutant of the generators on V
    idx = [(p, q) for p in range(18) for q in range(p, 18)]
    E = np.zeros((324, len(idx)))
    for c, (p, q) in enumerate(idx):
        M = np.zeros((18, 18))
        M[p, q] = M[q, p] = 1.0
        E[:, c] = M.ravel()
    A = np.vstack([(np.kron(repV(g), repV(g)) - np.eye(324)) @ E for g in gens])
    assert len(idx) - np.linalg.matrix_rank(A, tol=1e-9) == n_D


# ------------------------------------------------------------------ site symmetries
# name: (generators, order, constants of D, contains the inversion)
SITE_GROUPS = {
    "Oh (m-3m)": ([rot([0, 0, 1], np.pi / 2), rot([1, 1, 1], 2 * np.pi / 3), INV], 48, 11, True),
    "Td (-43m), diamond atom": ([rot([1, 1, 1], 2 * np.pi / 3), INV @ rot([0, 0, 1], np.pi / 2),
                                 mirror([1, -1, 0])], 24, 11, False),
    "D3d (-3m), diamond bond midpoint": ([rot([1, 1, 1], 2 * np.pi / 3), rot([1, -1, 0], np.pi), INV],
                                         12, 34, True),
    "D6h (6/mmm)": ([rot([0, 0, 1], np.pi / 3), rot([1, 0, 0], np.pi), INV], 24, 22, True),
    "D3h (-6m2), hcp atom": ([rot([0, 0, 1], 2 * np.pi / 3), mirror([0, 0, 1]), mirror([0, 1, 0])],
                             12, 22, False),
    "D3d (-3m), hcp inversion centre": ([rot([0, 0, 1], 2 * np.pi / 3), rot([1, 0, 0], np.pi), INV],
                                        12, 34, True),
}


@pytest.mark.parametrize("name", list(SITE_GROUPS))
def test_site_symmetry_constants(name):
    gens, order, n_D, has_inv = SITE_GROUPS[name]
    G = closure(gens)
    assert len(G) == order
    assert any(np.allclose(g, INV) for g in G) == has_inv
    assert commutant_dimension(gens, symmetric=True) == n_D
