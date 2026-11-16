"""The single hexagonal (D6) modulus absent from transverse isotropy (D_inf).

Auffray, Le Quang & He (2013, Sect. 3.3.5) write A_D6 = A_O(2) + eta*A_c in matrix
form. This test identifies that extra direction intrinsically: it is a rank-2,
traceless operator supported on the totally symmetric traceless part H3 of V,
on which the 60-degree rotation acts as -1 (the m = +-3 harmonics), i.e. the
difference between the B1 and B2 block moduli. Self-contained (no gradinv).
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import null_space

# orthonormal basis of V = {eta_ijk = eta_jik}, dim 18
_B = []
for i in range(3):
    for j in range(i, 3):
        for k in range(3):
            T = np.zeros((3, 3, 3))
            if i == j:
                T[i, j, k] = 1.0
            else:
                T[i, j, k] = T[j, i, k] = 1 / np.sqrt(2)
            _B.append(T.ravel())
B = np.array(_B)


def rep(Q):
    R = np.einsum("ia,jb,kc->ijkabc", Q, Q, Q).reshape(27, 27)
    return B @ R @ B.T


def rot(axis, th):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def sym_commutant(gens):
    idx = [(p, q) for p in range(18) for q in range(p, 18)]
    E = np.zeros((324, len(idx)))
    for n, (p, q) in enumerate(idx):
        M = np.zeros((18, 18))
        M[p, q] = M[q, p] = 1.0
        E[:, n] = M.ravel()
    A = np.vstack([np.kron(rep(Q), rep(Q)) @ E - E for Q in gens])
    return E @ null_space(A)  # columns: vec(M)


def h3_projector():
    P = np.zeros((27, 27))
    for col in range(27):
        T = np.zeros(27)
        T[col] = 1.0
        T = T.reshape(3, 3, 3)
        S = sum(T.transpose(p) for p in itertools.permutations(range(3))) / 6
        tr = np.einsum("iik->k", S)
        d = np.eye(3)
        S = S - (np.einsum("ij,k->ijk", d, tr) + np.einsum("ik,j->ijk", d, tr)
                 + np.einsum("jk,i->ijk", d, tr)) / 5
        P[:, col] = S.ravel()
    return B @ P @ B.T


C6 = rot([0, 0, 1], np.pi / 3)
C2x = rot([1, 0, 0], np.pi)
Cgen = rot([0, 0, 1], 0.7319)  # generic angle: generates SO(2) densely


def extra_direction():
    hexa = sym_commutant([C6, C2x])
    ti = sym_commutant([Cgen, C6, C2x])
    X = hexa - ti @ np.linalg.pinv(ti) @ hexa
    u, s, _ = np.linalg.svd(X)
    return hexa.shape[1], ti.shape[1], int(np.sum(s > 1e-8)), u[:, 0].reshape(18, 18)


def test_counts_and_single_extra_direction():
    n_hex, n_ti, n_extra, _ = extra_direction()
    assert (n_hex, n_ti, n_extra) == (22, 21, 1)


@pytest.mark.core
def test_extra_direction_is_the_m3_splitting_of_H3():
    *_, X = extra_direction()
    w, v = np.linalg.eigh(X)
    nz = np.abs(w) > 1e-8
    assert nz.sum() == 2 and np.isclose(w[nz].sum(), 0.0)   # traceless, rank 2
    support = v[:, nz]
    P = h3_projector()
    assert np.isclose(np.trace(P), 7)
    assert np.allclose(P @ support, support)                  # inside H3
    assert np.allclose(rep(C6) @ support, -support)           # m = +-3
