"""Shared helpers for the tests."""
import numpy as np
import sympy as sp


def _entry_is_zero(e):
    if e == 0:
        return True
    z = sp.cancel(sp.radsimp(e))
    if z == 0:
        return True
    return sp.simplify(z) == 0


def is_zero(M):
    """Exact test that a sympy matrix (or expression) vanishes.

    Entry by entry: cancellation after rationalising the
    denominators, and sympy's simplify only as a last resort.
    """
    if not isinstance(M, sp.MatrixBase):
        return _entry_is_zero(sp.sympify(M))
    return all(_entry_is_zero(e) for e in M)


def random_rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rot_z(t):
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def basis_V():
    """27 x 18 orthonormal basis of V (range of the identity on V)."""
    from gradinv.skeleton import identity_V
    w, U = np.linalg.eigh(identity_V())
    return U[:, w > 0.5]


def commutant_dimension(group, symmetric=False):
    """Dimension of {X on V : X g = g X for every g} (optionally symmetric X)."""
    from gradinv._tensor import rotation_27
    Bv = basis_V()
    reps = [Bv.T @ rotation_27(Q) @ Bv for Q in group]
    if symmetric:
        idx = [(p, q) for p in range(18) for q in range(p, 18)]
        Em = np.zeros((324, len(idx)))
        for c, (p, q) in enumerate(idx):
            M = np.zeros((18, 18))
            M[p, q] = M[q, p] = 1.0
            Em[:, c] = M.ravel()
    else:
        Em = np.eye(324)
    A = np.vstack([(np.kron(Rg, np.eye(18)) - np.kron(np.eye(18), Rg.T)) @ Em for Rg in reps])
    return Em.shape[1] - np.linalg.matrix_rank(A, tol=1e-9)
