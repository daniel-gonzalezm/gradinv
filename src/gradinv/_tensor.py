"""Storage conventions for sixth-order tensors on the strain-gradient space V.

Notation: Mindlin form II,
eta_ijm = d_m eps_ij (symmetric in i, j), so V has dimension 18. A sixth-order
tensor D_ijmkln is stored as a 27 x 27 matrix whose row index is
(i, j, m) -> 9 i + 3 j + m and whose column index is (k, l, n). With this
storage tau = D eta is a matrix-vector product, composition (contraction over
three indices) is a matrix product and the Frobenius product of two tensors is
the Euclidean product of their 729 components. No Voigt factors appear.

Every builder exists in an exact version (sympy) and a numerical one (numpy).
The numerical versions are the default public interface.
"""
import itertools

import numpy as np
import sympy as sp

N = 27
TRIPLES = list(itertools.product(range(3), repeat=3))
R = sp.Rational


def index(i, j, m):
    """Row/column index of the component (i, j, m)."""
    return 9 * i + 3 * j + m


def kd(a, b):
    """Kronecker delta on integer indices."""
    return 1 if a == b else 0


def tensor6(func, exact=True):
    """27 x 27 matrix with entries func(i, j, m, k, l, n)."""
    if exact:
        M = sp.zeros(N, N)
        for (i, j, m) in TRIPLES:
            for (k, l, n) in TRIPLES:
                v = func(i, j, m, k, l, n)
                if v != 0:
                    M[index(i, j, m), index(k, l, n)] = v
        return M
    M = np.zeros((N, N))
    for (i, j, m) in TRIPLES:
        for (k, l, n) in TRIPLES:
            M[index(i, j, m), index(k, l, n)] = func(i, j, m, k, l, n)
    return M


def to_float(M):
    """Float numpy array of a sympy matrix (or anything array-like)."""
    if isinstance(M, sp.MatrixBase):
        return np.array(M.evalf(), dtype=float)
    return np.asarray(M, dtype=float)


def rotation_27(Q):
    """Action Q x Q x Q of an orthogonal 3 x 3 matrix on the 27 index triples.

    A tensor D transforms as rotation_27(Q) @ D @ rotation_27(Q).T.
    """
    if isinstance(Q, sp.MatrixBase):
        return sp.kronecker_product(sp.kronecker_product(Q, Q), Q)
    Q = np.asarray(Q, dtype=float)
    return np.kron(np.kron(Q, Q), Q)


def frobenius(A, B):
    """Frobenius product <A, B> = A_ijmkln B_ijmkln."""
    if isinstance(A, sp.MatrixBase) or isinstance(B, sp.MatrixBase):
        return sum(A[t] * B[t] for t in range(A.shape[0] * A.shape[1]))
    return float(np.sum(np.asarray(A) * np.asarray(B)))
