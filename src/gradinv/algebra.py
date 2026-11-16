"""Block algebras of sixth-order tensors on V.

A symmetry class with point group G splits V into isotypic components. Inside
each irreducible representation p of dimension d_p, V contains n_p copies, each
given by an isometric embedding E_a (27 x d_p) that carries the same matrices
of G as the other copies. The commutant of G on V (the algebra of the class)
has the matrix units

    U^p_ab = E_a E_b^T,   U^p_ab U^q_cd = d_pq d_bc U^p_ad,
    <U^p_ab, U^q_cd> = d_p d_pq d_ac d_bd,

so an element is a set of n_p x n_p blocks X^p with X = sum X^p_ab U^p_ab and
X^p_ab = <X, U^p_ab>/d_p = tr(E_a^T X E_b)/d_p. Composition, inversion and
positive definiteness reduce to the blocks; elements with major symmetry have
symmetric blocks. For any X (not necessarily in the algebra) the same formula
gives its orthogonal (Frobenius) projection on the algebra.

`BlockAlgebra` holds the embeddings either as sympy matrices (exact=True) or
as float arrays (exact=False); the operations work on both.
"""
import numpy as np
import sympy as sp

from ._tensor import N, rotation_27, to_float

__all__ = ["BlockAlgebra"]


class BlockAlgebra:
    """Algebra of a symmetry class on V, described by its irreps and embeddings.

    Parameters
    ----------
    name : str
    irreps : tuple of (label, copies, d) with copies a tuple of copy labels.
    embeddings : dict {label: {copy: 27 x d matrix}}.
    exact : bool, whether the embeddings are sympy matrices.
    """

    def __init__(self, name, irreps, embeddings, exact):
        self.name = name
        self.irreps = tuple((p, tuple(c), int(d)) for p, c, d in irreps)
        self.exact = bool(exact)
        self._emb = {p: dict(embeddings[p]) for p, _c, _d in self.irreps}
        self._units = None

    # ------------------------------------------------------------ structure
    def __repr__(self):
        return (f"BlockAlgebra({self.name!r}, dimension {self.dimension}, "
                f"{'exact' if self.exact else 'float'})")

    @property
    def labels(self):
        return tuple(p for p, _c, _d in self.irreps)

    def copies(self, p):
        return dict((q, c) for q, c, _d in self.irreps)[p]

    def irrep_dimension(self, p):
        return dict((q, d) for q, _c, d in self.irreps)[p]

    @property
    def dimension(self):
        """Dimension of the algebra, sum of n_p^2."""
        return sum(len(c) ** 2 for _p, c, _d in self.irreps)

    @property
    def symmetric_dimension(self):
        """Number of independent elements with major symmetry, sum of n_p (n_p + 1)/2."""
        return sum(len(c) * (len(c) + 1) // 2 for _p, c, _d in self.irreps)

    def embedding(self, p, a):
        """Isometric embedding (27 x d_p) of copy a of irrep p."""
        return self._emb[p][a]

    def embedding_matrix(self):
        """27 x 18 matrix of all embedding columns (irreps and copies in order).

        Its columns are an orthonormal basis of V adapted to the class.
        """
        cols = [self._emb[p][a] for p, copies, _d in self.irreps for a in copies]
        if self.exact:
            return sp.Matrix.hstack(*cols)
        return np.hstack(cols)

    def units(self):
        """Matrix units {(p, a, b): U^p_ab = E_a E_b^T} (27 x 27)."""
        if self._units is None:
            U = {}
            for p, copies, _d in self.irreps:
                for a in copies:
                    for b in copies:
                        Ea, Eb = self._emb[p][a], self._emb[p][b]
                        U[(p, a, b)] = Ea * Eb.T if self.exact else Ea @ Eb.T
            self._units = U
        return self._units

    def rotated(self, Q):
        """The same algebra for the crystal rotated by Q (embeddings Q x Q x Q E)."""
        RQ = rotation_27(Q)
        if self.exact:
            RQ = sp.Matrix(RQ)
            emb = {p: {a: RQ * E for a, E in cs.items()} for p, cs in self._emb.items()}
        else:
            RQ = to_float(RQ)
            emb = {p: {a: RQ @ E for a, E in cs.items()} for p, cs in self._emb.items()}
        return BlockAlgebra(self.name, self.irreps, emb, self.exact)

    # ------------------------------------------------------------ elements
    def moduli(self, D, expand=True):
        """Blocks {p: n_p x n_p matrix} of D, X^p_ab = tr(E_a^T D E_b)/d_p.

        For D outside the algebra this is its orthogonal projection on it.
        """
        out = {}
        if self.exact:
            D = sp.Matrix(D)
            f = sp.expand if expand else (lambda x: x)
            for p, copies, d in self.irreps:
                n = len(copies)
                DE = {b: D * self._emb[p][b] for b in copies}
                out[p] = sp.Matrix(n, n, lambda i, j: f(
                    (self._emb[p][copies[i]].T * DE[copies[j]]).trace() / d))
            return out
        D = np.asarray(D, dtype=float)
        for p, copies, d in self.irreps:
            n = len(copies)
            DE = [D @ self._emb[p][b] for b in copies]
            out[p] = np.array([[np.sum(self._emb[p][a] * DE[j]) / d for j in range(n)]
                               for a in copies])
        return out

    def assemble(self, blocks):
        """X = sum_p sum_ab X^p_ab U^p_ab (27 x 27)."""
        U = self.units()
        out = sp.zeros(N, N) if self.exact else np.zeros((N, N))
        for p, copies, _d in self.irreps:
            B = blocks[p]
            for i, a in enumerate(copies):
                for j, b in enumerate(copies):
                    if B[i, j] != 0:
                        out += B[i, j] * U[(p, a, b)]
        return out

    def residual(self, D):
        """D minus its projection on the algebra (zero iff D belongs to it)."""
        return D - self.assemble(self.moduli(D))

    @staticmethod
    def multiply(A, B):
        """Product of two elements given by their blocks: (AB)^p = A^p B^p."""
        return {p: A[p] * B[p] if isinstance(A[p], sp.MatrixBase) else A[p] @ B[p] for p in A}

    @staticmethod
    def inverse(blocks):
        """Blocks of the inverse: (X^-1)^p = (X^p)^-1."""
        out = {}
        for p, B in blocks.items():
            out[p] = B.inv() if isinstance(B, sp.MatrixBase) else np.linalg.inv(B)
        return out

    def invert(self, D):
        """Inverse on V of an invertible element D of the algebra (27 x 27)."""
        return self.assemble(self.inverse(self.moduli(D)))

    # ------------------------------------------------------------ conditions
    @staticmethod
    def determinants(blocks):
        """{p: det X^p}; X is invertible on V iff every determinant is non-zero."""
        return {p: (B.det() if isinstance(B, sp.MatrixBase) else float(np.linalg.det(B)))
                for p, B in blocks.items()}

    @staticmethod
    def is_invertible(blocks, tol=0.0):
        return all(abs(complex(v)) > tol for v in BlockAlgebra.determinants(blocks).values())

    @staticmethod
    def non_positive_blocks(blocks, tol=0.0):
        """Labels of the symmetric blocks that are not positive definite."""
        out = []
        for p, B in blocks.items():
            Bf = to_float(B)
            if np.linalg.eigvalsh(0.5 * (Bf + Bf.T)).min() <= tol:
                out.append(p)
        return out

    @staticmethod
    def is_positive_definite(blocks, tol=0.0):
        """A symmetric X is positive definite on V iff every block is."""
        return not BlockAlgebra.non_positive_blocks(blocks, tol)

    def spectrum(self, blocks):
        """Eigenvalues of a symmetric element on V (18 values, ascending).

        Each eigenvalue of X^p appears d_p times.
        """
        vals = []
        for p, _copies, d in self.irreps:
            Bf = to_float(blocks[p])
            vals.extend(np.repeat(np.linalg.eigvalsh(0.5 * (Bf + Bf.T)), d))
        return np.sort(np.array(vals))
