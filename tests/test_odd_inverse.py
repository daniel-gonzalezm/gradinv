"""Block inverse of non-symmetric ("odd") elements of the cubic and hexagonal
algebras against the 18 x 18 inverse on V.

The algebra of a class is the set of all endomorphisms of V that commute with
the group, symmetric or not; its non-symmetric elements are the odd tensors of
the class (Poorsolhjouy 2026 counts them). The symbolic tests
test_inverse_of_general_element_exact (test_cubic.py, test_hexagonal.py) already
show S D = I_V for general blocks. Here, numerically and by a route that does
not use the algebra:

  * the random element is genuinely odd (D != D^T) and commutes with the group;
  * the inverse of its 18 x 18 restriction to V, lifted to 27 x 27, equals the
    sum of the inverse blocks, is a left and right inverse on V, and commutes
    with the group, so inversion closes inside the class of odd tensors;
  * (D^T)^-1 = (D^-1)^T block by block;
  * the same holds for a crystal in a rotated (general) orientation.
"""
import numpy as np
import pytest

from gradinv import cubic, hexagonal
from gradinv._tensor import rotation_27
from gradinv.algebra import BlockAlgebra as BA
from gradinv.skeleton import identity_V
from helpers import basis_V, random_rotation

CLASSES = {"cubic": (cubic, cubic.oh_group), "hexagonal": (hexagonal, hexagonal.d6_group)}
IV = identity_V()


def _odd_blocks(mod, rng):
    """Random non-symmetric blocks, well conditioned (identity plus a random matrix)."""
    out = {}
    for p, c, _d in mod.IRREPS:
        n = len(c)
        A = rng.normal(size=(n, n))
        if n == 1:
            out[p] = np.array([[1.5 + abs(A[0, 0])]])
        else:
            out[p] = 2 * np.eye(n) + A          # non-symmetric for n > 1
    return out


def _check(alg, group, X):
    Bv = basis_V()
    D = alg.assemble(X)
    rel = lambda M, ref: np.abs(M).max() / np.abs(ref).max()  # noqa: E731
    # the element is odd and belongs to the class
    assert rel(D - D.T, D) > 1e-2
    reps = [rotation_27(Q) for Q in group]
    assert max(rel(R @ D @ R.T - D, D) for R in reps) < 1e-13
    assert rel(alg.residual(D), D) < 1e-13
    # 18 x 18 route (no algebra)
    S18 = np.linalg.inv(Bv.T @ D @ Bv)
    S_lift = Bv @ S18 @ Bv.T
    S_blk = alg.assemble(BA.inverse(X))
    assert rel(S_blk - S_lift, S_lift) < 1e-12
    assert rel(S_blk @ D - IV, IV) < 1e-12 and rel(D @ S_blk - IV, IV) < 1e-12
    # the 18 x 18 inverse is again odd and in the class
    assert rel(S_lift - S_lift.T, S_lift) > 1e-3
    assert max(rel(R @ S_lift @ R.T - S_lift, S_lift) for R in reps) < 1e-12
    assert rel(alg.residual(S_lift), S_lift) < 1e-12
    # transpose: (D^T)^-1 = (D^-1)^T, block by block
    XT = {p: B.T for p, B in X.items()}
    assert rel(alg.assemble(BA.inverse(XT)) - S_blk.T, S_blk) < 1e-12
    assert rel(alg.assemble(XT) - D.T, D) < 1e-14


@pytest.mark.core
@pytest.mark.parametrize("name", sorted(CLASSES))
@pytest.mark.parametrize("seed", [5301, 5302, 5303])
def test_odd_block_inverse_equals_18x18_inverse(name, seed):
    mod, group = CLASSES[name]
    _check(mod.algebra(), group(), _odd_blocks(mod, np.random.default_rng(seed)))


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_odd_block_inverse_in_a_rotated_crystal(name):
    mod, group = CLASSES[name]
    rng = np.random.default_rng(5310)
    Q = random_rotation(rng)
    rotated_group = [Q @ g @ Q.T for g in group()]
    _check(mod.algebra(rotation=Q), rotated_group, _odd_blocks(mod, rng))


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_singular_odd_element_is_detected_by_one_block(name):
    """Invertibility is block-wise for odd elements too: a singular block makes the
    18 x 18 restriction singular, with the block's dimension as the rank deficiency."""
    mod, _group = CLASSES[name]
    alg = mod.algebra()
    X = _odd_blocks(mod, np.random.default_rng(5320))
    p, c, d = [r for r in mod.IRREPS if len(r[1]) > 1][0]
    u, s, vt = np.linalg.svd(X[p])
    s[-1] = 0.0
    X[p] = u @ np.diag(s) @ vt                     # rank n_p - 1, still non-symmetric
    assert not BA.is_invertible(X, tol=1e-12)
    Bv = basis_V()
    sv = np.linalg.svd(Bv.T @ alg.assemble(X) @ Bv, compute_uv=False)
    assert int(np.sum(sv < 1e-10 * sv.max())) == d
