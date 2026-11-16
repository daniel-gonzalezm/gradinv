"""Block lengths of a gradient tensor relative to the Helmholtz form.

The Helmholtz (separable) gradient tensor of form II is D = l^2 C (x) delta,
i.e. l^2 G_hat with G_hat = C (x) delta acting on V = Sym (x) R^3. For a
gradient tensor D and a classical stiffness C of the same symmetry class, the
block lengths l_p are defined by the generalized eigenvalues of the block pair

    D^p w = l^2 G_hat^p w      (p over the irreps of the class),

so that D is of Helmholtz form iff every block length equals the same l. They
are a diagnostic of how far D is from a one-parameter (single-length) model;
they are defined only if every block of D is positive definite.

Identity (Lazar and Po 2024, eq. 107): with S = C^-1,
    l_LP^2 = S_ijkl D_ijmklm / 18 = tr_V(G_hat^-1 D) / 18
           = sum_p d_p sum_eig (l_p^2) / 18,
the dimension-weighted mean of the squared block lengths (`lazar_po_length`).

Neighborhood radius (Admal, Marian and Po 2017, eqs. 3.10-3.11): the energy of
a constant strain gradient over a sphere of radius R is positive iff
(R^2/5) C (x) delta + D is positive definite on V; `neighborhood_radius` gives
the smallest such R, block by block (0 when D itself is positive definite).

Classical estimates of a cubic stiffness (for pairing with the gradient
estimates: Voigt with Voigt, Reuss with Reuss, Hill with Hill), so that a
Helmholtz grain l0^2 C_gr (x) delta has every block length equal to l0 in
every estimate, because <C (x) delta> = <C> (x) delta and
(C (x) delta)^-1 = C^-1 (x) delta.
"""
import numpy as np
import scipy.linalg as sl

from . import cubic, isotropic

__all__ = ["helmholtz_isotropic", "helmholtz_cubic", "cubic_classical_estimates",
           "block_lengths", "lazar_po_length", "neighborhood_radius"]


def helmholtz_isotropic(lam, mu, l2=1.0):
    """Isotropic blocks (form II) of l^2 C(lam, mu) (x) delta."""
    return isotropic.from_lazar2022([0, lam * l2 / 2, 0, mu * l2, 0])


def helmholtz_cubic(C11, C12, C44, l2=1.0):
    """Cubic blocks (form II) of l^2 C (x) delta for a cubic C (C11, C12, C44)."""
    a = np.array([0, C12 / 2, 0, C44, 0, 0, 0, C11 - C12 - 2 * C44, 0, 0, 0]) * l2
    return cubic.from_lazar2022(a)


def cubic_classical_estimates(C11, C12, C44):
    """Lame constants (lam, mu) of the Voigt, Reuss and Hill estimates of a cubic C.

    Bulk modulus k = (C11 + 2 C12)/3 in all three; shear moduli
    mu_V = (C11 - C12 + 3 C44)/5,  mu_R = 5 (C11 - C12) C44 / (4 C44 + 3 (C11 - C12)),
    mu_H = (mu_V + mu_R)/2.
    """
    k = (C11 + 2 * C12) / 3
    muV = (C11 - C12 + 3 * C44) / 5
    muR = 5 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    muH = (muV + muR) / 2
    return {name: (k - 2 * mu / 3, mu) for name, mu in (("voigt", muV), ("reuss", muR), ("hill", muH))}


def block_lengths(blocks, ref):
    """{p: ascending array of block lengths l_p} from D^p w = l^2 G_hat^p w.

    blocks: float blocks of D (symmetric); ref: blocks of G_hat (positive definite).
    A non-positive generalized eigenvalue gives nan (D not positive definite).
    """
    out = {}
    for p, B in blocks.items():
        B = np.asarray(B, float)
        G = np.asarray(ref[p], float)
        w = sl.eigh(0.5 * (B + B.T), 0.5 * (G + G.T), eigvals_only=True)
        with np.errstate(invalid="ignore"):
            out[p] = np.where(w > 0, np.sqrt(np.abs(w)), np.nan)
    return out


def lazar_po_length(alg, blocks, ref):
    """l_LP = sqrt(tr_V(G_hat^-1 D)/18) = dimension-weighted RMS of the block lengths.

    Equals Lazar and Po's (2024) eq. (107), l^2 = S_ijkl D_ijmklm / 18, with
    S = C^-1 the compliance of the stiffness that defines G_hat.
    """
    tot = 0.0
    for p, _copies, d in alg.irreps:
        tot += d * float(np.trace(np.linalg.solve(np.asarray(ref[p], float), np.asarray(blocks[p], float))))
    return float(np.sqrt(tot / 18))


def neighborhood_radius(blocks, ref):
    """Smallest R with (R^2/5) G_hat + D positive definite (Admal et al. 2017, eq. 3.11).

    blocks: float blocks of D (symmetric); ref: blocks of G_hat = C (x) delta
    (positive definite). Block by block, the condition is R^2/5 > -lambda_min
    with lambda_min the smallest generalized eigenvalue of (D^p, G_hat^p).
    Returns (R, p) with p the irrep that sets R (None and R = 0 when D is
    positive definite). Units: lengths of D/C (A for eV/A over eV/A^3).
    """
    worst, label = 0.0, None
    for p, B in blocks.items():
        B = np.asarray(B, float)
        G = np.asarray(ref[p], float)
        lam = sl.eigh(0.5 * (B + B.T), 0.5 * (G + G.T), eigvals_only=True)[0]
        if -lam > worst:
            worst, label = -lam, p
    return float(np.sqrt(5 * worst)), label
