"""Conversion between Mindlin's form I and form II kinematic variables.

Form I uses the second displacement gradient, kappa_ijk = d_j d_k u_i
(symmetric in j, k). Form II uses the strain gradient, eta_ijk = d_k eps_ij
(symmetric in i, j). The map between them is linear, invertible and
isotropic, but it is not an isometry.
"""

import numpy as np


def form1_to_form2(kappa):
    """Return eta_ijk = (kappa_ijk + kappa_jik) / 2."""
    kappa = np.asarray(kappa, dtype=float)
    return 0.5 * (kappa + np.einsum("jik->ijk", kappa))


def form2_to_form1(eta):
    """Return kappa_ijk = eta_ijk + eta_ikj - eta_jki."""
    eta = np.asarray(eta, dtype=float)
    return eta + np.einsum("ikj->ijk", eta) - np.einsum("jki->ijk", eta)
