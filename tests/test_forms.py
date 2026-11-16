"""Checks of the form I <-> form II conversion."""

import numpy as np
import pytest

from gradinv.forms import form1_to_form2, form2_to_form1


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _rotate(R, a):
    return np.einsum("ip,jq,kr,pqr->ijk", R, R, R, a)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def _random_kappa(rng):
    k = rng.normal(size=(3, 3, 3))
    return 0.5 * (k + np.einsum("ikj->ijk", k))


def _random_eta(rng):
    e = rng.normal(size=(3, 3, 3))
    return 0.5 * (e + np.einsum("jik->ijk", e))


def test_round_trips(rng):
    kappa, eta = _random_kappa(rng), _random_eta(rng)
    assert np.allclose(form2_to_form1(form1_to_form2(kappa)), kappa, atol=1e-14)
    assert np.allclose(form1_to_form2(form2_to_form1(eta)), eta, atol=1e-14)


def test_index_symmetries(rng):
    eta = form1_to_form2(_random_kappa(rng))
    kappa = form2_to_form1(_random_eta(rng))
    assert np.allclose(eta, np.einsum("jik->ijk", eta))
    assert np.allclose(kappa, np.einsum("ikj->ijk", kappa))


def test_commutes_with_rotations(rng):
    R = _random_rotation(rng)
    kappa = _random_kappa(rng)
    assert np.allclose(form1_to_form2(_rotate(R, kappa)), _rotate(R, form1_to_form2(kappa)), atol=1e-13)


def test_not_an_isometry(rng):
    kappa = _random_kappa(rng)
    ratio = np.linalg.norm(form1_to_form2(kappa)) / np.linalg.norm(kappa)
    assert not np.isclose(ratio, 1.0)


def test_matches_definition_from_displacement(rng):
    # Quadratic displacement u_i = 1/2 A_ijk x_j x_k gives d_j d_k u_i = A_ijk (sym in jk).
    A = _random_kappa(rng)
    grad_u = lambda x: np.einsum("ijk,k->ij", A, x)  # du_i/dx_j
    x = rng.normal(size=3)
    h = 1e-6
    eta_fd = np.empty((3, 3, 3))
    for k in range(3):
        dx = np.zeros(3)
        dx[k] = h
        G1, G0 = grad_u(x + dx), grad_u(x - dx)
        eps1, eps0 = 0.5 * (G1 + G1.T), 0.5 * (G0 + G0.T)
        eta_fd[:, :, k] = (eps1 - eps0) / (2 * h)
    assert np.allclose(form1_to_form2(A), eta_fd, atol=1e-8)
