"""Unit tests verifying mathematical exactness of closed-form LOO identities."""

import numpy as np
import pytest

from kernel_louis.kernels import default_gamma, gaussian_kernel
from kernel_louis.heads import self_normalized_predict, gp_predict
from kernel_louis.loo import incontext_loo_nll, gp_loo_logdensity, press_residuals
from kernel_louis.modulation import tail_multiplier, smooth_multiplier, exponential_multiplier
from kernel_louis.audit import audit_kernel_head_influence


def test_soft_vote_loo_exactness():
    """Verify that incontext_loo_nll matches genuine brute-force leave-one-out."""
    rng = np.random.default_rng(42)
    n = 30
    d = 4
    X = rng.normal(0, 1, (n, d))  # float64
    y = rng.integers(0, 2, size=n)

    gamma = default_gamma(d)
    K = gaussian_kernel(X, X, gamma)

    # 1. Closed-form score
    L_closed, p_closed = incontext_loo_nll(K, y, return_probs=True)

    # 2. Brute force leave-one-out
    p_brute = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_sub, y_sub = X[mask], y[mask]

        K_sub = gaussian_kernel(X[i:i+1], X_sub, gamma)
        p1, _ = self_normalized_predict(K_sub, y_sub)
        p_brute[i] = p1[0] if y[i] == 1 else (1.0 - p1[0])

    # Assert exact match to high numerical precision
    np.testing.assert_allclose(p_closed, p_brute, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(L_closed, -np.log(p_brute), rtol=1e-9, atol=1e-9)


def test_gp_loo_exactness():
    """Verify that Rasmussen & Williams GP LOO matches brute-force GP retraining."""
    rng = np.random.default_rng(123)
    n = 25
    d = 3
    X = rng.normal(0, 1, (n, d))
    y = rng.integers(0, 2, size=n)

    gamma = default_gamma(d)
    sigma2 = 0.2
    K = gaussian_kernel(X, X, gamma)

    # Closed-form
    L_gp_closed = gp_loo_logdensity(K, y, sigma2=sigma2)

    # Brute-force GP leave-one-out
    L_gp_brute = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_sub, y_sub = X[mask], y[mask]

        K_sub_tr = gaussian_kernel(X_sub, X_sub, gamma)
        K_sub_qt = gaussian_kernel(X[i:i+1], X_sub, gamma)

        mu, var = gp_predict(K_sub_qt, K_sub_tr, y_sub, sigma2=sigma2)
        total_var = var[0] + sigma2
        log_dens = -0.5 * np.log(2 * np.pi * total_var) - 0.5 * ((y[i] - mu[0]) ** 2) / total_var
        L_gp_brute[i] = -log_dens

    np.testing.assert_allclose(L_gp_closed, L_gp_brute, rtol=1e-5, atol=1e-5)


def test_modulation_baseline_recovery():
    """Verify that lam=0 exactly recovers unmodified predictions."""
    rng = np.random.default_rng(0)
    n = 20
    X = rng.normal(0, 1, (n, 3))
    y = rng.integers(0, 2, size=n)
    gamma = default_gamma(3)
    K = gaussian_kernel(X, X, gamma)

    L = incontext_loo_nll(K, y)
    m_zero = tail_multiplier(L, lam=0.0)
    assert np.all(m_zero == 1.0)

    p_base, W_base = self_normalized_predict(K, y)
    p_mod, W_mod = self_normalized_predict(K, y, m=m_zero)

    np.testing.assert_allclose(p_base, p_mod)
    np.testing.assert_allclose(W_base, W_mod)


def test_context_influence_audit_exactness():
    """Verify that for a fixed kernel head, representation discrepancy is 0 (Eq. 12)."""
    rng = np.random.default_rng(99)
    Xtr = rng.normal(0, 1, (40, 4))
    ytr = rng.integers(0, 2, size=40)
    Xq = rng.normal(0, 1, (10, 4))

    gamma = default_gamma(4)
    K_tr = gaussian_kernel(Xtr, Xtr, gamma)
    K_qt = gaussian_kernel(Xq, Xtr, gamma)

    audit = audit_kernel_head_influence(K_qt, K_tr, ytr, del_idx=5)
    np.testing.assert_allclose(audit["discrepancy"], 0.0, atol=1e-12)
