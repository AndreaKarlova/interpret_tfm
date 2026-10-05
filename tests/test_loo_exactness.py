"""Unit tests verifying mathematical exactness of closed-form LOO identities."""

import numpy as np
import pytest

from kernel_louis.kernels import default_gamma, gaussian_kernel
from kernel_louis.heads import self_normalized_predict, gp_predict
from kernel_louis.loo import incontext_loo_nll, gp_loo_logdensity, press_residuals
from kernel_louis.modulation import tail_multiplier, smooth_multiplier, exponential_multiplier
from kernel_louis.audit import (
    audit_kernel_head_influence,
    audit_context_admission,
    compute_audit_metrics,
)


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


def test_context_admission_audit_exactness():
    """Verify that on fixed features, admission remainder rho_z is identically 0 (Eq. 12)."""
    rng = np.random.default_rng(101)
    Xtr = rng.normal(0, 1, (30, 3))
    ytr = rng.integers(0, 2, size=30)
    Xq = rng.normal(0, 1, (8, 3))
    z = rng.normal(0, 1, (1, 3))
    v = 1.0

    gamma = default_gamma(3)
    K_qt = gaussian_kernel(Xq, Xtr, gamma)
    k_qz = gaussian_kernel(Xq, z, gamma)

    audit = audit_context_admission(K_qt, k_qz, ytr, candidate_label=v)
    np.testing.assert_allclose(audit["remainder"], 0.0, atol=1e-12)
    assert audit["max_abs_remainder"] < 1e-12

    # Verify Table 1 audit metrics computation
    metrics = compute_audit_metrics(audit["actual_addition_effect"], audit["closed_form_addition"])
    assert metrics["rms"] < 1e-12
    assert metrics["bias"] < 1e-12
    assert metrics["sign_agreement"] == 1.0
    assert metrics["top_k_overlap"] == 1.0



# ---------------------------------------------------------------- KernelICL audit formulas


from kernel_louis.heads import kernel_head_proba
from kernel_louis.kernels import log_gaussian_kernel
from kernel_louis.audit import (
    frozen_admission_effects,
    frozen_deletion_effects,
    frozen_deletion_vectors,
    frozen_relabel_effects,
)
from kernel_louis.loo import (
    gp_frozen_deletion_effects,
    gp_heldout_loglik,
    gp_loo_precision,
    select_gp_hyperparameters_cv,
)
from kernel_louis.evaluation import purity


def multiclass_problem(seed=0, n=30, m=8, C=4, d=3):
    rng = np.random.default_rng(seed)
    X, Xq = rng.normal(size=(n, d)), rng.normal(size=(m, d))
    return X, Xq, rng.integers(0, C, size=n), rng.integers(0, C, size=m), C


def test_frozen_deletion_effects_match_brute_force():
    X, Xq, y, out_cls, C = multiclass_problem()
    log_K = log_gaussian_kernel(Xq, X, 0.5)
    P, _, W = kernel_head_proba(log_K, y, C, return_weights=True)
    I_head = frozen_deletion_effects(P, W, y, out_cls)
    V = frozen_deletion_vectors(P, W, y, np.arange(len(y)))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        P_del, _ = kernel_head_proba(log_K[:, keep], y[keep], C)
        np.testing.assert_allclose(I_head[i], P[np.arange(len(Xq)), out_cls] - P_del[np.arange(len(Xq)), out_cls], atol=1e-12)
        np.testing.assert_allclose(V[i], P - P_del, atol=1e-12)


def test_frozen_relabel_and_admission_effects_match_brute_force():
    X, Xq, y, out_cls, C = multiclass_problem(seed=1)
    log_K = log_gaussian_kernel(Xq, X, 0.5)
    P, _, W = kernel_head_proba(log_K, y, C, return_weights=True)
    rows = np.array([0, 3, 7])
    y_new = (y[rows] + 1) % C
    effects = frozen_relabel_effects(W, y[rows], y_new, out_cls, rows)
    for r, i in enumerate(rows):
        y_edit = y.copy()
        y_edit[i] = y_new[r]
        P_new, _ = kernel_head_proba(log_K, y_edit, C)
        np.testing.assert_allclose(effects[r], P_new[np.arange(len(Xq)), out_cls] - P[np.arange(len(Xq)), out_cls], atol=1e-12)
    Z = np.random.default_rng(2).normal(size=(4, 3))
    v = np.array([0, 1, 2, 3])
    log_k_z = log_gaussian_kernel(Z, Xq, 0.5)                    # (n_cand, m)
    log_Z = np.logaddexp.reduce(log_K, axis=1)
    effects = frozen_admission_effects(log_k_z, log_Z, P, v, out_cls)
    for c in range(len(Z)):
        P_new, _ = kernel_head_proba(np.column_stack([log_K, log_k_z[c]]), np.append(y, v[c]), C)
        np.testing.assert_allclose(effects[c], P_new[np.arange(len(Xq)), out_cls] - P[np.arange(len(Xq)), out_cls], atol=1e-12)


def test_audit_metrics_follow_appendix_c3():
    actual = np.array([[0.5, 0.0], [-0.2, 0.0], [1e-6, 0.3]])
    head = np.array([[0.4, 0.9], [0.1, -0.9], [-0.1, 0.2]])
    m = compute_audit_metrics(actual, head, tol=1e-3, top_k=1)
    # only |actual| > tol counts for signs: entries 0.5 (agree), -0.2 (disagree), 0.3 (agree)
    assert m["sign_agreement"] == pytest.approx(2 / 3)
    # per query: query 0 top point 0 in both; query 1 top actual = point 2, top head = point 0 (tie by index)
    assert m["top_k_overlap"] == pytest.approx(0.5)
    assert m["relative_rms"] == pytest.approx(m["rms"] / m["rms_actual"])


def test_gp_frozen_deletion_matches_brute_force_refit():
    rng = np.random.default_rng(3)
    X, Xq = rng.normal(size=(25, 3)), rng.normal(size=(6, 3))
    Y = np.eye(3)[rng.integers(0, 3, size=25)]
    out_cls = rng.integers(0, 3, size=6)
    K, K_q = gaussian_kernel(X, X, 0.4), gaussian_kernel(Xq, X, 0.4)
    Q = gp_loo_precision(K, 0.1)
    effects = gp_frozen_deletion_effects(K_q, Q, Q @ Y, out_cls)
    mu, _ = gp_predict(K_q, K, Y, sigma2=0.1)
    for i in range(25):
        keep = np.arange(25) != i
        mu_del, _ = gp_predict(K_q[:, keep], K[np.ix_(keep, keep)], Y[keep], sigma2=0.1)
        np.testing.assert_allclose(effects[i], mu[np.arange(6), out_cls] - mu_del[np.arange(6), out_cls], atol=1e-8)


def test_select_gp_hyperparameters_cv_returns_table_argmax():
    rng = np.random.default_rng(4)
    H = rng.normal(size=(60, 3))
    Y = np.eye(2)[(H[:, 0] > 0).astype(int)]
    folds = [(H[:40], H[40:], Y[:40], Y[40:]), (H[20:], H[:20], Y[20:], Y[:20])]
    gammas, sigma2s = np.array([0.01, 0.3, 3.0]), np.array([0.01, 0.1, 1.0])
    gamma, sigma2, table = select_gp_hyperparameters_cv(folds, gammas, sigma2s)
    a, b = np.unravel_index(np.argmax(table), table.shape)
    assert (gamma, sigma2) == (gammas[a], sigma2s[b])
    direct = gp_heldout_loglik(gaussian_kernel(H[40:], H[:40], 0.3), gaussian_kernel(H[:40], H[:40], 0.3), Y[:40], Y[40:], 0.1)
    direct += gp_heldout_loglik(gaussian_kernel(H[:20], H[20:], 0.3), gaussian_kernel(H[20:], H[20:], 0.3), Y[20:], Y[:20], 0.1)
    assert table[1, 1] == pytest.approx(direct / 2)


def test_purity():
    P = np.array([[0.0], [0.1], [5.0], [5.1]])
    assert purity(P, np.array([0, 0, 1, 1]), k=1) == 1.0
    assert purity(P, np.array([0, 1, 0, 1]), k=1) == 0.0
    assert purity(P, np.array([0, 0, 1, 0]), k=1, subset=np.array([True, True, False, False])) == 1.0
