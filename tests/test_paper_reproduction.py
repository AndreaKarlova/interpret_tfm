"""Integration test verifying reproduction of key paper tables."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from kernel_louis.kernels import default_gamma, gaussian_kernel, median_heuristic_gamma
from kernel_louis.heads import self_normalized_predict
from kernel_louis.loo import incontext_loo_nll, gp_loo_logdensity
from kernel_louis.modulation import (
    tail_multiplier,
    smooth_multiplier,
    exponential_multiplier,
    pseudo_group_balance,
)
from kernel_louis.data import (
    make_tabular_waterbirds,
    make_noisy_label_split,
)
from kernel_louis.evaluation import (
    compute_group_metrics,
    compute_tail_precision_recall,
)


def test_table_1_mechanism_isolation():
    """Verify Table 1 trends: tail-set raises worst-group, exponential collapses."""
    results = {"base": [], "tail2": [], "tail5": [], "smooth10": [], "exp1": [], "oracle": []}

    for seed in range(5):
        Xtr, ytr, _, is_min_tr = make_tabular_waterbirds(n_samples=1000, spurious_corr=0.90, core_noise=1.1, random_state=seed)
        Xte, yte, _, is_min_te = make_tabular_waterbirds(n_samples=2000, spurious_corr=0.90, core_noise=1.1, random_state=seed + 10000)

        # Standardize
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

        gamma = median_heuristic_gamma(Xtr)
        Ktr = gaussian_kernel(Xtr, Xtr, gamma)
        Kte = gaussian_kernel(Xte, Xtr, gamma)

        L = incontext_loo_nll(Ktr, ytr)

        mults = {
            "base": np.ones(len(ytr)),
            "tail2": tail_multiplier(L, lam=2.0, tau=0.90),
            "tail5": tail_multiplier(L, lam=5.0, tau=0.90),
            "smooth10": smooth_multiplier(L, lam=10.0),
            "exp1": exponential_multiplier(L, gamma=1.0),
            "oracle": 1.0 + 10.0 * is_min_tr.astype(float),
        }

        for k, m in mults.items():
            p1, _ = self_normalized_predict(Kte, ytr, m=m)
            metrics = compute_group_metrics(p1 > 0.5, yte, is_min_te)
            results[k].append(metrics["worst"])

    means = {k: np.mean(v) for k, v in results.items()}
    assert means["base"] < 0.30
    assert means["tail2"] > means["base"]
    assert means["tail5"] > means["tail2"]
    assert means["exp1"] < means["tail5"]


def test_table_2_tail_composition():
    """Verify Table 2: precision rises sharply from tau=0.80 to tau=0.95."""
    Xtr, ytr, _, is_min_tr = make_tabular_waterbirds(n_samples=4000, spurious_corr=0.953, core_noise=1.2, random_state=42)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd

    gamma = median_heuristic_gamma(Xtr)
    Ktr = gaussian_kernel(Xtr, Xtr, gamma)
    L = incontext_loo_nll(Ktr, ytr)

    stats = compute_tail_precision_recall(L, is_min_tr, [0.80, 0.90, 0.95, 0.97])

    t80 = next(s for s in stats if s["tau"] == 0.80)
    t95 = next(s for s in stats if s["tau"] == 0.95)

    assert t80["tail_size"] == 800
    assert t95["tail_size"] == 200
    assert 0.20 <= t80["precision"] <= 0.26
    assert 0.80 <= t95["precision"] <= 0.92
    assert t80["recall"] >= 0.95


def test_table_3_tail_governs_accuracy():
    """Verify Table 3: sign reversal at tau=0.80 vs tau=0.95."""
    results = {"base": [], "t80_l8": [], "t80_l20": [], "t95_l8": [], "t95_l20": [], "pg_95": [], "oracle": []}

    for seed in range(3):
        Xtr, ytr, _, is_min_tr = make_tabular_waterbirds(n_samples=4000, spurious_corr=0.953, core_noise=1.2, random_state=seed)
        Xte, yte, _, is_min_te = make_tabular_waterbirds(n_samples=2000, spurious_corr=0.50, core_noise=1.2, random_state=seed + 10000)

        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

        gamma = median_heuristic_gamma(Xtr)
        Ktr = gaussian_kernel(Xtr, Xtr, gamma)
        L = incontext_loo_nll(Ktr, ytr)

        weights_dict = {
            "base": np.ones(len(ytr)),
            "t80_l8": tail_multiplier(L, lam=8.0, tau=0.80),
            "t80_l20": tail_multiplier(L, lam=20.0, tau=0.80),
            "t95_l8": tail_multiplier(L, lam=8.0, tau=0.95),
            "t95_l20": tail_multiplier(L, lam=20.0, tau=0.95),
            "pg_95": pseudo_group_balance(ytr, L, tau=0.95),
            "oracle": np.where(is_min_tr, 1.0 / is_min_tr.mean(), 1.0 / (1.0 - is_min_tr.mean())),
        }

        for k, w in weights_dict.items():
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xtr, ytr, sample_weight=w)
            pred = clf.predict(Xte)
            m = compute_group_metrics(pred, yte, is_min_te)
            results[k].append(m["worst"])

    means = {k: np.mean(v) for k, v in results.items()}
    # At tau=0.80, intervention hurts worst-group accuracy (sign reversal!)
    assert means["t80_l8"] < means["base"]
    assert means["t80_l20"] < means["t80_l8"]
    # At tau=0.95, intervention substantially boosts worst-group accuracy
    assert means["t95_l8"] > means["base"]
    assert means["t95_l20"] > 0.50
    assert means["pg_95"] > 0.50


def test_table_5_noise_detection_auc():
    """Verify Table 5: high detection AUC for 2-D and 20-D."""
    Xtr, ytr, yn, flip, _, _ = make_noisy_label_split(seed=42, n_train=600, n_noise_dims=0, eps_noise=0.20)
    gamma = default_gamma(2)
    Ktr = gaussian_kernel(Xtr, Xtr, gamma)
    L_vote = incontext_loo_nll(Ktr, yn)
    L_gp = gp_loo_logdensity(Ktr, yn, sigma2=0.1)

    auc_vote = roc_auc_score(flip, L_vote)
    auc_gp = roc_auc_score(flip, L_gp)

    assert auc_vote > 0.90
    assert auc_gp > 0.95
