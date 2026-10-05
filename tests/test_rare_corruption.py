"""Tests for the multiclass helpers used by the MNIST-C rare-corruption notebook."""

import os

import numpy as np
import pandas as pd
import pytest

from kernel_louis.kernels import gaussian_kernel, log_gaussian_kernel, median_heuristic_gamma, squared_distances
from kernel_louis.heads import kernel_head_proba, self_normalized_predict
from kernel_louis.loo import (
    crossfit_difficulty,
    frozen_loo_proba,
    frozen_loo_score,
    gp_loo_logdensity,
    gp_loo_precision,
    gp_loo_score_multiclass,
    incontext_loo_nll,
    true_class_proba,
)
from kernel_louis.modulation import (
    group_balance_multiplier,
    prune_multiplier,
    pseudo_group_balance,
    random_prune_multiplier,
    random_tail_multiplier,
    surprisal_from_score,
    tail_multiplier,
    top_fraction_mask,
)
from kernel_louis.evaluation import (
    detection_auc,
    group_accuracy_metrics,
    mean_and_se,
    paired_differences,
    tail_composition,
)
from kernel_louis.predictors import fit_weighted_logreg, make_logreg, one_nn_predict
from kernel_louis import mnist_c


def random_problem(seed=0, n=40, d=3, n_classes=4, n_query=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, n_classes, size=n)
    Xq = rng.normal(size=(n_query, d))
    return X, y, Xq, median_heuristic_gamma(X, factor=1.0)


# ---------------------------------------------------------------- scores


def test_frozen_loo_score_matches_brute_force_deletion():
    """Eq. 9 equals deleting column i, renormalising, and reading p(y_i) at x_i."""
    X, y, _, gamma = random_problem()
    K = gaussian_kernel(X, X, gamma)
    L, n_zero = frozen_loo_score(K, y, 4)
    assert n_zero == 0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        proba, _ = kernel_head_proba(log_gaussian_kernel(X[i:i + 1], X[keep], gamma), y[keep], 4)
        assert abs(L[i] - (1.0 - proba[0, y[i]])) < 1e-12


def test_frozen_loo_score_agrees_with_press_form():
    """1 - Eq. 9 equals the PRESS-identity probability of incontext_loo_nll."""
    X, y, _, gamma = random_problem(seed=1)
    K = gaussian_kernel(X, X, gamma)
    L, _ = frozen_loo_score(K, y, 4)
    _, p_press = incontext_loo_nll(K, y, return_probs=True)
    np.testing.assert_allclose(1.0 - L, p_press, atol=1e-9)


def test_frozen_loo_zero_denominator_gives_score_one():
    K = np.eye(3)  # every point is only similar to itself
    L, n_zero = frozen_loo_score(K, np.array([0, 1, 2]), 3)
    assert n_zero == 3 and np.all(L == 1.0)


def test_frozen_loo_proba_rows_sum_to_one():
    X, y, _, gamma = random_problem(seed=2)
    P, _ = frozen_loo_proba(gaussian_kernel(X, X, gamma), y, 4)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_gp_multiclass_score_is_mean_of_binary_scores():
    """Eq. A.20 on one-hot targets equals averaging the scalar GP LOO score per class."""
    X, y, _, gamma = random_problem(seed=3)
    K = gaussian_kernel(X, X, gamma)
    L_multi = gp_loo_score_multiclass(gp_loo_precision(K, 0.1), y, 4)
    per_class = [gp_loo_logdensity(K, (y == c).astype(float), sigma2=0.1) for c in range(4)]
    np.testing.assert_allclose(L_multi, np.mean(per_class, axis=0), rtol=1e-8)


def test_crossfit_multiclass_and_true_class_proba():
    X, y, _, _ = random_problem(seed=4, n=80)
    L, p = crossfit_difficulty(make_logreg(), X, y, n_splits=5, return_probs=True)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(L, -np.log(np.clip(p, 1e-12, 1)))
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    np.testing.assert_allclose(true_class_proba(proba, np.array([0, 1]), np.array([1, 0])), [0.8, 0.6])
    assert true_class_proba(proba, np.array([0, 1]), np.array([5, 0]))[0] == 0.0


# ---------------------------------------------------------------- kernel head


def test_kernel_head_matches_plain_normalised_vote():
    X, y, Xq, gamma = random_problem(seed=5)
    m = np.random.default_rng(0).uniform(0.5, 3, size=len(y))
    proba, n_zero = kernel_head_proba(log_gaussian_kernel(Xq, X, gamma), y, 4, m=m)
    expected, _ = self_normalized_predict(gaussian_kernel(Xq, X, gamma), np.eye(4)[y], m=m)
    assert n_zero == 0
    np.testing.assert_allclose(proba, expected, atol=1e-12)


def test_kernel_head_survives_tiny_bandwidth_and_all_zero_weights():
    X, y, Xq, _ = random_problem(seed=6)
    log_K = log_gaussian_kernel(Xq, X, 1e4)  # exp() would underflow to 0 everywhere
    proba, n_zero = kernel_head_proba(log_K, y, 4)
    assert n_zero == 0 and np.all(np.isfinite(proba))
    proba, n_zero = kernel_head_proba(log_K, y, 4, m=np.zeros(len(y)))
    assert n_zero == len(Xq)
    np.testing.assert_allclose(proba, 0.25)


def test_fixed_feature_negative_control_multiclass():
    """Sanity check 6: frozen deletion equals recomputed deletion on fixed features (E_i = 0)."""
    X, y, Xq, gamma = random_problem(seed=7)
    log_K = log_gaussian_kernel(Xq, X, gamma)
    proba, _, W = kernel_head_proba(log_K, y, 4, return_weights=True)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        frozen = (proba - W[:, [i]] * np.eye(4)[y[i]]) / (1.0 - W[:, [i]])  # Eq. 8 rearranged
        recomputed, _ = kernel_head_proba(log_gaussian_kernel(Xq, X[keep], gamma), y[keep], 4)
        assert np.max(np.abs(frozen - recomputed)) < 1e-10


def test_resample_counts_equal_duplicated_context():
    """Multiplicities as multipliers equal physically duplicating rows (frozen kernel)."""
    X, y, Xq, gamma = random_problem(seed=8)
    idx = np.random.default_rng(1).choice(len(y), size=len(y), replace=True)
    counts = np.bincount(idx, minlength=len(y))
    a, _ = kernel_head_proba(log_gaussian_kernel(Xq, X, gamma), y, 4, m=counts)
    b, _ = kernel_head_proba(log_gaussian_kernel(Xq, X[idx], gamma), y[idx], 4)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_median_heuristic_factor_and_full_sample():
    X = np.array([[0.0], [1.0], [3.0], [3.0]])  # nonzero squared distances: 1, 9, 9, 4, 4
    assert median_heuristic_gamma(X, max_samples=None, factor=1.0) == pytest.approx(1 / 4)
    assert median_heuristic_gamma(X, max_samples=None) == pytest.approx(1 / 8)


# ---------------------------------------------------------------- interventions


def test_group_balance_gives_equal_mass_and_mean_one():
    g = np.array([0] * 90 + [1] * 7 + [2] * 3)
    m = group_balance_multiplier(g)
    assert m.mean() == pytest.approx(1.0)
    np.testing.assert_allclose([m[g == k].sum() for k in range(3)], 100 / 3)


def test_pseudo_group_balance_multiclass():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, size=500)
    L = rng.random(500)
    m = pseudo_group_balance(y, L, tau=0.95)
    cells = 2 * y + (L >= np.quantile(L, 0.95))
    sums = [m[cells == c].sum() for c in np.unique(cells)]
    np.testing.assert_allclose(sums, sums[0])
    assert m.mean() == pytest.approx(1.0)


def test_tail_prune_and_random_controls_have_matching_sizes():
    rng = np.random.default_rng(0)
    L = rng.random(1000)
    m_tail = tail_multiplier(L, lam=8, tau=0.95)
    size = int((m_tail > 1).sum())
    m_rand = random_tail_multiplier(1000, size, 8, rng)
    assert int((m_rand > 1).sum()) == size and set(np.unique(m_rand)) == {1.0, 9.0}
    assert int((prune_multiplier(L, 0.1) == 0).sum()) == 100
    assert int((random_prune_multiplier(1000, 0.1, rng) == 0).sum()) == 100
    assert top_fraction_mask(np.array([1.0, 1.0, 0.0]), 1 / 3).tolist() == [True, False, False]


def test_surprisal_from_score():
    np.testing.assert_allclose(surprisal_from_score(np.array([0.0, 0.5, 1.0])),
                               [0.0, np.log(2), -np.log(1e-12)])


# ---------------------------------------------------------------- metrics


def test_group_accuracy_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    g = np.array([0, 0, 1, 1, 2, 2])
    out = group_accuracy_metrics(y_pred, y_true, g, n_groups=3, rho=0.1)
    assert out["acc_g0"] == 1.0 and out["acc_g1"] == 0.5 and out["worst_group"] == 0.5
    assert out["balanced_mean"] == pytest.approx(2 / 3)
    assert out["context_mean"] == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)
    assert out["worst_cell"] == 0.0


def test_tail_composition_and_auc():
    L = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    g = np.array([0, 0, 0, 1, 2])
    row = tail_composition(L, g, tau=0.6, n_groups=3)
    assert row["tail_size"] == 2 and row["precision"] == 1.0 and row["recall"] == 1.0
    assert row["n_g1"] == 1 and row["n_g2"] == 1
    assert detection_auc(L, g > 0) == 1.0
    assert np.isnan(detection_auc(L, np.zeros(5, dtype=bool)))


def test_mean_and_se_and_paired_differences():
    df = pd.DataFrame({
        "seed": [0, 0, 1, 1],
        "predictor": ["k"] * 4,
        "intervention": ["baseline", "tail", "baseline", "tail"],
        "worst_group": [0.5, 0.6, 0.4, 0.7],
    })
    summary = mean_and_se(df, ["predictor", "intervention"], ["worst_group"])
    tail = summary[summary.intervention == "tail"].iloc[0]
    assert tail.worst_group_mean == pytest.approx(0.65)
    assert tail.worst_group_se == pytest.approx(np.std([0.6, 0.7], ddof=1) / np.sqrt(2))
    diffs = paired_differences(df, "worst_group", "baseline", ["predictor"], ["intervention"])
    assert diffs[diffs.intervention == "tail"].iloc[0].diff_mean == pytest.approx(0.2)


# ---------------------------------------------------------------- predictors


def test_weighted_logreg_drops_zeros_and_rescales():
    X, y, _, _ = random_problem(seed=9, n=120)
    m = np.ones(len(y))
    m[:20] = 0.0
    a = fit_weighted_logreg(X, y, m)
    b = fit_weighted_logreg(X[20:], y[20:])
    c = fit_weighted_logreg(X, y, 5.0 * m)  # constant rescaling must not change the fit
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-4)
    np.testing.assert_allclose(a.coef_, c.coef_, atol=1e-4)


def test_one_nn_with_pruning():
    X = np.array([[0.0], [1.0], [10.0]])
    y = np.array([0, 1, 2])
    assert one_nn_predict(X, y, np.array([[0.1]])).tolist() == [0]
    keep = np.array([False, True, True])
    assert one_nn_predict(X, y, np.array([[0.1]]), keep).tolist() == [1]


# ---------------------------------------------------------------- MNIST-C data


def write_fake_mnist_c(root, versions, n_train=400, n_test=100, corrupt=None):
    rng = np.random.default_rng(0)
    labels = {"train": np.arange(n_train) % 10, "test": np.arange(n_test) % 10}
    for k, v in enumerate(versions):
        os.makedirs(os.path.join(root, v))
        for split, y in labels.items():
            images = np.full((len(y), 28, 28, 1), k, dtype=np.uint8)
            y_saved = y.copy()
            if v == corrupt and split == "train":
                y_saved[0] = (y_saved[0] + 1) % 10
            np.save(os.path.join(root, v, f"{split}_images.npy"), images)
            np.save(os.path.join(root, v, f"{split}_labels.npy"), y_saved)
    return labels


def test_verify_mnist_c(tmp_path):
    versions = ["identity", "fog", "zigzag"]
    write_fake_mnist_c(str(tmp_path / "ok"), versions)
    labels = mnist_c.verify_mnist_c(str(tmp_path / "ok"), versions, {"train": 400, "test": 100})
    assert labels["train"].shape == (400,)
    write_fake_mnist_c(str(tmp_path / "bad"), versions, corrupt="fog")
    with pytest.raises(AssertionError):
        mnist_c.verify_mnist_c(str(tmp_path / "bad"), versions, {"train": 400, "test": 100})


def test_index_splits_are_disjoint_and_stratified():
    labels = np.arange(60000) % 10
    splits = mnist_c.make_index_splits(labels, seed=12345)
    assert len(splits["val"]) == 5000 and len(splits["cnn"]) == 20000
    assert len(splits["context"]) == 35000
    all_idx = np.concatenate([splits["val"], splits["cnn"], splits["context"]])
    assert len(np.unique(all_idx)) == 60000
    assert np.all(np.bincount(labels[splits["cnn"]]) == 2000)


def test_build_context_and_paired_set(tmp_path):
    versions = ["identity", "fog", "zigzag"]
    labels = write_fake_mnist_c(str(tmp_path), versions)["train"]
    rng = np.random.default_rng(0)
    r = mnist_c.rare_per_class(30, 0.1, 2)
    assert r == 2 and mnist_c.rare_per_class(600, 0.01, 3) == 2
    ctx = mnist_c.build_context(np.arange(400), labels, versions, [30 - 2 * r, r, r], rng)
    assert len(ctx["idx"]) == 300 and np.all(np.bincount(ctx["g"]) == [260, 20, 20])
    assert np.all(ctx["version"][ctx["g"] == 1] == "fog")
    images = mnist_c.gather_images(str(tmp_path), "train", ctx["idx"], ctx["version"])
    np.testing.assert_array_equal(images[:, 0, 0], ctx["g"])  # fake image pixel = version number
    test = mnist_c.build_paired_set(np.arange(100), np.arange(100) % 10, versions, 5, rng)
    assert len(test["idx"]) == 150
    for k in range(3):
        np.testing.assert_array_equal(test["idx"][test["g"] == k], test["idx"][test["g"] == 0])


def test_flip_labels():
    rng = np.random.default_rng(0)
    y = np.arange(1000) % 10
    y_sym, flipped = mnist_c.flip_labels(y, 0.2, "symmetric", rng)
    assert flipped.sum() == 200 and np.all(y_sym[flipped] != y[flipped])
    assert np.all(y_sym[~flipped] == y[~flipped])
    y_asym, flipped = mnist_c.flip_labels(y, 0.4, "asymmetric", rng)
    assert all(y_asym[i] == mnist_c.ASYMMETRIC_MAP[y[i]] for i in np.where(flipped)[0])


def test_small_cnn_feature_shape():
    torch = pytest.importorskip("torch")
    from kernel_louis.cnn import SmallCNN, extract_features
    model = SmallCNN().eval()
    feats = extract_features(model, np.zeros((5, 28, 28), dtype=np.uint8))
    assert feats.shape == (5, 128) and feats.dtype == np.float64


def test_thread_limit_does_not_change_results():
    """Single-threaded fitting gives the same model; the thread setting is restored afterwards."""
    from threadpoolctl import threadpool_info
    X, y, _, _ = random_problem(seed=10, n=150)
    before = [pool["num_threads"] for pool in threadpool_info()]
    L_default = crossfit_difficulty(make_logreg(), X, y, n_splits=5)
    L_single = crossfit_difficulty(make_logreg(), X, y, n_splits=5, max_threads=1)
    np.testing.assert_allclose(L_default, L_single, atol=1e-6)
    a = fit_weighted_logreg(X, y)
    assert [pool["num_threads"] for pool in threadpool_info()] == before
    np.testing.assert_allclose(a.predict_proba(X), make_logreg().fit(X, y).predict_proba(X), atol=1e-6)


def test_loo_kernel_from_log_matches_plain_kernel_and_survives_narrow_bandwidth():
    from kernel_louis.loo import loo_kernel_from_log
    X, y, _, gamma = random_problem(seed=11)
    log_K = log_gaussian_kernel(X, X, gamma)
    P_plain, _ = frozen_loo_proba(np.exp(log_K), y, 4)
    P_stable, _ = frozen_loo_proba(loo_kernel_from_log(log_K), y, 4)
    np.testing.assert_allclose(P_stable, P_plain, atol=1e-12)
    # At 1e4 x the bandwidth every off-diagonal exp() underflows, but the stable version
    # still gives each point's nearest neighbour all the weight.
    log_K_narrow = log_gaussian_kernel(X, X, 1e4 * gamma)
    _, n_zero_plain = frozen_loo_proba(np.exp(log_K_narrow), y, 4)
    P_narrow, n_zero = frozen_loo_proba(loo_kernel_from_log(log_K_narrow), y, 4)
    assert n_zero_plain > 0 and n_zero == 0
    D = squared_distances(X, X) + np.diag(np.full(len(y), np.inf))
    np.testing.assert_allclose(P_narrow.argmax(axis=1), y[D.argmin(axis=1)])


def test_select_gamma_loo_maximises_brute_force_loo_likelihood():
    from kernel_louis.loo import select_gamma_loo
    X, y, _, gamma0 = random_problem(seed=12, n=60)
    gammas = gamma0 * np.geomspace(1, 100, 7)
    best, loo_ll = select_gamma_loo(squared_distances(X, X), y, 4, gammas)
    brute = []
    for gamma in gammas:
        ll = 0.0
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            proba, _ = kernel_head_proba(log_gaussian_kernel(X[i:i + 1], X[keep], gamma), y[keep], 4)
            ll += np.log(max(proba[0, y[i]], 1e-12))
        brute.append(ll / len(y))
    np.testing.assert_allclose(loo_ll, brute, atol=1e-9)
    assert best == gammas[int(np.argmax(brute))]


def test_mean_and_se_ignores_missing_values_in_the_count():
    df = pd.DataFrame({"seed": [0, 1, 2], "g": ["a"] * 3, "x": [1.0, 3.0, np.nan]})
    row = mean_and_se(df, ["g"], ["x"]).iloc[0]
    assert row.x_mean == pytest.approx(2.0)
    assert row.x_se == pytest.approx(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
