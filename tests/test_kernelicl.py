"""Matched edited contexts and calibration for KernelICL (kernelicl/), on a tiny random model."""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(REPO, "src"), os.path.join(REPO, "kernelicl")):
    if path not in sys.path:
        sys.path.insert(0, path)

from tabicl._model.kernel_head import KernelHead  # noqa: E402
from tabicl._model.tabicl import TabICL  # noqa: E402
from kernelicl_diagnostics import KernelICL, KernelICLPredictor  # noqa: E402
from kernelicl_clinical import _make_folds, calibrate_scale, fold_embeddings  # noqa: E402
from kernel_louis.heads import kernel_head_proba  # noqa: E402
from kernel_louis.kernels import squared_distances  # noqa: E402

CONFIG = dict(max_classes=4, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=8,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              zero_init=False)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    backbone = TabICL(**CONFIG).eval()
    d_model = CONFIG["embed_dim"] * CONFIG["row_num_cls"]
    head = KernelHead(d_model=d_model, d_k=d_model, kernel="gaussian").eval()
    return KernelICL(backbone=backbone, head=head, config=CONFIG, val_loss=float("nan"), finetune_config={})


def data(seed=0, n=40, m=10, d=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    return X, (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0), rng.normal(size=(m, d)).astype(np.float32)


@pytest.fixture(scope="module")
def setup():
    X, y, Xq = data()
    pred = KernelICLPredictor(tiny_model(), X, y, gamma=0.5)
    return pred, X, y, Xq


def test_unedited_context_reproduces_predict_proba(setup):
    pred, X, y, Xq = setup
    rows, queries, labels = pred.processed(Xq)
    (_, probs), = list(pred.edited_passes([dict(rows=rows, labels=labels, queries=queries)]))
    np.testing.assert_allclose(probs[0], pred.predict_proba(Xq), atol=1e-6)


def test_matched_preprocessing_is_not_refitted(setup):
    pred, X, y, Xq = setup
    rows, queries, labels = pred.processed(Xq)
    keep = torch.arange(len(y)) != 0
    (_, probs), = list(pred.edited_passes([dict(rows=rows[keep], labels=labels[keep], queries=queries)]))
    # A deletion re-encoded by hand from the same fitted rows gives the same result...
    E_train, E_test = pred.embed_rows(rows[keep][None], labels[keep][None].float(), queries[None])
    direct, _ = pred.head(E_train, E_test, labels[keep][None].float(), num_classes=pred.n_classes_, gamma=0.5)
    np.testing.assert_allclose(probs[0], direct[0].detach().numpy(), atol=1e-6)
    # ...and the full-context preprocessing is unchanged by the edit.
    rows_again, _, _ = pred.processed(Xq)
    assert torch.equal(rows, rows_again)


def test_batched_edits_equal_one_at_a_time(setup):
    pred, X, y, Xq = setup
    rows, queries, labels = pred.processed(Xq)
    contexts = []
    for i in range(5):
        keep = torch.arange(len(y)) != i
        contexts.append(dict(rows=rows[keep], labels=labels[keep], queries=queries))
    batched = np.concatenate([p for _, p in pred.edited_passes(contexts, batch_size=3)])
    single = np.concatenate([p for _, p in pred.edited_passes(contexts, batch_size=1)])
    np.testing.assert_allclose(batched, single, atol=1e-5)


def test_kernel_head_proba_equals_kernel_head(setup):
    pred, X, y, Xq = setup
    P, W = pred.predict_proba_and_weights(Xq)
    E_train, E_test = pred._embed(pred.clf, Xq)
    H_train = pred.head.embed(E_train)[0].detach().numpy().astype(np.float64)
    H_test = pred.head.embed(E_test)[0].detach().numpy().astype(np.float64)
    y_enc = pred.clf.y_encoder_.transform(y)
    probs, _, weights = kernel_head_proba(-0.5 * squared_distances(H_test, H_train), y_enc,
                                          pred.n_classes_, return_weights=True)
    np.testing.assert_allclose(probs, P, atol=1e-5)
    np.testing.assert_allclose(weights, W, atol=1e-5)


def test_determinism_and_row_order_invariance(setup):
    pred, X, y, Xq = setup
    P = pred.predict_proba(Xq)
    np.testing.assert_allclose(pred.predict_proba(Xq), P, atol=1e-7)
    perm = np.random.default_rng(1).permutation(len(y))
    other = KernelICLPredictor(pred.model, X[perm], y[perm], gamma=0.5)
    np.testing.assert_allclose(other.predict_proba(Xq), P, atol=1e-4)


def test_calibrate_scale_follows_its_selection_rule(setup):
    pred, X, y, Xq = setup
    folds = _make_folds(y, 3, 0.2, 0)
    fold_data = fold_embeddings(pred._fit, X, y, folds)
    grid = [0.05, 0.5, 5.0]
    scale, accuracy, perplexity, scores = calibrate_scale(pred.head, fold_data, grid, 0.01)
    means = {s: np.mean(v) for s, v in scores.items()}
    assert accuracy == pytest.approx(means[scale])
    assert accuracy >= max(means.values()) - 0.01


def test_frozen_loo_proba_is_stable_for_sharp_kernels():
    """loo_proba('frozen') must equal renormalising without the self-weight, also when it saturates."""
    from kernel_louis.heads import kernel_head_proba
    from kernel_louis.kernels import squared_distances as sq_np
    X, y, _ = data(seed=3)
    for gamma in [0.5, 500.0]:
        pred = KernelICLPredictor(tiny_model(), X, y, gamma=gamma)
        H = pred.head.embed(pred.E_train)[0].detach().numpy().astype(np.float64)
        log_K = -gamma * sq_np(H, H)
        np.fill_diagonal(log_K, -np.inf)
        y_enc = pred.clf.y_encoder_.transform(y)
        P, _ = kernel_head_proba(log_K, y_enc, pred.n_classes_)
        np.testing.assert_allclose(pred.loo_proba("frozen"), P[np.arange(len(y)), y_enc], atol=1e-5)
