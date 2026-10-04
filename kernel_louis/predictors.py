"""Predictors that consume an (edited) context, besides the kernel head in heads.py.

1. Multinomial logistic regression with per-point multipliers ("linear last layer").
2. 1-nearest-neighbour, where editing a context can only remove points.
3. A native tabular foundation model (TabICLv2), or a fast "stub" stand-in.
"""

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_logreg(C: float = 1.0, max_iter: int = 2000) -> LogisticRegression:
    """An unfitted multinomial logistic regression."""
    return LogisticRegression(C=C, max_iter=max_iter)


def fit_weighted_logreg(X: np.ndarray, y: np.ndarray, m: Optional[np.ndarray] = None,
                        C: float = 1.0, max_iter: int = 2000) -> LogisticRegression:
    """Logistic regression with sample_weight = m.

    Points with m = 0 are dropped and the remaining weights rescaled to mean 1:
    scikit-learn sums weighted losses, so unscaled upweighting would also weaken
    the effective regularisation.
    """
    m = np.ones(len(y)) if m is None else np.asarray(m, dtype=float)
    keep = m > 0
    weights = m[keep] / m[keep].mean()
    clf = make_logreg(C, max_iter)
    clf.fit(X[keep], y[keep], sample_weight=weights)
    return clf


def one_nn_predict(X_ctx: np.ndarray, y_ctx: np.ndarray, X_query: np.ndarray,
                   keep: Optional[np.ndarray] = None) -> np.ndarray:
    """1-NN predictions (Euclidean) using only the context points with keep == True."""
    keep = np.ones(len(y_ctx), dtype=bool) if keep is None else keep
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_ctx[keep], y_ctx[keep])
    return clf.predict(X_query)


def make_tfm(backend: str, seed: int, device: str = "cpu"):
    """An unfitted native TFM: TabICLv2 (backend='tabicl') or logistic regression (backend='stub').

    TabICL uses one estimator (no ensembling over feature/class permutations), as
    in the Waterbirds notebook, so every fit is a single in-context forward pass.
    """
    if backend == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(n_estimators=1, device=device, random_state=seed)
    assert backend == "stub", f"unknown backend {backend!r}"
    return make_logreg()


def tfm_fit_predict(backend: str, seed: int, device: str, X_ctx: np.ndarray,
                    y_ctx: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Fit the TFM on a context (an in-context forward pass for TabICL) and predict X_query."""
    clf = make_tfm(backend, seed, device)
    clf.fit(np.asarray(X_ctx, np.float32), np.asarray(y_ctx))
    return clf.predict(np.asarray(X_query, np.float32))
