"""Exact In-Context Leave-One-Out (LOO) scores from the Hat Matrix.

Mathematical foundation:
Because the KernelICL head is a linear smoother y_hat = S y, exact LOO
predictions follow in closed form from S without retraining, without Taylor
expansions, and without Hessian solves.

Eq. (2): PRESS identity for linear smoothers:
    y_i - y_hat_i^-i = (y_i - y_hat_i) / (1 - S_ii)

Eq. (3): Exact soft class-vote LOO:
    q_i = sum_j S_ij 1[y_j = y_i]
    p^-i(y_i) = (q_i - S_ii) / (1 - S_ii)
    L_i = -log p^-i(y_i)

Eq. (4): Deep Gaussian Process LOO (Rasmussen & Williams 2006):
    mu_i^-i = y_i - alpha_i / [K_tilde^-1]_ii
    (sigma_i^-i)^2 = 1 / [K_tilde^-1]_ii
    L_i^GP = -log N(y_i | mu_i^-i, (sigma_i^-i)^2)
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def incontext_loo_nll(
    K_tr: np.ndarray,
    y_train: np.ndarray,
    eps: float = 1e-12,
    return_probs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Exact in-context leave-one-out NLL difficulty score (Eq. 3).

    For the soft class-vote head, dropping context point i, renormalizing the
    weights over the remaining n-1 points, and evaluating the predictive
    probability of y_i is mathematically exact via the PRESS hat-matrix identity:
        p^-i(y_i) = (q_i - S_ii) / (1 - S_ii)

    Parameters
    ----------
    K_tr : (n, n) symmetric context Gram matrix
    y_train : (n,) context class labels in {0, 1}
    eps : numerical stability constant
    return_probs : if True, also return leave-one-out probability p^-i(y_i)

    Returns
    -------
    L : (n,) exact LOO negative log-likelihood scores (higher = harder / anomalous)
    p_loo : (n,) leave-one-out true-label probability mass (optional)
    """
    y = np.asarray(y_train, dtype=int)
    n = len(y)

    # Hat matrix S_ij = K_ij / sum_k K_ik
    row_sums = np.sum(K_tr, axis=1, keepdims=True) + eps
    S = K_tr / row_sums

    # S_ii is leverage / self-weight
    S_diag = np.clip(np.diag(S), 0.0, 1.0 - eps)

    # q_i is in-sample predicted mass on the true class y_i
    # same_label[i, j] = 1 if y_i == y_j else 0
    same_label = (y[:, None] == y[None, :]).astype(float)
    q = np.sum(S * same_label, axis=1)

    # Exact leave-one-out mass: (q_i - S_ii) / (1 - S_ii)
    p_loo = np.clip((q - S_diag) / (1.0 - S_diag + eps), eps, 1.0)
    L = -np.log(p_loo)

    if return_probs:
        return L, p_loo
    return L


def press_residuals(
    K_tr: np.ndarray,
    y_train: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Exact PRESS residuals for linear regression smoother (Eq. 2):
        y_i - y_hat_i^-i = (y_i - y_hat_i) / (1 - S_ii)
    """
    row_sums = np.sum(K_tr, axis=1, keepdims=True) + eps
    S = K_tr / row_sums
    y = np.asarray(y_train, dtype=float)
    y_hat = S @ y
    S_ii = np.clip(np.diag(S), 0.0, 1.0 - eps)
    return (y - y_hat) / (1.0 - S_ii)


def gp_loo_logdensity(
    K_tr: np.ndarray,
    y_train: np.ndarray,
    sigma2: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Exact Gaussian Process leave-one-out score (Eq. 4, Rasmussen & Williams):
    From the single inverse K_tilde^-1 = (K + sigma2 * I)^-1:
        mu_i^-i = y_i - alpha_i / [K_tilde^-1]_ii
        (sigma_i^-i)^2 = 1 / [K_tilde^-1]_ii
        L_i^GP = -log N(y_i | mu_i^-i, (sigma_i^-i)^2)

    Penalizes both large residuals and small predictive variance.
    """
    y = np.asarray(y_train, dtype=float)
    n = len(y)
    K_tilde = K_tr + sigma2 * np.eye(n)
    K_inv = np.linalg.inv(K_tilde)

    d = np.clip(np.diag(K_inv), eps, None)
    alpha = K_inv @ y
    mu_loo = y - alpha / d
    var_loo = 1.0 / d

    log_dens = -0.5 * np.log(2.0 * np.pi * var_loo) - 0.5 * ((y - mu_loo) ** 2) / var_loo
    return -log_dens


def frozen_loo_proba(
    K_tr: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, int]:
    """Frozen-head LOO class probabilities for every context point (paper Eq. A.13).

    Row i predicts x_i from the other context points: the diagonal affinity is
    removed and the remaining weights renormalised, with the kernel held fixed.

    Returns
    -------
    P_loo : (n, C) probabilities; rows with a zero retained denominator are all zero
    n_zero : number of such rows
    """
    y = np.asarray(y_train, dtype=int)
    K = np.array(K_tr, dtype=float, copy=True)
    np.fill_diagonal(K, 0.0)
    denom = K.sum(axis=1)
    zero = denom <= 0
    Y = np.eye(n_classes)[y]
    P_loo = (K @ Y) / np.where(zero, 1.0, denom)[:, None]
    P_loo[zero] = 0.0
    return P_loo, int(zero.sum())


def frozen_loo_score(
    K_tr: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, int]:
    """Frozen-head difficulty score of paper Eq. 9, L_i = 1 - p^{fr,-i}(y_i), in [0, 1].

    Same ranking as incontext_loo_nll (which returns -log p). A zero retained
    denominator gives L_i = 1. Returns (L, n_zero).
    """
    y = np.asarray(y_train, dtype=int)
    P_loo, n_zero = frozen_loo_proba(K_tr, y, n_classes)
    return 1.0 - P_loo[np.arange(len(y)), y], n_zero


def gp_loo_precision(K_tr: np.ndarray, sigma2: float) -> np.ndarray:
    """Q = (K + sigma2 I)^-1 via a Cholesky factorisation.

    The frozen-covariance GP LOO identities (paper Eq. A.20) need only Q, and Q
    does not depend on the labels, so it can be reused across label sets.
    """
    n = len(K_tr)
    factor = cho_factor(K_tr + sigma2 * np.eye(n), lower=True)
    return cho_solve(factor, np.eye(n))


def gp_loo_score_multiclass(
    Q: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Multiclass GP LOO score (paper Eq. A.20) on one-hot targets.

    For each class c: mu_ic = Y_ic - (Q Y)_ic / Q_ii and v_i = 1 / Q_ii.
    Returns L_i = mean_c [-log N(Y_ic | mu_ic, v_i)] (higher = harder).
    """
    Y = np.eye(n_classes)[np.asarray(y_train, dtype=int)]
    d = np.diag(Q)
    mu = Y - (Q @ Y) / d[:, None]
    v = 1.0 / d
    nll = 0.5 * np.log(2.0 * np.pi * v)[:, None] + 0.5 * (Y - mu) ** 2 / v[:, None]
    return nll.mean(axis=1)


def true_class_proba(proba: np.ndarray, classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability each row assigns to its own label y; 0 if y is not in `classes`."""
    classes = np.asarray(classes)
    col = np.clip(np.searchsorted(classes, y), 0, len(classes) - 1)
    known = classes[col] == y
    return np.where(known, proba[np.arange(len(y)), col], 0.0)


def crossfit_difficulty(
    clf_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    random_state: int = 42,
    eps: float = 1e-12,
    return_probs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Black-box K-fold cross-fitted out-of-fold NLL surrogate score.
    Used when foundation model internal embeddings or heads are inaccessible.

    clf_factory is either a function returning a fresh classifier or an unfitted
    scikit-learn estimator (cloned for every fold). Works for any number of classes.
    With return_probs=True, also returns the out-of-fold probability of the true label.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p_true = np.zeros(len(y), dtype=float)

    for tr_idx, oof_idx in skf.split(X, y):
        clf = clone(clf_factory) if hasattr(clf_factory, "fit") else clf_factory()
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[oof_idx])
        p_true[oof_idx] = true_class_proba(probs, clf.classes_, y[oof_idx])

    L = -np.log(np.clip(p_true, eps, 1.0))
    if return_probs:
        return L, p_true
    return L
