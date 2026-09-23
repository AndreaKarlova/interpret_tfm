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


def crossfit_difficulty(
    clf_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 4,
    random_state: int = 42,
    eps: float = 1e-12,
) -> np.ndarray:
    """Black-box K-fold cross-fitted out-of-fold NLL surrogate score.
    Used when foundation model internal embeddings or heads are inaccessible.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p_true = np.zeros(len(y), dtype=float)

    for tr_idx, oof_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[oof_idx])
        p1 = probs[:, 1] if probs.ndim == 2 else probs
        p_true[oof_idx] = np.where(y[oof_idx] == 1, p1, 1.0 - p1)

    return -np.log(np.clip(p_true, eps, 1.0))
