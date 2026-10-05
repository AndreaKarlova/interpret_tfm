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

from kernel_louis.heads import gp_predict
from kernel_louis.kernels import squared_distances
from threadpoolctl import threadpool_limits


def incontext_loo_nll(
    K_tr: np.ndarray,
    y_train: np.ndarray,
    eps: float = 1e-12,
    return_probs: bool = False,
    max_threads: Optional[int] = None,
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


def loo_kernel_from_log(log_K_tr: np.ndarray) -> np.ndarray:
    """Context kernel for the frozen LOO, computed stably from log-affinities.

    Returns exp(log K) with a zero diagonal and each row divided by its largest
    off-diagonal entry. The row scaling cancels in the renormalised frozen LOO
    (paper Eq. A.13), so frozen_loo_proba gives the same result as with exp(log K),
    but a narrow bandwidth no longer underflows to all-zero rows.
    """
    L = np.array(log_K_tr, dtype=float, copy=True)
    np.fill_diagonal(L, -np.inf)
    L -= L.max(axis=1, keepdims=True)
    return np.exp(L)


def select_gamma_loo(
    sq_dists: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    gammas: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """Gaussian bandwidth chosen by leave-one-out likelihood on the context.

    For each gamma, K = exp(-gamma * sq_dists) and the frozen LOO probability of
    every point's own label (paper Eq. 9 / A.13) is computed. Returns the gamma
    with the largest mean log p^{fr,-i}(y_i), and that mean for every gamma.
    Uses only the context labels: this is leave-one-out cross-validation of the
    normalised kernel head's bandwidth.
    """
    y = np.asarray(y_train, dtype=int)
    rows = np.arange(len(y))
    loo_ll = np.zeros(len(gammas))
    for k, gamma in enumerate(gammas):
        P_loo, _ = frozen_loo_proba(loo_kernel_from_log(-gamma * sq_dists), y, n_classes)
        loo_ll[k] = np.mean(np.log(np.clip(P_loo[rows, y], eps, 1.0)))
    return float(gammas[int(np.argmax(loo_ll))]), loo_ll


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


def gp_frozen_deletion_effects(
    K_q: np.ndarray,
    Q: np.ndarray,
    alpha: np.ndarray,
    out_cls: np.ndarray,
    rows: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Frozen-covariance GP deletion effects on one output coordinate (paper App. A.6).

    With Q = (K + sigma2 I)^-1 and alpha = Q Y fixed, removing context point i changes
    the posterior mean at x by mu(x) - mu^{-i}(x) = (k_x^T Q[:, i]) alpha[i, c] / Q_ii.
    K_q (m, n) query-context covariance, out_cls (m,) the audited class per query.
    Returns (len(rows), m): effects before minus after, like deletion influence (Eq. 2).
    """
    rows = np.arange(Q.shape[0]) if rows is None else np.asarray(rows)
    B = K_q @ Q[:, rows]                                    # (m, r)
    a = alpha[rows][:, np.asarray(out_cls)]                 # (r, m)
    return B.T * a / np.diag(Q)[rows][:, None]


def gp_heldout_loglik(K_val_tr, K_tr, Y_tr, Y_val, sigma2: float) -> float:
    """Mean held-out Gaussian log-likelihood of one-hot targets under GP regression.

    Uses the predictive mean and the predictive variance of a new noisy observation
    (latent variance from gp_predict + sigma2), averaged over points and classes.
    """
    mu, v_latent = gp_predict(K_val_tr, K_tr, Y_tr, sigma2=sigma2)
    var = (v_latent + sigma2)[:, None]
    return float(np.mean(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (Y_val - mu) ** 2 / var))


def select_gp_hyperparameters_cv(folds, gammas, sigma2s) -> Tuple[float, float, np.ndarray]:
    """GP kernel scale and noise by cross-validated held-out log-likelihood.

    folds: list of (H_tr, H_val, Y_tr, Y_val) with H the (projected) embeddings of each
    fold's context and of its label-free validation queries, Y one-hot. Choosing on
    held-out queries avoids the label leak of context embeddings (paper App. A.6 caveat).
    Returns (gamma, sigma2, table (len(gammas), len(sigma2s)) of mean log-likelihoods).
    """
    table = np.zeros((len(gammas), len(sigma2s)))
    for H_tr, H_val, Y_tr, Y_val in folds:
        D_tr = squared_distances(H_tr, H_tr)
        D_val = squared_distances(H_val, H_tr)
        for a, gamma in enumerate(gammas):
            K_tr, K_val = np.exp(-gamma * D_tr), np.exp(-gamma * D_val)
            for b, sigma2 in enumerate(sigma2s):
                table[a, b] += gp_heldout_loglik(K_val, K_tr, Y_tr, Y_val, sigma2) / len(folds)
    a, b = np.unravel_index(np.argmax(table), table.shape)
    return float(gammas[a]), float(sigma2s[b]), table


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
    max_threads: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Black-box K-fold cross-fitted out-of-fold NLL surrogate score.
    Used when foundation model internal embeddings or heads are inaccessible.

    clf_factory is either a function returning a fresh classifier or an unfitted
    scikit-learn estimator (cloned for every fold). Works for any number of classes.
    With return_probs=True, also returns the out-of-fold probability of the true label.
    max_threads limits BLAS/OpenMP threads during the fits (1 is much faster for
    small models such as logistic regression when thread pools compete).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p_true = np.zeros(len(y), dtype=float)

    with threadpool_limits(limits=max_threads):
        for tr_idx, oof_idx in skf.split(X, y):
            clf = clone(clf_factory) if hasattr(clf_factory, "fit") else clf_factory()
            clf.fit(X[tr_idx], y[tr_idx])
            probs = clf.predict_proba(X[oof_idx])
            p_true[oof_idx] = true_class_proba(probs, clf.classes_, y[oof_idx])

    L = -np.log(np.clip(p_true, eps, 1.0))
    if return_probs:
        return L, p_true
    return L
