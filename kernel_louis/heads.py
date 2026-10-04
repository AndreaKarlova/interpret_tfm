"""Prediction heads for KernelICL and Kernel-LOUIS.

Implements:
1. The self-normalized Nadaraya-Watson kernel regression head (Eq. 1 & 7).
2. The deep Gaussian Process regression head (Eq. 4).
3. The censored (Tobit) likelihood predictive moments (Eq. 5-6).
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats


def self_normalized_predict(
    K_qt: np.ndarray,
    y_train: np.ndarray,
    m: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """KernelICL prediction head with optional bounded modulation (Eq. 1 & 7).

    y_hat_m(x) = sum_i m_i K(h(x), h(x_i)) y_i / sum_j m_j K(h(x), h(x_j))

    Parameters
    ----------
    K_qt : (m, n) query-vs-context kernel matrix
    y_train : (n,) binary context labels in {0, 1}, or (n, C) one-hot labels
    m : (n,) optional non-negative multipliers m_i (m_i = 1 recovers baseline)
    eps : numerical stability constant

    Returns
    -------
    proba1 : (m,) class 1 probability estimates, or (m, C) for one-hot labels
    W : (m, n) self-normalized attention weights, each row sums to 1
    """
    if m is not None:
        K_eff = K_qt * np.asarray(m, dtype=float)[None, :]
    else:
        K_eff = K_qt

    denom = np.sum(K_eff, axis=1, keepdims=True) + eps
    W = K_eff / denom
    proba1 = W @ np.asarray(y_train, dtype=float)
    return proba1, W


def kernel_head_proba(
    log_K_qt: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    m: Optional[np.ndarray] = None,
    return_weights: bool = False,
):
    """Multiclass normalised kernel head (paper Eq. 4 / Eq. 13) computed in log space.

    Each row's largest log-weight is subtracted before exponentiating, which
    leaves the normalised weights unchanged but cannot underflow. Rows whose
    weights are all zero (e.g. every multiplier m_i = 0) get uniform probabilities.

    Parameters
    ----------
    log_K_qt : (q, n) log kernel between queries and context, e.g. log_gaussian_kernel
    y_train : (n,) integer class labels in {0, ..., n_classes - 1}
    n_classes : number of classes C
    m : (n,) optional non-negative multipliers (paper Eq. 13); None means all ones

    Returns
    -------
    proba : (q, C) class probabilities
    n_zero : number of rows that fell back to uniform probabilities
    W : (q, n) normalised weights, only if return_weights=True
    """
    n = log_K_qt.shape[1]
    m = np.ones(n) if m is None else np.asarray(m, dtype=float)
    with np.errstate(divide="ignore"):
        log_W = log_K_qt + np.log(m)[None, :]  # m_i = 0 gives -inf, i.e. weight 0
    row_max = np.max(log_W, axis=1, keepdims=True)
    zero_rows = ~np.isfinite(row_max[:, 0])
    row_max[zero_rows] = 0.0
    K_shift = np.exp(log_W - row_max)

    Y = np.eye(n_classes)[np.asarray(y_train, dtype=int)]
    proba, W = self_normalized_predict(K_shift, Y)
    proba[zero_rows] = 1.0 / n_classes
    if return_weights:
        return proba, int(zero_rows.sum()), W
    return proba, int(zero_rows.sum())


def gp_predict(
    K_qt: np.ndarray,
    K_tr: np.ndarray,
    y_train: np.ndarray,
    sigma2: float = 0.1,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian Process regression head posterior (Eq. 4):
    f ~ GP(0, k), y | f ~ N(f, sigma^2).

    Parameters
    ----------
    K_qt : (m, n) query-vs-context kernel matrix
    K_tr : (n, n) context kernel matrix
    y_train : (n,) context targets
    sigma2 : noise variance ridge parameter

    Returns
    -------
    mu : (m,) posterior predictive mean
    var : (m,) posterior predictive variance
    """
    n = len(y_train)
    K_tilde = K_tr + sigma2 * np.eye(n)
    K_inv = np.linalg.inv(K_tilde)
    alpha = K_inv @ np.asarray(y_train, dtype=float)

    mu = K_qt @ alpha
    # Variance: v(x) = k(x, x) - k_x^T K_tilde^-1 k_x
    # Assuming k(x, x) = 1 (RBF kernel)
    v = np.maximum(1.0 - np.sum((K_qt @ K_inv) * K_qt, axis=1), eps)
    return mu, v


def tobit_log_likelihood(
    y: np.ndarray,
    f: np.ndarray,
    c: np.ndarray,
    sigma: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Tobit (censored-normal) log-likelihood (Eq. 5).

    c_i = -1: left-censored (y_i <= l_i) -> log Phi((l_i - f_i) / sigma)
    c_i =  0: uncensored                -> log (1/sigma * phi((y_i - f_i) / sigma))
    c_i =  1: right-censored (y_i >= u_i) -> log (1 - Phi((u_i - f_i) / sigma))
    """
    z = (y - f) / sigma
    ll = np.zeros_like(y, dtype=float)

    uncensored = c == 0
    left = c == -1
    right = c == 1

    if np.any(uncensored):
        ll[uncensored] = -np.log(sigma) + stats.norm.logpdf(z[uncensored])
    if np.any(left):
        ll[left] = stats.norm.logcdf(z[left])
    if np.any(right):
        ll[right] = stats.norm.logsf(z[right])

    return ll
