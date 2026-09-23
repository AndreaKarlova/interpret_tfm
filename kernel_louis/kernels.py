"""Kernel functions and bandwidth heuristics for KernelICL and Kernel-LOUIS."""

import numpy as np


def default_gamma(d_k: int) -> float:
    """Default Gaussian kernel scale (Miftachov et al. 2026, Eq. 17):
    gamma = 1 / (2 * sqrt(d_k)).
    """
    return float(1.0 / (2.0 * np.sqrt(max(d_k, 1))))


def median_heuristic_gamma(X: np.ndarray, max_samples: int = 1000, random_state: int = 42) -> float:
    """Compute Gaussian kernel scale using the median heuristic:
    gamma = 1 / (2 * median(||x_i - x_j||^2)).
    """
    n = len(X)
    if n <= 1:
        return 1.0
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        sub = X[idx]
    else:
        sub = X

    # Compute pairwise squared Euclidean distances
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    triu_idx = np.triu_indices(len(sub), k=1)
    med = np.median(d2[triu_idx])
    if med <= 0:
        return 1.0
    return float(1.0 / (2.0 * med))


def gaussian_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """Vectorized Gaussian RBF kernel matrix:
    K(q, k) = exp(-gamma * ||q - k||^2).

    Parameters
    ----------
    A : (m, d) array of query/point vectors
    B : (n, d) array of key/point vectors
    gamma : kernel bandwidth scale parameter

    Returns
    -------
    K : (m, n) kernel Gram matrix
    """
    a2 = np.sum(A * A, axis=1)[:, None]  # (m, 1)
    b2 = np.sum(B * B, axis=1)[None, :]  # (1, n)
    d2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
    return np.exp(-gamma * d2)
