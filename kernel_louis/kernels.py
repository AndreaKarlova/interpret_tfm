"""Kernel functions and bandwidth heuristics for KernelICL and Kernel-LOUIS."""

from typing import Optional

import numpy as np


def default_gamma(d_k: int) -> float:
    """Default Gaussian kernel scale (Miftachov et al. 2026, Eq. 17):
    gamma = 1 / (2 * sqrt(d_k)).
    """
    return float(1.0 / (2.0 * np.sqrt(max(d_k, 1))))


def median_heuristic_gamma(
    X: np.ndarray,
    max_samples: Optional[int] = 1000,
    random_state: int = 42,
    factor: float = 2.0,
) -> float:
    """Compute Gaussian kernel scale using the median heuristic:
    gamma = 1 / (factor * median of nonzero ||x_i - x_j||^2).

    factor=2 is the package default; factor=1 gives gamma = 1 / median.
    max_samples=None uses every point instead of a random subsample.
    """
    n = len(X)
    if n <= 1:
        return 1.0
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        sub = X[idx]
    else:
        sub = X

    d2 = squared_distances(sub, sub)
    d2 = d2[np.triu_indices(len(sub), k=1)]
    d2 = d2[d2 > 0]  # exact duplicates carry no scale information
    if len(d2) == 0:
        return 1.0
    return float(1.0 / (factor * np.median(d2)))


def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances ||a - b||^2, shape (len(A), len(B))."""
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def log_gaussian_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """Log of the Gaussian RBF kernel, -gamma * ||a - b||^2 (never underflows)."""
    return -gamma * squared_distances(A, B)


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
    return np.exp(log_gaussian_kernel(A, B, gamma))
