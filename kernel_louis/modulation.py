"""Bounded Modulation and Prompt Resampling for Kernel-LOUIS.

Implements:
1. Tail-set multiplier (Eq. 8, recommended): m_i = 1 + lam * 1[L_i >= Q_tau(L)]
2. Smooth linear multiplier (Eq. 9): m_i = 1 + lam * L_tilde_i
3. Exponential multiplier (Eq. 10, unstable cautionary): m_i = exp(gamma * L_tilde_i)
4. Pseudo-group balancing (group-free analogue of DFR, Section 3.2)
5. Context resampling (Eq. 11, data-centric surrogate with diversity capping)
"""

from typing import Optional, Tuple
import numpy as np


def min_max_normalize(L: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize difficulty scores into [0, 1]."""
    L_min, L_max = np.min(L), np.max(L)
    denom = max(L_max - L_min, eps)
    return (L - L_min) / denom


def tail_multiplier(L: np.ndarray, lam: float, tau: float = 0.80) -> np.ndarray:
    """Bounded, tail-targeted multiplier (Eq. 8):
        m_i = 1 + lam * 1[L_i >= Q_tau(L)]

    Parameters
    ----------
    L : (n,) difficulty scores
    lam : upweighting strength parameter (lam=0 recovers baseline m_i = 1)
    tau : quantile threshold in [0, 1]

    Returns
    -------
    m : (n,) non-negative sample multipliers
    """
    if lam <= 0:
        return np.ones_like(L, dtype=float)
    thr = np.quantile(L, tau)
    return 1.0 + float(lam) * (L >= thr).astype(float)


def smooth_multiplier(L: np.ndarray, lam: float) -> np.ndarray:
    """Smooth linear multiplier (Eq. 9):
        m_i = 1 + lam * L_tilde_i
    where L_tilde_i is the min-max normalized difficulty score in [0, 1].
    """
    if lam <= 0:
        return np.ones_like(L, dtype=float)
    L_norm = min_max_normalize(L)
    return 1.0 + float(lam) * L_norm


def exponential_multiplier(L: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Exponential multiplier (Eq. 10, cautionary/unstable):
        m_i = exp(gamma * L_tilde_i)
    Over-concentrates mass on the extreme tail and collapses to chance.
    """
    L_norm = min_max_normalize(L)
    return np.exp(float(gamma) * L_norm)


def pseudo_group_balance(
    y: np.ndarray,
    L: np.ndarray,
    tau: float = 0.95,
) -> np.ndarray:
    """Pseudo-group balancing (Section 3.2):
    Forms 4 cells from (class y in {0, 1} x LOO-hard in {0, 1}) and balances them,
    serving as the group-label-free analogue of Deep Feature Reweighting (DFR).

    Returns
    -------
    weights : (n,) sample weights balancing the 4 pseudo-groups
    """
    y = np.asarray(y, dtype=int)
    thr = np.quantile(L, tau)
    hard = (L >= thr).astype(int)
    pseudo_groups = 2 * y + hard

    weights = np.zeros(len(y), dtype=float)
    for pg in range(4):
        mask = pseudo_groups == pg
        cnt = mask.sum()
        if cnt > 0:
            weights[mask] = 1.0 / cnt
    # Normalize so mean weight is 1.0
    return weights / (np.mean(weights) + 1e-12)


def resample_context(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    size: Optional[int] = None,
    max_mult: Optional[int] = None,
    target_unique: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample in-context prompt with replacement prop. to weights (Eq. 11).

    Includes optional multiplicity cap `max_mult` to prevent collapsing context
    diversity into duplicates of a tiny handful of points.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y)
    size = size or n
    w = np.asarray(weights, dtype=float)
    p = w / np.sum(w)

    if max_mult is None and target_unique is not None:
        max_mult = int(np.clip(round(1.0 / max(target_unique, 1e-3)), 2, 8))

    if max_mult is None or max_mult >= size:
        idx = rng.choice(n, size=size, replace=True, p=p)
        return X[idx], y[idx]

    # Pool sampling with diversity cap
    pool = rng.choice(n, size=6 * size, replace=True, p=p)
    counts = {}
    keep = []
    for i in pool:
        c = counts.get(i, 0)
        if c < max_mult:
            counts[i] = c + 1
            keep.append(i)
        if len(keep) >= size:
            break
    idx = np.array(keep, dtype=int)
    return X[idx], y[idx]
