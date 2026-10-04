"""Bounded Modulation and Prompt Resampling for Kernel-LOUIS.

Implements:
1. Tail-set multiplier (Eq. 8, recommended): m_i = 1 + lam * 1[L_i >= Q_tau(L)]
2. Smooth linear multiplier (Eq. 9): m_i = 1 + lam * L_tilde_i
3. Exponential multiplier (Eq. 10, unstable cautionary): m_i = exp(gamma * L_tilde_i)
4. Pseudo-group and oracle group balancing (group-free / group-labelled DFR analogues)
5. Random-tail and pruning rules, plus their random controls
6. Context resampling (Eq. 11, data-centric surrogate with diversity capping)
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


def group_balance_multiplier(groups: np.ndarray) -> np.ndarray:
    """Balance arbitrary groups: m_i = n / (#non-empty groups * |group of i|).

    Every non-empty group receives the same total multiplier mass, and the
    multipliers have mean 1. With true group labels this is the oracle balance.
    """
    groups = np.asarray(groups)
    labels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return len(groups) / (len(labels) * counts[inverse].astype(float))


def pseudo_group_balance(
    y: np.ndarray,
    L: np.ndarray,
    tau: float = 0.95,
) -> np.ndarray:
    """Pseudo-group balancing (Section 3.2):
    Forms cells from (class y x LOO-hard in {0, 1}) and balances them, serving
    as the group-label-free analogue of Deep Feature Reweighting (DFR).
    Works for any number of classes; empty cells are ignored.

    Returns
    -------
    weights : (n,) sample weights with mean 1, equal total weight per non-empty cell
    """
    y = np.asarray(y, dtype=int)
    hard = (L >= np.quantile(L, tau)).astype(int)
    return group_balance_multiplier(2 * y + hard)


def random_tail_multiplier(
    n: int,
    tail_size: int,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Control for tail_multiplier: m_i = 1 + lam on `tail_size` uniformly random points."""
    m = np.ones(n)
    m[rng.choice(n, size=tail_size, replace=False)] += float(lam)
    return m


def top_fraction_mask(L: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of the round(q * n) highest scores (ties broken by lower index first)."""
    n = len(L)
    k = int(round(q * n))
    order = np.argsort(-np.asarray(L), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def prune_multiplier(L: np.ndarray, q: float) -> np.ndarray:
    """Pruning: m_i = 0 for the round(q * n) highest-score points, 1 otherwise."""
    return np.where(top_fraction_mask(L, q), 0.0, 1.0)


def random_prune_multiplier(n: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """Control for prune_multiplier: m_i = 0 for round(q * n) uniformly random points."""
    m = np.ones(n)
    m[rng.choice(n, size=int(round(q * n)), replace=False)] = 0.0
    return m


def surprisal_from_score(L: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """s_i = -log p_i with p_i = 1 - L_i clipped to [eps, 1], for scores of the form L = 1 - p."""
    return -np.log(np.clip(1.0 - np.asarray(L, dtype=float), eps, 1.0))


def resample_context(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    size: Optional[int] = None,
    max_mult: Optional[int] = None,
    target_unique: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    return_indices: bool = False,
):
    """Sample in-context prompt with replacement prop. to weights (Eq. 11).

    Includes optional multiplicity cap `max_mult` to prevent collapsing context
    diversity into duplicates of a tiny handful of points. Pass
    target_unique=None and max_mult=None for plain sampling with replacement.
    With return_indices=True, also returns the sampled row indices.
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
        if return_indices:
            return X[idx], y[idx], idx
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
    if return_indices:
        return X[idx], y[idx], idx
    return X[idx], y[idx]
