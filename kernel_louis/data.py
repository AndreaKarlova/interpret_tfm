"""Data generators for synthetic benchmarks and subpopulation shift.

Implements:
1. Tabular Waterbirds generator (weak core + strong spurious feature, Section 3.1 & 3.2).
2. Spurious Two-Moons generator (two-moons + shortcut feature, Section 3.3).
3. Pure Moons with label noise injection (2-D and 20-D regimes, Section 3.4).
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.datasets import make_moons


def make_tabular_waterbirds(
    n_samples: int = 1000,
    spurious_corr: float = 0.90,
    core_noise: float = 1.1,
    spurious_strength: float = 2.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tabular analogue of Waterbirds (Section 3.1 & 3.2).

    Structure:
    - Binary label y in {0, 1} balanced 50/50.
    - Weak core feature tracking the label for all points:
        x_core = (2*y - 1) + N(0, core_noise^2)
    - Strong spurious attribute a in {0, 1}:
        P(a = y) = spurious_corr (e.g. 0.90 or 0.95)
    - Continuous spurious feature:
        x_spurious = (2*a - 1) * spurious_strength + N(0, 0.1^2)

    Returns
    -------
    X : (n, 2) features [x_core, x_spurious]
    y : (n,) binary label in {0, 1}
    groups : (n,) group indices in {0, 1, 2, 3} where g = 2*y + a
    is_minority : (n,) boolean mask indicating minority points (y != a)
    """
    rng = np.random.default_rng(random_state)
    y = rng.integers(0, 2, size=n_samples, dtype=np.int64)

    # Stratified spurious attribute assignment
    spurious_attr = np.empty(n_samples, dtype=np.int64)
    for label in (0, 1):
        mask = y == label
        n_label = mask.sum()
        n_maj = int(round(spurious_corr * n_label))
        assign = np.concatenate([
            np.full(n_maj, label, dtype=np.int64),
            np.full(n_label - n_maj, 1 - label, dtype=np.int64),
        ])
        rng.shuffle(assign)
        spurious_attr[mask] = assign

    x_core = (2.0 * y - 1.0) + rng.normal(0, core_noise, size=n_samples)
    x_spurious = (2.0 * spurious_attr - 1.0) * spurious_strength + rng.normal(0, 0.1, size=n_samples)

    X = np.column_stack([x_core, x_spurious]).astype(np.float32)
    groups = (2 * y + spurious_attr).astype(np.int64)
    is_minority = y != spurious_attr

    return X, y, groups, is_minority


def make_spurious_moons(
    n_samples: int,
    spurious_corr: float = 0.90,
    noise: float = 0.20,
    spurious_strength: float = 3.0,
    n_noise_features: int = 0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Two-moons with a controllable spurious correlation (Section 3.3).

    Returns
    -------
    X : (n, 3+n_noise_features) float32 array [moon_x, moon_y, spurious, ...]
    y : (n,) binary label in {0, 1}
    groups : (n,) group index in {0, 1, 2, 3}
    metadata : dict with minority counts and summary info
    """
    local_rng = np.random.default_rng(random_state)
    X_core, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=int(local_rng.integers(1 << 30)),
    )

    spurious_attr = np.empty(n_samples, dtype=np.int64)
    for label in (0, 1):
        mask = y == label
        n_label = mask.sum()
        n_majority = int(round(spurious_corr * n_label))
        n_minority = n_label - n_majority
        assignments = np.concatenate([
            np.full(n_majority, label, dtype=np.int64),
            np.full(n_minority, 1 - label, dtype=np.int64),
        ])
        local_rng.shuffle(assignments)
        spurious_attr[mask] = assignments

    spurious_feature = (spurious_attr * 2.0 - 1.0) * spurious_strength
    spurious_feature += local_rng.normal(0, 0.15, n_samples)

    X = np.column_stack([X_core, spurious_feature])
    if n_noise_features > 0:
        X = np.column_stack([X, local_rng.normal(0, 1, (n_samples, n_noise_features))])

    groups = (2 * y + spurious_attr).astype(np.int64)
    is_minority = y != spurious_attr

    group_counts = {g: int((groups == g).sum()) for g in range(4)}
    metadata = dict(
        spurious_attr=spurious_attr,
        is_minority=is_minority,
        group_counts=group_counts,
        n_minority=int(is_minority.sum()),
        n_majority=int((~is_minority).sum()),
        spurious_corr=spurious_corr,
    )
    return X.astype(np.float32), y.astype(np.int64), groups, metadata


def make_spurious_split(
    seed: int,
    n_train: int = 2000,
    n_test: int = 2000,
    n_noise_features: int = 0,
    corr_train: float = 0.90,
    corr_test: float = 0.50,
    noise: float = 0.20,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split on spurious moons with correlation shift."""
    Xtr, ytr, gtr, m_tr = make_spurious_moons(
        n_train, corr_train, noise=noise,
        n_noise_features=n_noise_features, random_state=seed,
    )
    Xte, yte, gte, m_te = make_spurious_moons(
        n_test, corr_test, noise=noise,
        n_noise_features=n_noise_features, random_state=seed + 10_000,
    )
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
    return (Xtr, ytr, gtr, m_tr["is_minority"],
            Xte, yte, gte, m_te["is_minority"])


def make_noisy_label_split(
    seed: int,
    n_train: int = 600,
    n_test: int = 1200,
    n_noise_dims: int = 0,
    eps_noise: float = 0.20,
    noise: float = 0.20,
    standardize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure two-moons with context label noise (Section 3.4, Table 5).

    When n_noise_dims=0: 2-D features [moon_x, moon_y].
    When n_noise_dims=18: 20-D features [moon_x, moon_y, 18 i.i.d. Gaussian noise dims].

    Returns
    -------
    Xtr : (n_train, d) training context features
    ytr : (n_train,) clean training labels
    yn : (n_train,) noisy training labels with eps_noise fraction flipped
    flip : (n_train,) boolean mask of truly flipped points
    Xte : (n_test, d) test features
    yte : (n_test,) clean test labels
    """
    rng = np.random.default_rng(seed)
    Xtr_core, ytr = make_moons(
        n_samples=n_train, noise=noise,
        random_state=int(rng.integers(1 << 30)),
    )
    Xte_core, yte = make_moons(
        n_samples=n_test, noise=noise,
        random_state=int(rng.integers(1 << 30)),
    )

    if n_noise_dims > 0:
        Xtr = np.column_stack([Xtr_core, rng.normal(0, 1, (n_train, n_noise_dims))])
        Xte = np.column_stack([Xte_core, rng.normal(0, 1, (n_test, n_noise_dims))])
    else:
        Xtr, Xte = Xtr_core, Xte_core

    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)

    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd

    rng_flip = np.random.default_rng(seed + 777)
    flip = rng_flip.random(len(ytr)) < eps_noise
    yn = ytr.copy()
    yn[flip] = 1 - yn[flip]

    return Xtr, ytr, yn, flip, Xte, yte
