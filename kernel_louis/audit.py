r"""Context Influence Audit (Section 5, Eq. 12).

Decomposes actual deletion effect into:
1. Frozen-head effect (our closed-form LOO).
2. Representation-mediated discrepancy (discarded by the frozen-embedding assumption).
"""

from typing import Dict
import numpy as np
from kernel_louis.heads import self_normalized_predict


def audit_kernel_head_influence(
    K_qt: np.ndarray,
    K_tr: np.ndarray,
    y_train: np.ndarray,
    del_idx: int,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    r"""Audit the deletion influence on a fixed-kernel head (where representation discrepancy is 0).

    For query points x:
    actual deletion effect = f_D(x) - f_{D \setminus i}(x)
    frozen head closed-form = w_i(x; D) / (1 - w_i(x; D)) * (y_i - f_D(x))

    Parameters
    ----------
    K_qt : (m, n) query-train kernel
    K_tr : (n, n) train-train kernel
    y_train : (n,) train labels
    del_idx : index i of the deleted context row

    Returns
    -------
    dict containing:
        - actual_deletion_effect: f_D(x) - f_{D \setminus i}(x)
        - closed_form_effect: predicted head-level LOO effect
        - discrepancy: actual - closed_form (identically 0 for fixed kernel)
    """
    m_queries, n_ctx = K_qt.shape
    y = np.asarray(y_train, dtype=float)

    # Full context predictions and weights
    f_D, W = self_normalized_predict(K_qt, y)
    w_i_x = W[:, del_idx]  # w_i(x; D)

    # Closed-form prediction: w_i(x; D) / (1 - w_i(x; D)) * (y_i - f_D(x))
    closed_form = (w_i_x / np.clip(1.0 - w_i_x, eps, None)) * (y[del_idx] - f_D)

    # Genuine deletion: remove column del_idx from K_qt and y
    keep = np.ones(n_ctx, dtype=bool)
    keep[del_idx] = False
    K_qt_del = K_qt[:, keep]
    y_del = y[keep]

    f_del, _ = self_normalized_predict(K_qt_del, y_del)
    actual = f_D - f_del

    discrepancy = actual - closed_form

    return {
        "actual_deletion_effect": actual,
        "closed_form_effect": closed_form,
        "discrepancy": discrepancy,
        "max_abs_discrepancy": float(np.max(np.abs(discrepancy))),
    }
