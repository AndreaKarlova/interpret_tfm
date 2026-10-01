r"""Context Influence Audit (Section 4, Eq. 8, Eq. 12, and Table 1).

Decomposes actual context modification effects into:
1. Frozen-head effect (closed-form deletion / admission effect).
2. Representation-mediated discrepancy (discarded by the frozen-embedding assumption).
"""

from typing import Dict, Optional, Set
import numpy as np
from kernel_louis.heads import self_normalized_predict


def audit_kernel_head_influence(
    K_qt: np.ndarray,
    K_tr: np.ndarray,
    y_train: np.ndarray,
    del_idx: int,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    r"""Audit the deletion influence on a normalized kernel head (Section 4.1, Eq. 8).

    For query points x:
    actual deletion effect:
        I_i(x; D) = f_D(x) - f_{D \setminus i}(x)
    frozen head closed-form explanation:
        I_i^{head}(x; D) = \frac{w_i(x; D)}{1 - w_i(x; D)} (y_i - f_D(x))
    signed audit error:
        E_i(x; D) = I_i(x; D) - I_i^{head}(x; D) = f_{D \setminus i | D}^{fr}(x) - f_{D \setminus i}(x)

    Parameters
    ----------
    K_qt : (m, n) query-train kernel
    K_tr : (n, n) train-train kernel
    y_train : (n,) train labels
    del_idx : index i of the deleted context row
    eps : numerical stability constant

    Returns
    -------
    dict containing:
        - actual_deletion_effect: f_D(x) - f_{D \setminus i}(x)
        - closed_form_effect: predicted head-level LOO effect I_i^{head}
        - discrepancy: E_i = actual - closed_form (identically 0 for fixed kernel)
        - max_abs_discrepancy: max absolute error
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


def audit_context_admission(
    K_qt: np.ndarray,
    k_qz: np.ndarray,
    y_train: np.ndarray,
    candidate_label: float,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    r"""Audit context admission influence (Section 4.3, Eq. 11, Eq. 12, Appendix C.4).

    Appends candidate row (z, v) to context D: D^{+z, v} = D \oplus (z, v).
    The candidate's frozen normalized weight is:
        a_z(x; D) = \frac{k_D(x, z)}{Z_D(x) + k_D(x, z)}
    The corresponding addition effect decomposes as:
        f_{D^{+z, v}}(x) - f_D(x) = a_z(x; D)[v - f_D(x)] + \rho_z(x; v)

    Parameters
    ----------
    K_qt : (m, n) query-train kernel matrix
    k_qz : (m,) or (m, 1) kernel affinities between queries and candidate z
    y_train : (n,) existing context labels
    candidate_label : float, candidate label v
    eps : numerical stability constant

    Returns
    -------
    dict containing:
        - actual_addition_effect: f_{D^{+z, v}}(x) - f_D(x)
        - closed_form_addition: a_z(x; D)[v - f_D(x)]
        - remainder: \rho_z(x; v) (identically 0 on fixed features)
        - max_abs_remainder: max absolute remainder
    """
    y = np.asarray(y_train, dtype=float)
    v = float(candidate_label)
    k_z = np.asarray(k_qz, dtype=float).ravel()

    # Original context prediction
    f_D, W = self_normalized_predict(K_qt, y)
    Z_D = np.sum(K_qt, axis=1)

    # Frozen normalized weight
    a_z = k_z / np.clip(Z_D + k_z, eps, None)
    closed_form_addition = a_z * (v - f_D)

    # Recomputed context prediction with appended column
    K_plus = np.column_stack([K_qt, k_z])
    y_plus = np.append(y, v)
    f_plus, _ = self_normalized_predict(K_plus, y_plus)

    actual_addition = f_plus - f_D
    remainder = actual_addition - closed_form_addition

    return {
        "actual_addition_effect": actual_addition,
        "closed_form_addition": closed_form_addition,
        "remainder": remainder,
        "max_abs_remainder": float(np.max(np.abs(remainder))),
    }


def compute_audit_metrics(
    actual_influence: np.ndarray,
    head_influence: np.ndarray,
    tol: float = 1e-4,
    top_k: int = 5,
) -> Dict[str, float]:
    r"""Compute Matched-Intervention Protocol metrics (Table 1 & Eq. 10).

    Metrics:
    - Signed error E_i = actual - head
    - Bias = (1/|A|) \sum E_i
    - RMS = \sqrt{(1/|A|) \sum E_i^2}
    - Sign agreement = fraction where sign(actual) == sign(head) (with tol)
    - Top-k overlap = Jaccard / fraction overlap of largest absolute influences
    """
    actual = np.asarray(actual_influence, dtype=float).ravel()
    head = np.asarray(head_influence, dtype=float).ravel()

    E = actual - head
    bias = float(np.mean(E))
    rms = float(np.sqrt(np.mean(E ** 2)))

    # Sign agreement on non-trivial effects
    active = (np.abs(actual) > tol) | (np.abs(head) > tol)
    if np.any(active):
        sign_agree = float(np.mean(np.sign(actual[active]) == np.sign(head[active])))
    else:
        sign_agree = 1.0

    # Top-k overlap
    k = min(top_k, len(actual))
    if k > 0:
        idx_act: Set[int] = set(np.argsort(np.abs(actual))[-k:])
        idx_head: Set[int] = set(np.argsort(np.abs(head))[-k:])
        top_k_overlap = float(len(idx_act & idx_head) / k)
    else:
        top_k_overlap = 1.0

    return {
        "bias": bias,
        "rms": rms,
        "sign_agreement": sign_agree,
        "top_k_overlap": top_k_overlap,
        "max_abs_error": float(np.max(np.abs(E))) if len(E) > 0 else 0.0,
    }

