r"""Context Influence Audit (Section 4, Eq. 8, Eq. 12, and Table 1).

The fixed-kernel audits below recompute on the same kernel (E = 0 by construction:
the negative control). The frozen_*_effects functions take measured weights and are
compared with recomputed predictions from a context-dependent model such as KernelICL.

Decomposes actual context modification effects into:
1. Frozen-head effect (closed-form deletion / admission effect).
2. Representation-mediated discrepancy (discarded by the frozen-embedding assumption).
"""

from typing import Dict, Optional
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


def frozen_deletion_effects(
    P: np.ndarray,
    W: np.ndarray,
    y_ctx: np.ndarray,
    out_cls: np.ndarray,
    rows: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    r"""Frozen-head deletion effects I^head_i(x) on one output coordinate (paper Eq. 8).

    I^head_i(x) = w_i(x) / (1 - w_i(x)) * ([y_i = c_x] - p_D(c_x | x)), the change in the
    probability of class c_x = out_cls[x] when context point i is removed from the vote
    and the remaining weights are renormalised, embeddings held fixed.

    P (m, C) and W (m, n) are the full-context probabilities and kernel weights,
    y_ctx (n,) the encoded context labels, out_cls (m,) the audited class per query.
    Returns (len(rows), m), with rows = all context points by default.

    Loses precision as w_i(x) -> 1 (division by 1 - w_i): with sharp kernels use
    frozen_deletion_effects_log, which is exact from log-affinities.
    """
    rows = np.arange(W.shape[1]) if rows is None else np.asarray(rows)
    out_cls = np.asarray(out_cls)
    p_c = P[np.arange(len(out_cls)), out_cls]                       # (m,)
    w = W[:, rows].T                                                 # (r, m)
    hit = (np.asarray(y_ctx)[rows][:, None] == out_cls[None, :])     # (r, m)
    return w / np.clip(1.0 - w, eps, None) * (hit - p_c[None, :])


def frozen_deletion_vectors(P: np.ndarray, W: np.ndarray, y_ctx: np.ndarray,
                            rows: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Eq. 8 for the whole probability vector: (len(rows), m, C), for size measures such as TV."""
    n_classes = P.shape[1]
    w = W[:, rows].T[:, :, None]                                     # (r, m, 1)
    e = np.eye(n_classes)[np.asarray(y_ctx)[rows]][:, None, :]       # (r, 1, C)
    return w / np.clip(1.0 - w, eps, None) * (e - P[None, :, :])


def frozen_deletion_proba_log(
    log_K: np.ndarray,
    y_ctx: np.ndarray,
    n_classes: int,
    rows: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Frozen-head predictions after deleting each context point, computed stably (paper Eq. 5).

    p^fr_{D^-i|D}(c | x) = sum_{j != i, y_j = c} k_j(x) / sum_{j != i} k_j(x), from log-affinities
    log_K (m, n). Unlike Eq. 8 written with weights, this never divides by 1 - w_i(x), so it stays
    exact when one point carries (numerically) all of a query's weight. Each row is scaled by its
    largest affinity; deleting any other point leaves a denominator >= 1, and deleting the largest
    one is recomputed directly from the remaining points. Returns (len(rows), m, n_classes).
    """
    y = np.asarray(y_ctx, dtype=int)
    m, n = log_K.shape
    rows = np.arange(n) if rows is None else np.asarray(rows)
    top = np.argmax(log_K, axis=1)
    K = np.exp(log_K - log_K[np.arange(m), top][:, None])       # largest entry per row is 1
    Y = np.eye(n_classes)[y]
    A, Z = K @ Y, K.sum(axis=1)                                   # (m, C), (m,)
    k = K[:, rows].T[:, :, None]                                  # (r, m, 1)
    e = Y[rows][:, None, :]                                       # (r, 1, C)
    with np.errstate(divide="ignore", invalid="ignore"):  # only the top point's entries can be 0/0;
        out = (A[None] - k * e) / (Z[None, :, None] - k)   # they are recomputed below
    for x in range(m):
        hit = np.where(rows == top[x])[0]
        if len(hit) > 0:
            keep = np.arange(n) != top[x]
            logits = log_K[x, keep]
            w = np.exp(logits - logits.max())
            out[hit[0], x] = (w @ Y[keep]) / w.sum()
    return out


def frozen_deletion_effects_log(log_K: np.ndarray, y_ctx: np.ndarray, out_cls: np.ndarray,
                                n_classes: int, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Stable frozen deletion effects I^head_i(x) on one coordinate: p_D(c_x | x) - p^fr_{D^-i|D}(c_x | x).

    Same quantity as frozen_deletion_effects, without its loss of precision as w_i(x) -> 1.
    Returns (len(rows), m).
    """
    out_cls = np.asarray(out_cls)
    m = len(out_cls)
    p_full = frozen_deletion_full(log_K, y_ctx, n_classes)[np.arange(m), out_cls]
    p_del = frozen_deletion_proba_log(log_K, y_ctx, n_classes, rows)[:, np.arange(m), out_cls]
    return p_full[None, :] - p_del


def frozen_deletion_full(log_K: np.ndarray, y_ctx: np.ndarray, n_classes: int) -> np.ndarray:
    """Full-context kernel-vote probabilities (m, C) from log-affinities, computed stably."""
    K = np.exp(log_K - log_K.max(axis=1, keepdims=True))
    return (K @ np.eye(n_classes)[np.asarray(y_ctx, dtype=int)]) / K.sum(axis=1, keepdims=True)


def frozen_relabel_effects(W: np.ndarray, y_old: np.ndarray, y_new: np.ndarray,
                           out_cls: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Frozen label-replacement effects, after minus before (paper App. C.3).

    Replacing y_i by v with the weights fixed changes the probability of class c by
    w_i(x) ([v = c] - [y_i = c]). Returns (len(rows), m).
    """
    out_cls = np.asarray(out_cls)
    new_hit = np.asarray(y_new)[:, None] == out_cls[None, :]
    old_hit = np.asarray(y_old)[:, None] == out_cls[None, :]
    return W[:, rows].T * (new_hit.astype(float) - old_hit)


def frozen_admission_effects(log_k_z: np.ndarray, log_Z: np.ndarray, P: np.ndarray,
                             v: np.ndarray, out_cls: np.ndarray) -> np.ndarray:
    r"""Frozen admission effects a_z(x) ([v = c] - p_D(c | x)), after minus before (paper Eq. 11-12).

    a_z(x) = k(x, z) / (Z_D(x) + k(x, z)), computed in log space: log_k_z (n_cand, m) are
    log affinities between each candidate and each query, log_Z (m,) the log total
    affinity of each query to the context. Returns (n_cand, m).
    """
    out_cls = np.asarray(out_cls)
    a = np.exp(log_k_z - np.logaddexp(log_Z[None, :], log_k_z))
    p_c = P[np.arange(len(out_cls)), out_cls]
    hit = np.asarray(v)[:, None] == out_cls[None, :]
    return a * (hit - p_c[None, :])


def compute_audit_metrics(
    actual_influence: np.ndarray,
    head_influence: np.ndarray,
    tol: float = 1e-4,
    top_k: int = 5,
) -> Dict[str, float]:
    r"""Audit metrics of paper Eq. 10, Table 1 and App. C.3.

    Inputs are the recomputed ("actual") and frozen-head effects, either (n_points,) for a
    single query or (n_points, n_queries).

    - bias, rms: mean and root-mean-square of E = actual - head (Eq. 10)
    - rms_actual: RMS of the actual effects (the scale E is compared with); relative_rms = rms / rms_actual
    - sign_agreement: fraction of entries with |actual| > tol whose signs agree
      (only the *true* effect is thresholded, App. C.3)
    - top_k_overlap: per query, overlap of the k context points with the largest
      |actual| and |head| effects, averaged over queries (App. C.3)
    """
    actual = np.asarray(actual_influence, dtype=float)
    head = np.asarray(head_influence, dtype=float)
    if actual.ndim == 1:
        actual, head = actual[:, None], head[:, None]

    E = actual - head
    bias = float(np.mean(E))
    rms = float(np.sqrt(np.mean(E ** 2)))
    rms_actual = float(np.sqrt(np.mean(actual ** 2)))

    active = np.abs(actual) > tol
    sign_agree = float(np.mean(np.sign(actual[active]) == np.sign(head[active]))) if active.any() else float("nan")

    k = min(top_k, actual.shape[0])
    overlaps = []
    for q in range(actual.shape[1]):
        top_actual = set(np.argsort(-np.abs(actual[:, q]), kind="stable")[:k])
        top_head = set(np.argsort(-np.abs(head[:, q]), kind="stable")[:k])
        overlaps.append(len(top_actual & top_head) / k)

    return {
        "bias": bias,
        "rms": rms,
        "rms_actual": rms_actual,
        "relative_rms": rms / rms_actual if rms_actual > 0 else float("nan"),
        "sign_agreement": sign_agree,
        "top_k_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "max_abs_error": float(np.max(np.abs(E))) if E.size else 0.0,
    }
