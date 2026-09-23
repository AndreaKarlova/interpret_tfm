"""Evaluation metrics for subpopulation shift, detection AUC, and interpretability."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_group_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    is_minority: np.ndarray,
) -> Dict[str, float]:
    """Compute subpopulation accuracy metrics: mean, majority, minority, worst-group."""
    y_p = np.asarray(y_pred, dtype=int)
    y_t = np.asarray(y_true, dtype=int)
    m = np.asarray(is_minority, dtype=bool)

    correct = y_p == y_t
    mean_acc = float(np.mean(correct))
    maj_acc = float(np.mean(correct[~m])) if np.any(~m) else 0.0
    mino_acc = float(np.mean(correct[m])) if np.any(m) else 0.0
    worst_acc = min(maj_acc, mino_acc)

    return {
        "mean": mean_acc,
        "majority": maj_acc,
        "minority": mino_acc,
        "worst": worst_acc,
    }


def compute_tail_precision_recall(
    L: np.ndarray,
    is_minority: np.ndarray,
    tau_list: List[float] = [0.80, 0.90, 0.95, 0.97],
) -> List[Dict[str, float]]:
    """Compute tail size, precision, and recall for Table 2."""
    L = np.asarray(L)
    m = np.asarray(is_minority, dtype=bool)
    n_total = len(L)
    n_minority = int(np.sum(m))

    rows = []
    for tau in tau_list:
        thr = np.quantile(L, tau)
        flagged = L >= thr
        tail_size = int(np.sum(flagged))
        true_positives = int(np.sum(m[flagged]))
        precision = float(true_positives / tail_size) if tail_size > 0 else 0.0
        recall = float(true_positives / n_minority) if n_minority > 0 else 0.0
        rows.append({
            "tau": float(tau),
            "tail_size": tail_size,
            "precision": precision,
            "recall": recall,
        })
    return rows


def weight_perplexity(W: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Inspectability metric: PPL(w) = exp(-sum_i w_i log w_i) per query row.
    Lower = sparser = easier to inspect.
    """
    w = np.clip(W, eps, 1.0)
    entropy = -np.sum(W * np.log(w), axis=1)
    ppl = np.exp(entropy)
    rel_ppl = ppl / float(W.shape[1])
    return ppl, rel_ppl
