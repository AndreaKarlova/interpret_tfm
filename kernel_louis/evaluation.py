"""Evaluation metrics for subpopulation shift, detection AUC, interpretability,
and mean ± SE aggregation over seeds."""

from typing import Dict, List, Optional, Sequence, Tuple
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


def group_accuracy_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    rho: float,
    rare_groups: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Accuracy per group plus summary metrics.

    rare_groups lists the rare groups; the others are common. The default
    (MNIST-C) is group 0 = clean and groups 1..n_groups-1 = rare.

    worst_group   : min accuracy over the n_groups groups
    balanced_mean : mean of the group accuracies
    context_mean  : (1 - rho) * mean(acc_common) + rho * mean(acc_rare), rho = rare fraction
    worst_cell    : min accuracy over (class, group) cells
    """
    correct = np.asarray(y_pred) == np.asarray(y_true)
    groups = np.asarray(groups)
    rare = list(range(1, n_groups)) if rare_groups is None else list(rare_groups)
    common = [k for k in range(n_groups) if k not in rare]
    acc = [float(np.mean(correct[groups == k])) for k in range(n_groups)]
    out = {f"acc_g{k}": acc[k] for k in range(n_groups)}
    out["worst_group"] = min(acc)
    out["balanced_mean"] = float(np.mean(acc))
    out["context_mean"] = ((1.0 - rho) * float(np.mean([acc[k] for k in common]))
                           + rho * float(np.mean([acc[k] for k in rare])))
    cells = []
    for c in np.unique(y_true):
        for k in range(n_groups):
            in_cell = (np.asarray(y_true) == c) & (groups == k)
            if in_cell.any():
                cells.append(np.mean(correct[in_cell]))
    out["worst_cell"] = float(min(cells))
    return out


def detection_auc(L: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC of score L for flagging the `positive` points (0.5 = no information)."""
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return float("nan")
    return float(roc_auc_score(positive, L))


def tail_composition(
    L: np.ndarray,
    groups: np.ndarray,
    tau: float,
    n_groups: int,
    rare_groups: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """What the tail {L_i >= quantile_tau(L)} contains (Table B / paper Table D.2).

    Returns tail size, precision (fraction rare), recall (fraction of rare selected)
    and the number of selected points from each group. rare_groups defaults to
    every group except 0 (MNIST-C).
    """
    groups = np.asarray(groups)
    rare = list(range(1, n_groups)) if rare_groups is None else list(rare_groups)
    row = compute_tail_precision_recall(L, np.isin(groups, rare), [tau])[0]
    tail = np.asarray(L) >= np.quantile(L, tau)
    for k in range(n_groups):
        row[f"n_g{k}"] = int(np.sum(tail & (groups == k)))
    return row


def mean_and_se(df, group_cols: List[str], metric_cols: List[str]):
    """Mean and standard error over seeds: one row per combination of group_cols.

    Columns <metric>_mean and <metric>_se are added, plus n_seeds. SE = sd / sqrt(n).
    """
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[metric_cols].mean().add_suffix("_mean")
    se = grouped[metric_cols].std(ddof=1) / np.sqrt(grouped.size().to_numpy()[:, None])
    out = out.join(se.add_suffix("_se"))
    out["n_seeds"] = grouped["seed"].nunique()
    return out.reset_index()


def paired_differences(df, metric: str, baseline: str, match_cols: List[str], cond_cols: List[str]):
    """Per-seed paired difference metric(condition) - metric(baseline), then mean ± SE.

    The baseline row is the row with intervention == baseline that shares the
    seed and every column in match_cols (e.g. predictor and score).
    """
    keys = ["seed"] + match_cols
    base = df[df["intervention"] == baseline][keys + [metric]]
    base = base.rename(columns={metric: "baseline_value"})
    merged = df.merge(base, on=keys)
    merged["diff"] = merged[metric] - merged["baseline_value"]
    return mean_and_se(merged, match_cols + cond_cols, ["diff"])
