"""Kernel-LOUIS: Exact In-Context Leave-One-Out Scores and Modulation.

For Interpretable Tabular Foundation Models.

Submodules not imported here: kernel_louis.mnist_c (MNIST-C data),
kernel_louis.predictors (logistic regression / 1-NN / TFM consumers) and
kernel_louis.cnn (needs torch).
"""

from kernel_louis.kernels import (
    default_gamma,
    gaussian_kernel,
    log_gaussian_kernel,
    median_heuristic_gamma,
    squared_distances,
)
from kernel_louis.heads import (
    self_normalized_predict,
    kernel_head_proba,
    gp_predict,
    tobit_log_likelihood,
)
from kernel_louis.loo import (
    incontext_loo_nll,
    gp_loo_logdensity,
    press_residuals,
    crossfit_difficulty,
    frozen_loo_proba,
    frozen_loo_score,
    loo_kernel_from_log,
    select_gamma_loo,
    gp_loo_precision,
    gp_loo_score_multiclass,
    true_class_proba,
)
from kernel_louis.modulation import (
    tail_multiplier,
    smooth_multiplier,
    exponential_multiplier,
    pseudo_group_balance,
    group_balance_multiplier,
    random_tail_multiplier,
    top_fraction_mask,
    prune_multiplier,
    random_prune_multiplier,
    surprisal_from_score,
    resample_context,
)
from kernel_louis.data import (
    make_tabular_waterbirds,
    make_spurious_moons,
    make_spurious_split,
    make_noisy_label_split,
)
from kernel_louis.evaluation import (
    compute_group_metrics,
    compute_tail_precision_recall,
    weight_perplexity,
    group_accuracy_metrics,
    detection_auc,
    tail_composition,
    mean_and_se,
    paired_differences,
)
from kernel_louis.audit import (
    audit_kernel_head_influence,
    audit_context_admission,
    compute_audit_metrics,
)
from kernel_louis.models import (
    KernelICLClassifier,
)

__all__ = [
    "default_gamma",
    "gaussian_kernel",
    "median_heuristic_gamma",
    "self_normalized_predict",
    "gp_predict",
    "tobit_log_likelihood",
    "incontext_loo_nll",
    "gp_loo_logdensity",
    "press_residuals",
    "crossfit_difficulty",
    "tail_multiplier",
    "smooth_multiplier",
    "exponential_multiplier",
    "pseudo_group_balance",
    "resample_context",
    "make_tabular_waterbirds",
    "make_spurious_moons",
    "make_spurious_split",
    "make_noisy_label_split",
    "compute_group_metrics",
    "compute_tail_precision_recall",
    "weight_perplexity",
    "audit_kernel_head_influence",
    "audit_context_admission",
    "compute_audit_metrics",
    "log_gaussian_kernel",
    "squared_distances",
    "kernel_head_proba",
    "frozen_loo_proba",
    "frozen_loo_score",
    "loo_kernel_from_log",
    "select_gamma_loo",
    "gp_loo_precision",
    "gp_loo_score_multiclass",
    "true_class_proba",
    "group_balance_multiplier",
    "random_tail_multiplier",
    "top_fraction_mask",
    "prune_multiplier",
    "random_prune_multiplier",
    "surprisal_from_score",
    "group_accuracy_metrics",
    "detection_auc",
    "tail_composition",
    "mean_and_se",
    "paired_differences",
    "KernelICLClassifier",
]
