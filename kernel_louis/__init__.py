"""Kernel-LOUIS: Exact In-Context Leave-One-Out Scores and Modulation.

For Interpretable Tabular Foundation Models.
"""

from kernel_louis.kernels import (
    default_gamma,
    gaussian_kernel,
    median_heuristic_gamma,
)
from kernel_louis.heads import (
    self_normalized_predict,
    gp_predict,
    tobit_log_likelihood,
)
from kernel_louis.loo import (
    incontext_loo_nll,
    gp_loo_logdensity,
    press_residuals,
    crossfit_difficulty,
)
from kernel_louis.modulation import (
    tail_multiplier,
    smooth_multiplier,
    exponential_multiplier,
    pseudo_group_balance,
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
    "KernelICLClassifier",
]
