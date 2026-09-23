"""Scikit-Learn compatible estimator for KernelICL and Kernel-LOUIS."""

from typing import Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from kernel_louis.kernels import default_gamma, gaussian_kernel, median_heuristic_gamma
from kernel_louis.heads import self_normalized_predict, gp_predict
from kernel_louis.loo import incontext_loo_nll, gp_loo_logdensity
from kernel_louis.modulation import (
    exponential_multiplier,
    pseudo_group_balance,
    smooth_multiplier,
    tail_multiplier,
)


class KernelICLClassifier(BaseEstimator, ClassifierMixin):
    """Interpretable In-Context Kernel Classifier with Closed-Form LOO Modulation.

    Parameters
    ----------
    gamma : float or str
        Kernel bandwidth. If 'median', computed via pairwise median heuristic.
        If 'default', computed via 1 / (2 * sqrt(d)).
    lam : float, default=0.0
        Upweighting modulation strength lambda. lam=0 recovers baseline KernelICL.
    tau : float, default=0.80
        Quantile threshold for tail-set modulation.
    modulation : {'tail', 'smooth', 'exponential', 'pseudo_group', 'none'}, default='tail'
        Modulation strategy for the in-context points.
    head_type : {'soft_vote', 'gp'}, default='soft_vote'
        Head type for prediction and difficulty scoring.
    sigma2 : float, default=0.1
        Noise variance regularizer for Gaussian Process head.
    """

    def __init__(
        self,
        gamma: Union[float, str] = "median",
        lam: float = 0.0,
        tau: float = 0.80,
        modulation: str = "tail",
        head_type: str = "soft_vote",
        sigma2: float = 0.1,
    ):
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.modulation = modulation
        self.head_type = head_type
        self.sigma2 = sigma2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelICLClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Currently only binary classification is supported.")

        self.X_train_ = X
        self.y_train_ = y

        d = X.shape[1]
        if self.gamma == "median":
            self.gamma_ = median_heuristic_gamma(X)
        elif self.gamma == "default":
            self.gamma_ = default_gamma(d)
        else:
            self.gamma_ = float(self.gamma)

        # Context Gram matrix
        self.K_tr_ = gaussian_kernel(X, X, self.gamma_)

        # Closed-form LOO difficulty score
        if self.head_type == "gp":
            self.loo_scores_ = gp_loo_logdensity(self.K_tr_, self.y_train_, sigma2=self.sigma2)
        else:
            self.loo_scores_ = incontext_loo_nll(self.K_tr_, self.y_train_)

        # Compute sample multipliers m_i
        if self.modulation == "tail":
            self.m_ = tail_multiplier(self.loo_scores_, self.lam, tau=self.tau)
        elif self.modulation == "smooth":
            self.m_ = smooth_multiplier(self.loo_scores_, self.lam)
        elif self.modulation == "exponential":
            self.m_ = exponential_multiplier(self.loo_scores_, gamma=self.lam)
        elif self.modulation == "pseudo_group":
            self.m_ = pseudo_group_balance(self.y_train_, self.loo_scores_, tau=self.tau)
        else:
            self.m_ = np.ones_like(self.loo_scores_, dtype=float)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["X_train_", "y_train_", "K_tr_", "m_"])
        X = np.asarray(X, dtype=np.float32)
        K_qt = gaussian_kernel(X, self.X_train_, self.gamma_)

        if self.head_type == "gp":
            mu, _ = gp_predict(K_qt, self.K_tr_, self.y_train_, sigma2=self.sigma2)
            p1 = np.clip(mu, 0.0, 1.0)
        else:
            p1, self.weights_ = self_normalized_predict(K_qt, self.y_train_, m=self.m_)

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
