"""A fine-tuned KernelICL as a plain predictor, plus context LOO difficulty.

Loads a checkpoint written by ``kernelicl_finetune.finetune()`` and wraps it for a
fixed context, so that ``predict_proba`` can be called on arbitrary points (a mesh
grid, an eval set) and ``context_difficulty`` can score every context row by how
hard it is to predict from the others.

    model = load_kernelicl("paper.pt")
    pred = KernelICLPredictor(model, X_ctx, y_ctx)
    pred.predict_proba(X_eval)          # (m, C), columns in pred.classes_ order
    pred.context_difficulty()           # (n,)  L_i = -log p^{-i}(y_i)

Independent of kernelicl_clinical.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from tabicl import TabICLClassifier
from tabicl._model.kernel_head import KernelHead, squared_distances
from tabicl._model.tabicl import TabICL

__all__ = ["KernelICL", "KernelICLPredictor", "load_kernelicl"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class KernelICL:
    """A fine-tuned backbone and its kernel head, as saved by the fine-tune."""

    backbone: TabICL
    head: KernelHead
    config: dict
    val_loss: float
    finetune_config: dict

    def __repr__(self):
        start = self.finetune_config.get("checkpoint", "?")
        return (f"KernelICL(from={start}, {self.head.extra_repr()}, "
                f"val_loss={self.val_loss:.4f})")


def load_kernelicl(path: str, device: Optional[str] = None,
                   kernel: Optional[str] = None) -> KernelICL:
    """Load a ``kernelicl_finetune`` checkpoint.

    ``kernel`` swaps the kernel at load time, e.g. ``"knn"``: the paper trains with
    the Gaussian kernel and swaps at evaluation, since kNN is not differentiable. The
    head's default scale is the one it was trained at, ``1 / (2 sqrt(d_k))`` for the
    Gaussian kernel, unless the fine-tune set ``gamma`` explicitly.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("config", "state_dict", "kernel_head", "head_config")
               if k not in payload]
    if missing:
        raise ValueError(f"{path} is missing {missing}; expected a checkpoint from "
                         f"kernelicl_finetune.finetune()")

    backbone = TabICL(**payload["config"])
    backbone.load_state_dict(payload["state_dict"])

    head_config = payload["head_config"]
    finetune_config = payload.get("finetune_config", {})
    head = KernelHead(d_model=head_config["d_model"], d_k=head_config["d_k"],
                      kernel=head_config["kernel"], gamma=finetune_config.get("gamma"))
    head.load_state_dict(payload["kernel_head"])
    if kernel is not None:
        head = _with_kernel(head, kernel)

    return KernelICL(backbone=backbone.to(device).eval(), head=head.to(device).eval(),
                     config=payload["config"], val_loss=float(payload.get("val_loss", np.nan)),
                     finetune_config=finetune_config)


def _with_kernel(head: KernelHead, kernel: str) -> KernelHead:
    """A copy of ``head`` with another kernel and that kernel's default scale."""
    other = KernelHead(d_model=head.d_model, d_k=head.d_k, kernel=kernel)
    other.load_state_dict(head.state_dict())
    return other.to(head.proj.weight.device)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class KernelICLPredictor:
    """A trained KernelICL conditioned on a fixed context.

    Parameters
    ----------
    model : KernelICL
        From :func:`load_kernelicl`.
    X_ctx, y_ctx : array-like
        The context. ``X_ctx`` may be a DataFrame with string columns and NaNs.
    gamma : float or int, optional
        Kernel scale override (``k`` for kNN). Defaults to the head's own scale,
        i.e. the one it was fine-tuned at.
    norm_method : str, default="none"
        TabICL normalization for the single ensemble member.
    batch_size : int, default=1024
        Query points per forward pass in :meth:`predict_proba`. A mesh grid has
        thousands of points; this bounds peak memory.

    Notes
    -----
    One ensemble member with no feature or class shuffling, so weight column ``i``
    is context row ``i``. TabICL averages 8 shuffled members by default, which
    destroys per-row attribution.
    """

    def __init__(self, model: KernelICL, X_ctx, y_ctx, *, gamma=None,
                 norm_method: str = "none", batch_size: int = 1024, random_state: int = 0):
        self.model, self.head = model, model.head
        self.gamma = gamma
        self.norm_method, self.random_state = norm_method, random_state
        self.batch_size = batch_size
        self.device = next(model.backbone.parameters()).device

        self.X_ctx = X_ctx
        self.y_ctx = (y_ctx.to_numpy() if hasattr(y_ctx, "to_numpy")
                      else np.asarray(y_ctx)).ravel()
        self.clf = self._fit(X_ctx, self.y_ctx)
        self.classes_ = self.clf.classes_
        self.n_classes_ = self.clf.n_classes_
        self._E_train = None

    # -- internals ---------------------------------------------------------- #
    def _fit(self, X, y) -> TabICLClassifier:
        """Fit TabICL's preprocessing on a context, reusing the loaded backbone.

        ``fit`` normally reloads a checkpoint from disk; overriding ``_load_model``
        on the instance points it at the fine-tuned weights instead, so the
        leave-one-out refits below cost a preprocessing pass, not a 110 MB load.
        """
        clf = TabICLClassifier(n_estimators=1, norm_methods=[self.norm_method],
                               feat_shuffle_method="none", class_shuffle_method="none",
                               device=self.device, random_state=self.random_state,
                               kv_cache=False)
        clf.model_ = self.model.backbone
        clf.model_config_ = self.model.config
        clf.model_path_ = None
        clf._load_model = lambda: None
        return clf.fit(X, y)

    @staticmethod
    def _tensors(clf, X_query):
        # The ensemble generator sits after TabICL's numeric encoder, so encode first.
        encoded = clf.X_encoder_.transform(X_query)
        X_ens, y_ens = next(iter(clf.ensemble_generator_.transform(encoded, mode="both").values()))
        device = next(clf.model_.parameters()).device
        return (torch.from_numpy(np.asarray(X_ens)).float().to(device),
                torch.from_numpy(np.asarray(y_ens)).float().to(device))

    @torch.no_grad()
    def _embed(self, clf, X_query):
        """Symmetric in-context embeddings, (1, n, d) and (1, m, d)."""
        X_t, y_t = self._tensors(clf, X_query)
        model = clf.model_
        R = model.row_interactor(
            model.col_embedder(X_t, y_train=y_t, mgr_config=clf.inference_config_.COL_CONFIG),
            mgr_config=clf.inference_config_.ROW_CONFIG,
        )
        return model.icl_predictor.embed(R, y_t, symmetric=True)

    @torch.no_grad()
    def _head(self, clf, y_raw, E_train, E_test):
        y_t = torch.from_numpy(clf.y_encoder_.transform(y_raw)).float().to(E_train.device)[None]
        probs, w = self.head(E_train, E_test, y_t, num_classes=clf.n_classes_, gamma=self.gamma)
        return probs[0].cpu().numpy(), w[0].cpu().numpy()

    def _batches(self, X):
        n = len(X)
        for start in range(0, n, self.batch_size):
            idx = np.arange(start, min(start + self.batch_size, n))
            yield idx, (X.iloc[idx] if hasattr(X, "iloc") else X[idx])

    # -- prediction --------------------------------------------------------- #
    def predict_proba_and_weights(self, X):
        """(probs (m, C), weights (m, n)). ``w[j, i]`` is context row i's share of
        the prediction for query j; ``probs = w @ onehot(y_ctx)`` exactly."""
        probs, weights = [], []
        for _, X_batch in self._batches(X):
            E_train, E_test = self._embed(self.clf, X_batch)
            if self._E_train is None:
                self._E_train = E_train
            p, w = self._head(self.clf, self.y_ctx, E_train, E_test)
            probs.append(p)
            weights.append(w)
        return np.concatenate(probs), np.concatenate(weights)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, columns in ``self.classes_`` order."""
        return self.predict_proba_and_weights(X)[0]

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def weights(self, X) -> np.ndarray:
        return self.predict_proba_and_weights(X)[1]

    # -- matched edited contexts ------------------------------------------- #
    def processed(self, X_query):
        """Rows after this predictor's fitted TabICL preprocessing.

        Returns (context rows (n, H), query rows (m, H), encoded context labels (n,)),
        as tensors on the model's device. Edited contexts are built from these rows,
        so they reuse the preprocessing fitted on the full context (paper §4.2).
        """
        X_t, y_t = self._tensors(self.clf, X_query)
        n = len(self.y_ctx)
        return X_t[0, :n], X_t[0, n:], y_t[0]

    @torch.no_grad()
    def embed_rows(self, ctx_rows, ctx_labels, query_rows):
        """Symmetric embeddings for a batch of already-preprocessed contexts.

        ctx_rows (B, k, H), ctx_labels (B, k) encoded, query_rows (B, m, H).
        Returns E_train (B, k, d) and E_test (B, m, d), before the projection W.
        """
        model, cfg = self.clf.model_, self.clf.inference_config_
        X_t = torch.cat([ctx_rows, query_rows], dim=1)
        R = model.row_interactor(
            model.col_embedder(X_t, y_train=ctx_labels, mgr_config=cfg.COL_CONFIG),
            mgr_config=cfg.ROW_CONFIG,
        )
        return model.icl_predictor.embed(R, ctx_labels, symmetric=True)

    @torch.no_grad()
    def edited_passes(self, contexts, batch_size: int = 8, return_embeddings: bool = False):
        """Matched recomputation for a list of edited contexts.

        Each context is a dict with ``rows`` (k, H) preprocessed context rows,
        ``labels`` (k,) encoded labels and ``queries`` (m, H) preprocessed query rows,
        typically built from :meth:`processed`. Nothing is refitted: unlike
        ``loo_proba(mode="refit")``, TabICL's preprocessing stays the one fitted on the
        full context. Contexts with equal (k, m) are run together, ``batch_size`` at a time.

        Yields (indices, probs (B, m, C)) per batch, or (indices, probs, E_train, E_test)
        with ``return_embeddings=True``. Embeddings are unprojected; use ``self.head.embed``.
        """
        start = 0
        while start < len(contexts):
            k, m = len(contexts[start]["labels"]), len(contexts[start]["queries"])
            stop = start
            while (stop < len(contexts) and stop - start < batch_size
                   and len(contexts[stop]["labels"]) == k and len(contexts[stop]["queries"]) == m):
                stop += 1
            batch = contexts[start:stop]
            rows = torch.stack([c["rows"] for c in batch])
            labels = torch.stack([c["labels"] for c in batch]).float()
            queries = torch.stack([c["queries"] for c in batch])
            E_train, E_test = self.embed_rows(rows, labels, queries)
            probs, _ = self.head(E_train, E_test, labels, num_classes=self.n_classes_, gamma=self.gamma)
            indices = list(range(start, stop))
            if return_embeddings:
                yield indices, probs.cpu().numpy(), E_train, E_test
            else:
                yield indices, probs.cpu().numpy()
            start = stop

    # -- leave-one-out ------------------------------------------------------ #
    @property
    def E_train(self) -> torch.Tensor:
        """Symmetric context embeddings. Independent of the queries: column
        statistics and ICL keys see context rows only."""
        if self._E_train is None:
            first = self.X_ctx.iloc[:1] if hasattr(self.X_ctx, "iloc") else self.X_ctx[:1]
            self._E_train = self._embed(self.clf, first)[0]
        return self._E_train

    @torch.no_grad()
    def context_gram(self) -> np.ndarray:
        """Row-normalized context-on-context weights S (n, n), the hat matrix of the
        kernel smoother, including the self-weights on the diagonal."""
        y_t = torch.from_numpy(self.clf.y_encoder_.transform(self.y_ctx)).float()
        _, w = self.head(self.E_train, self.E_train, y_t.to(self.device)[None],
                         num_classes=self.n_classes_, gamma=self.gamma)
        return w[0].cpu().numpy()

    @torch.no_grad()
    def _context_weights_without_self(self) -> np.ndarray:
        """Context-on-context kernel weights (n, n) renormalised without the diagonal, computed
        from logits with the self-affinity set to -inf, so they stay exact for sharp kernels."""
        g = self.gamma if self.gamma is not None else self.head.gamma
        H = self.head.embed(self.E_train)[0]
        if self.head.kernel == "dot":
            logits = g * (H @ H.T)
        else:
            logits = -g * squared_distances(H[None], H[None])[0]
        logits.fill_diagonal_(float("-inf"))
        return torch.softmax(logits.double(), dim=-1).cpu().numpy()

    def loo_proba(self, mode: str = "frozen") -> np.ndarray:
        """Probability each context row's own label gets when it is left out, (n,).

        ``mode="frozen"`` -- kernel-step LOO, paper Eq. 3: drop the self-weight and
        renormalize, with every embedding held fixed. Exact for the kernel step,
        free, but row i still sits in the context that produced the embeddings, so
        its own label has shaped them (see README: train-side embeddings encode
        their own labels).

        ``mode="refit"`` -- true in-context LOO: drop row i from the context,
        re-run the whole model, and predict it as a query. n forward passes; fine
        for a few hundred rows. This is the quantity the frozen score approximates.
        """
        y = self.clf.y_encoder_.transform(self.y_ctx)
        n = len(y)

        if mode == "frozen":
            if self.head.kernel == "knn":
                # Ask for k+1 neighbours so that k remain once the self-match (at
                # distance zero, always in the top k+1) is removed.
                k = int(self.gamma if self.gamma is not None else self.head.gamma)
                saved, self.gamma = self.gamma, k + 1
                S = self.context_gram()
                self.gamma = saved
                np.fill_diagonal(S, 0.0)
                denom = S.sum(1)
                same = (y[:, None] == y[None, :])
                return np.divide((S * same).sum(1), denom, out=np.zeros(n), where=denom > 0)
            # Gaussian / dot kernels: exclude the self-affinity *before* normalising. Zeroing
            # it after the softmax loses everything else once the self-weight rounds to 1,
            # which happens with sharp kernels (the self-distance is 0).
            S = self._context_weights_without_self()
            same = (y[:, None] == y[None, :])
            return (S * same).sum(1)

        if mode == "refit":
            p = np.zeros(n)
            for i in range(n):
                keep = np.delete(np.arange(n), i)
                X_keep = self.X_ctx.iloc[keep] if hasattr(self.X_ctx, "iloc") else self.X_ctx[keep]
                X_i = self.X_ctx.iloc[[i]] if hasattr(self.X_ctx, "iloc") else self.X_ctx[[i]]
                sub = self._fit(X_keep, self.y_ctx[keep])
                if self.y_ctx[i] not in sub.classes_:
                    continue        # its class vanished from the context: p = 0
                E_train, E_test = self._embed(sub, X_i)
                probs, _ = self._head(sub, self.y_ctx[keep], E_train, E_test)
                p[i] = probs[0, list(sub.classes_).index(self.y_ctx[i])]
            return p

        raise ValueError(f"mode must be 'frozen' or 'refit', got {mode!r}")

    def context_difficulty(self, mode: str = "frozen", score: str = "nll",
                           eps: float = 1e-12) -> np.ndarray:
        """LOO difficulty of each context row; higher is harder.

        ``score="nll"`` gives L_i = -log p^{-i}(y_i), matching
        ``kernel_louis.loo.incontext_loo_nll``; ``score="1-p"`` gives the bounded
        variant of ``frozen_loo_score``. Same ranking either way.
        """
        p = self.loo_proba(mode)
        if score == "nll":
            return -np.log(np.clip(p, eps, 1.0))
        if score == "1-p":
            return 1.0 - p
        raise ValueError(f"score must be 'nll' or '1-p', got {score!r}")
