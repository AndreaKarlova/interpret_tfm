"""TabICL symmetric embedding extraction hooks.

Construction (KernelICL Eq. 14):
Context points and queries are mapped into a shared, label-free symmetric
embedding space by embedding them in query position against a fixed context:
    q_D = k_D = h_D
"""

from typing import Optional, Tuple
import numpy as np


def describe_tabicl_modules(clf: object) -> None:
    """Print the fitted TabICL module tree to locate the ICL transformer."""
    try:
        import torch
    except ImportError:
        print("torch is not installed.")
        return

    net = None
    for attr in ("model_", "model", "_model", "network_", "net_"):
        obj = getattr(clf, attr, None)
        if isinstance(obj, torch.nn.Module):
            net = obj
            print(f"Found nn.Module at clf.{attr}")
            break

    if net is None:
        for k, v in vars(clf).items():
            if isinstance(v, torch.nn.Module):
                net = v
                print(f"Found nn.Module at clf.{k}")
                break

    if net is None:
        print("No nn.Module found on clf.")
        return

    for name, mod in net.named_modules():
        print(f"{name:60s} {type(mod).__name__}")


def extract_symmetric_embeddings(
    clf: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    hook_module_name: Optional[str] = None,
    projection: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract symmetric KernelICL embeddings h(.) for context and query points.

    Every point is embedded in QUERY position against the fixed training context,
    ensuring context and query share an identical representation space.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for TabICL embedding extraction.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clf.fit(np.asarray(X_train, np.float32), np.asarray(y_train))

    if hook_module_name is None:
        target = clf.model_.icl_predictor.tf_icl
    else:
        target = dict(clf.model_.named_modules())[hook_module_name]

    n_ctx = len(X_train)

    def embed_as_query(Q):
        blocks = []
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            blocks.append(t.detach().float().cpu())
        h = target.register_forward_hook(hook)
        try:
            clf.predict_proba(np.asarray(Q, np.float32))
        finally:
            h.remove()
        embs = [b.mean(dim=0)[n_ctx:] for b in blocks]
        if sum(e.shape[0] for e in embs) == len(Q):
            E = torch.cat(embs, dim=0).numpy()
        elif len(embs) > 0 and embs[0].shape[0] == len(Q):
            E = torch.stack(embs, dim=0).mean(dim=0).numpy()
        else:
            E = embs[0].numpy()
        assert E.shape[0] == len(Q), f"Shape mismatch: {E.shape} vs {len(Q)}"
        return E

    E_train = embed_as_query(X_train)
    E_query = embed_as_query(X_query)

    if projection is not None:
        W = np.asarray(projection, np.float32)
        E_train = E_train @ W
        E_query = E_query @ W

    return E_train, E_query
