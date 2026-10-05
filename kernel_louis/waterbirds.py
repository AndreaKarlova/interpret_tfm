"""Waterbirds data for the spurious-correlation experiment.

Each image is a bird (y = 0 landbird, 1 waterbird) pasted onto a background
(place = 0 land, 1 water). The group is g = 2 * y + place:

    g = 0  landbird on land     (common: background matches the bird)
    g = 1  landbird on water    (rare: background contradicts the bird)
    g = 2  waterbird on land    (rare)
    g = 3  waterbird on water   (common)

In the training split the background matches the bird 95% of the time, so a
model can learn the shortcut "water background -> waterbird". Rows are referred
to by their position in metadata.csv.
"""

import os
import tarfile
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

WATERBIRDS_URLS = [
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
]
FOLDER = "waterbird_complete95_forest2water2"
GROUP_NAMES = ["landbird / land", "landbird / water", "waterbird / land", "waterbird / water"]
RARE_GROUPS = (1, 2)
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
EXPECTED_GROUP_COUNTS = {
    "train": [3498, 184, 56, 1057],
    "val": [467, 466, 133, 133],
    "test": [2255, 2255, 642, 642],
}


def download_waterbirds(data_dir: str) -> str:
    """Download and extract the Waterbirds archive (about 0.5 GB) into data_dir, skipped if present.

    Uses the Stanford DRO mirror that the `wilds` package itself wraps (same images and
    metadata.csv). Returns the folder that holds metadata.csv.
    """
    root = os.path.join(data_dir, FOLDER)
    if os.path.exists(os.path.join(root, "metadata.csv")):
        print(f"Waterbirds already present at {root}")
        return root
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "waterbirds.tar.gz")
    for url in WATERBIRDS_URLS:
        try:
            print(f"downloading {url} (about 0.5 GB) ...")
            urllib.request.urlretrieve(url, tar_path)
            if os.path.getsize(tar_path) > 10_000_000:
                break
            print("  file looks truncated, trying the next mirror")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
    else:
        raise RuntimeError("All Waterbirds mirrors failed. Download the archive manually from "
                           f"{WATERBIRDS_URLS[0]}, extract it into {data_dir}, and re-run this cell.")
    print("extracting ...")
    with tarfile.open(tar_path) as archive:
        archive.extractall(data_dir)
    return root


def load_metadata(root: str) -> pd.DataFrame:
    """metadata.csv with an added group column g = 2 * y + place."""
    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    meta["group"] = 2 * meta["y"] + meta["place"]
    return meta


def verify_metadata(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Sanity check 1: the official splits have the expected size of every group (asserts).

    Returns the row indices of each split, {'train': ..., 'val': ..., 'test': ...}.
    """
    assert set(meta["y"]) == {0, 1} and set(meta["place"]) == {0, 1}, "y and place must be 0/1"
    splits = {}
    for name, code in SPLIT_CODES.items():
        rows = np.where(meta["split"].to_numpy() == code)[0]
        counts = np.bincount(meta["group"].to_numpy()[rows], minlength=4).tolist()
        assert counts == EXPECTED_GROUP_COUNTS[name], (
            f"{name} split group counts {counts} differ from {EXPECTED_GROUP_COUNTS[name]}")
        splits[name] = rows
    return splits


def split_validation(val_rows: np.ndarray, groups: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Split the official validation rows in two halves, stratified by group.

    'select' is used to choose tau/lambda; 'balanced' supplies the group-balanced
    reference context (as in DFR, which retrains on half of the validation set).
    """
    rng = np.random.default_rng(seed)
    select, balanced = [], []
    for k in range(4):
        rows = rng.permutation(val_rows[groups[val_rows] == k])
        half = len(rows) // 2
        select.append(rows[:half])
        balanced.append(rows[half:])
    return {"select": np.sort(np.concatenate(select)), "balanced": np.sort(np.concatenate(balanced))}


def context_group_counts(n_per_class: int, rho: float) -> List[int]:
    """Images per group [g0, g1, g2, g3] for a class-balanced context with rare fraction rho.

    Each class gets n_per_class images, of which max(1, round(n_per_class * rho)) have the
    contradicting background (groups 1 and 2), so the background matches the bird for
    a fraction 1 - rho of each class.
    """
    r = max(1, int(round(n_per_class * rho)))
    return [n_per_class - r, r, r, n_per_class - r]


def build_context(pool: np.ndarray, groups: np.ndarray, counts: List[int],
                  rng: np.random.Generator) -> np.ndarray:
    """Row indices with exactly counts[k] rows of group k, drawn from `pool` without replacement.

    Sanity check 2 (asserts): enough rows available, no row used twice, exact group counts.
    """
    chosen = []
    for k, count in enumerate(counts):
        candidates = pool[groups[pool] == k]
        assert len(candidates) >= count, f"group {k}: only {len(candidates)} rows available, need {count}"
        chosen.append(rng.choice(candidates, size=count, replace=False))
    idx = np.concatenate(chosen)
    assert len(np.unique(idx)) == len(idx), "a row is used twice in the context"
    assert np.bincount(groups[idx], minlength=len(counts)).tolist() == list(counts), "wrong group counts"
    return idx


def balanced_rows(rows: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same number of rows from every group (the size of the smallest group in `rows`)."""
    n_min = min(int(np.sum(groups[rows] == k)) for k in range(4))
    return build_context(rows, groups, [n_min] * 4, rng)


def load_image(root: str, filename: str):
    """One Waterbirds image as a PIL RGB image."""
    from PIL import Image
    return Image.open(os.path.join(root, filename)).convert("RGB")


def extract_resnet50_features(root: str, filenames: List[str], device: str = "cpu",
                              batch_size: int = 128) -> np.ndarray:
    """2048-d penultimate-layer features of an ImageNet ResNet-50 (IMAGENET1K_V2), not fine-tuned.

    Standard ImageNet preprocessing: resize to 256, centre crop 224, normalise.
    The network never sees Waterbirds labels, so the features do not encode them.
    """
    import torch
    import torchvision

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net = net.eval().to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(filenames), batch_size):
            batch = [transform(load_image(root, f)) for f in filenames[start:start + batch_size]]
            out.append(net(torch.stack(batch).to(device)).cpu().numpy())
            if (start // batch_size) % 20 == 0:
                print(f"  features: {start + len(batch)}/{len(filenames)} images")
    return np.concatenate(out).astype(np.float32)


def load_or_extract_features(root: str, meta: pd.DataFrame, cache_path: str,
                             device: str = "cpu") -> np.ndarray:
    """ResNet-50 features for every row of metadata.csv, cached in cache_path (.npz).

    The cache stores the image ids so that a stale or mismatched cache is detected.
    """
    ids = meta["img_id"].to_numpy()
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if np.array_equal(cached["img_id"], ids):
            print(f"loaded cached features from {cache_path}")
            return cached["features"]
        print("cached features do not match metadata.csv; recomputing")
    features = extract_resnet50_features(root, meta["img_filename"].tolist(), device)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, features=features, img_id=ids)
    print(f"saved features to {cache_path}")
    return features
