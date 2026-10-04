"""MNIST-C data for the rare-corruption experiment.

Image index k is the same handwritten digit in every corruption folder, so all
splitting is done on indices first; which version (clean or corrupted) of an
image is used is decided afterwards. A "version" is a folder name such as
"identity" (clean) or "zigzag".
"""

import os
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np

MNIST_C_URL = "https://zenodo.org/records/3239543/files/mnist_c.zip?download=1"

CORRUPTIONS = [
    "shot_noise", "impulse_noise", "glass_blur", "motion_blur", "shear",
    "scale", "rotate", "brightness", "translate", "stripe", "fog",
    "spatter", "dotted_line", "zigzag", "canny_edges",
]

# Secondary (asymmetric) label-noise map: visually confusable digits.
ASYMMETRIC_MAP = {1: 7, 7: 1, 3: 8, 8: 3, 4: 9, 9: 4, 5: 6, 6: 5, 2: 7, 0: 6}


def download_mnist_c(data_dir: str, url: str = MNIST_C_URL) -> str:
    """Download and unzip mnist_c.zip into data_dir (skipped if present). Returns the mnist_c folder."""
    root = os.path.join(data_dir, "mnist_c")
    if os.path.isdir(os.path.join(root, "identity")):
        print(f"MNIST-C already present at {root}")
        return root
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "mnist_c.zip")
    print(f"downloading {url} (about 250 MB) ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        raise RuntimeError(
            f"Download failed ({type(e).__name__}: {e}). Download mnist_c.zip manually from "
            f"https://zenodo.org/records/3239543, upload it to {data_dir}, and re-run this cell."
        )
    print("unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    return root


def list_versions(root: str) -> List[str]:
    """Sorted names of the version folders (identity plus the corruptions)."""
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def load_images(root: str, version: str, split: str) -> np.ndarray:
    """uint8 images of one version and split ('train' or 'test'), memory-mapped from disk."""
    return np.load(os.path.join(root, version, f"{split}_images.npy"), mmap_mode="r")


def load_labels(root: str, version: str, split: str) -> np.ndarray:
    """Integer digit labels of one version and split."""
    return np.load(os.path.join(root, version, f"{split}_labels.npy")).astype(int)


def verify_mnist_c(root: str, versions: List[str],
                   expected: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
    """Sanity check 1: expected shapes, and identical labels in every folder (asserts).

    expected maps split -> number of images (default 60,000 train, 10,000 test).
    Returns the labels of each split, e.g. {'train': (60000,), 'test': (10000,)}.
    """
    if expected is None:
        expected = {"train": 60000, "test": 10000}
    labels = {}
    for split, n in expected.items():
        reference = load_labels(root, "identity", split)
        for version in versions:
            images = load_images(root, version, split)
            assert images.shape == (n, 28, 28, 1), f"{version}/{split}: shape {images.shape}"
            assert np.array_equal(load_labels(root, version, split), reference), (
                f"labels of {version}/{split} differ from identity/{split}")
        labels[split] = reference
    return labels


def sample_per_class(pool: np.ndarray, labels: np.ndarray, n_per_class: int,
                     rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """For every class, n_per_class indices drawn from `pool` without replacement."""
    out = {}
    for c in np.unique(labels[pool]):
        candidates = pool[labels[pool] == c]
        assert len(candidates) >= n_per_class, f"class {c}: only {len(candidates)} in pool"
        out[int(c)] = rng.choice(candidates, size=n_per_class, replace=False)
    return out


def make_index_splits(train_labels: np.ndarray, seed: int, n_val_per_class: int = 500,
                      n_cnn_per_class: int = 2000) -> Dict[str, np.ndarray]:
    """Split the 60,000 train indices into validation, CNN and context pools (disjoint)."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(train_labels))
    val = np.concatenate(list(sample_per_class(all_idx, train_labels, n_val_per_class, rng).values()))
    rest = np.setdiff1d(all_idx, val)
    cnn = np.concatenate(list(sample_per_class(rest, train_labels, n_cnn_per_class, rng).values()))
    context = np.setdiff1d(rest, cnn)
    return {"val": np.sort(val), "cnn": np.sort(cnn), "context": context}


def rare_per_class(n_per_class: int, rho: float, n_rare: int) -> int:
    """Rare images per digit per corruption: max(1, round(n_per_class * rho / R))."""
    return max(1, int(round(n_per_class * rho / n_rare)))


def build_context(pool: np.ndarray, labels: np.ndarray, versions: List[str],
                  counts_per_class: List[int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Class-stratified context: per digit, draw sum(counts) indices and randomly give
    counts[k] of them version k. Group g = k (0 = versions[0], normally identity).

    Sanity check 2 (asserts): no index used twice, exact class stratification.
    """
    n_per_class = int(sum(counts_per_class))
    idx, y, g = [], [], []
    for c, chosen in sample_per_class(pool, labels, n_per_class, rng).items():
        chosen = rng.permutation(chosen)
        start = 0
        for k, count in enumerate(counts_per_class):
            idx.append(chosen[start:start + count])
            y.append(np.full(count, c))
            g.append(np.full(count, k))
            start += count
    idx, y, g = np.concatenate(idx), np.concatenate(y), np.concatenate(g)
    assert len(np.unique(idx)) == len(idx), "an image index is used twice in the context"
    assert np.all(np.bincount(y) == n_per_class), "class stratification is not exact"
    return {"idx": idx, "y": y, "g": g, "version": np.array(versions)[g]}


def build_paired_set(pool: np.ndarray, labels: np.ndarray, versions: List[str],
                     n_per_class: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Paired evaluation set: n_per_class digits per class, each shown in every version.

    Every group therefore contains exactly the same digits.
    """
    chosen = np.concatenate(list(sample_per_class(pool, labels, n_per_class, rng).values()))
    idx = np.tile(chosen, len(versions))
    g = np.repeat(np.arange(len(versions)), len(chosen))
    return {"idx": idx, "y": labels[idx], "g": g, "version": np.array(versions)[g]}


def gather_images(root: str, split: str, idx: np.ndarray, version: np.ndarray) -> np.ndarray:
    """uint8 images (n, 28, 28): row k is image idx[k] taken from folder version[k]."""
    out = np.zeros((len(idx), 28, 28), dtype=np.uint8)
    for v in np.unique(version):
        rows = np.where(version == v)[0]
        out[rows] = load_images(root, v, split)[idx[rows], :, :, 0]
    return out


def flip_labels(y: np.ndarray, eta: float, kind: str, rng: np.random.Generator,
                n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Flip a uniformly random fraction eta of labels, independent of group.

    kind='symmetric': new label uniform over the other n_classes - 1 classes.
    kind='asymmetric': new label from ASYMMETRIC_MAP.
    Returns (noisy labels, boolean mask of flipped points).
    """
    n = len(y)
    flipped = np.zeros(n, dtype=bool)
    flipped[rng.choice(n, size=int(round(eta * n)), replace=False)] = True
    y_noisy = np.array(y, copy=True)
    for i in np.where(flipped)[0]:
        if kind == "symmetric":
            others = [c for c in range(n_classes) if c != y[i]]
            y_noisy[i] = rng.choice(others)
        else:
            y_noisy[i] = ASYMMETRIC_MAP[int(y[i])]
    return y_noisy, flipped
