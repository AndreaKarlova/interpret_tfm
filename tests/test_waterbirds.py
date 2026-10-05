"""Tests for the Waterbirds helpers and the rare_groups option of the group metrics."""

import os

import numpy as np
import pandas as pd
import pytest

from kernel_louis import waterbirds
from kernel_louis.evaluation import group_accuracy_metrics, tail_composition


def fake_metadata():
    """metadata.csv-like table with the official split and group sizes."""
    rows = []
    for name, code in waterbirds.SPLIT_CODES.items():
        for g, count in enumerate(waterbirds.EXPECTED_GROUP_COUNTS[name]):
            for _ in range(count):
                rows.append({"split": code, "y": g // 2, "place": g % 2})
    meta = pd.DataFrame(rows)
    meta["img_id"] = np.arange(1, len(meta) + 1)
    meta["img_filename"] = [f"img_{i}.jpg" for i in meta["img_id"]]
    meta["group"] = 2 * meta["y"] + meta["place"]
    return meta


def test_verify_metadata_passes_and_catches_wrong_counts():
    meta = fake_metadata()
    splits = waterbirds.verify_metadata(meta)
    assert [len(splits[s]) for s in ["train", "val", "test"]] == [4795, 1199, 5794]
    broken = meta.drop(index=meta.index[0])
    with pytest.raises(AssertionError):
        waterbirds.verify_metadata(broken)


def test_load_metadata_adds_group(tmp_path):
    pd.DataFrame({"img_id": [1, 2], "img_filename": ["a", "b"], "y": [1, 0],
                  "split": [0, 2], "place": [0, 1]}).to_csv(tmp_path / "metadata.csv", index=False)
    assert waterbirds.load_metadata(str(tmp_path))["group"].tolist() == [2, 1]


def test_context_group_counts():
    assert waterbirds.context_group_counts(1000, 0.05) == [950, 50, 50, 950]
    assert waterbirds.context_group_counts(1000, 0.0001) == [999, 1, 1, 999]


def test_build_context_exact_counts_and_errors():
    meta = fake_metadata()
    groups = meta["group"].to_numpy()
    train = waterbirds.verify_metadata(meta)["train"]
    rng = np.random.default_rng(0)
    idx = waterbirds.build_context(train, groups, [950, 50, 50, 950], rng)
    assert len(idx) == 2000 and len(np.unique(idx)) == 2000
    assert np.bincount(groups[idx]).tolist() == [950, 50, 50, 950]
    assert np.all(np.isin(idx, train))
    with pytest.raises(AssertionError):  # only 56 waterbirds on land in the training split
        waterbirds.build_context(train, groups, [10, 10, 60, 10], rng)


def test_split_validation_and_balanced_rows():
    meta = fake_metadata()
    groups = meta["group"].to_numpy()
    val = waterbirds.verify_metadata(meta)["val"]
    halves = waterbirds.split_validation(val, groups, seed=0)
    assert len(np.intersect1d(halves["select"], halves["balanced"])) == 0
    assert len(halves["select"]) + len(halves["balanced"]) == len(val)
    bal = waterbirds.balanced_rows(halves["balanced"], groups, np.random.default_rng(0))
    counts = np.bincount(groups[bal], minlength=4)
    assert len(set(counts)) == 1 and counts[0] == 133 - 133 // 2


def test_feature_cache_round_trip(tmp_path, monkeypatch):
    meta = fake_metadata().head(5)
    calls = []

    def fake_extract(root, filenames, device="cpu", batch_size=128):
        calls.append(len(filenames))
        return np.arange(len(filenames) * 3, dtype=np.float32).reshape(len(filenames), 3)

    monkeypatch.setattr(waterbirds, "extract_resnet50_features", fake_extract)
    path = str(tmp_path / "cache" / "features.npz")
    first = waterbirds.load_or_extract_features("root", meta, path)
    second = waterbirds.load_or_extract_features("root", meta, path)
    np.testing.assert_array_equal(first, second)
    assert calls == [5]  # the second call used the cache
    other = meta.copy()
    other["img_id"] = other["img_id"] + 100
    waterbirds.load_or_extract_features("root", other, path)
    assert calls == [5, 5]  # mismatched ids: recomputed


def test_group_metrics_with_two_common_groups():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    g = np.array([0, 0, 1, 3, 3, 2, 3, 0])
    out = group_accuracy_metrics(y_pred, y_true, g, n_groups=4, rho=0.05, rare_groups=(1, 2))
    assert out["acc_g1"] == 0.0 and out["acc_g2"] == 0.0 and out["acc_g0"] == 1.0
    assert out["context_mean"] == pytest.approx(0.95 * 1.0 + 0.05 * 0.0)
    default = group_accuracy_metrics(y_pred, y_true, g, n_groups=4, rho=0.05)
    assert default["context_mean"] == pytest.approx(0.95 * 1.0 + 0.05 * np.mean([0.0, 0.0, 1.0]))


def test_tail_composition_with_rare_groups():
    L = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    g = np.array([1, 2, 0, 3, 3])
    row = tail_composition(L, g, tau=0.6, n_groups=4, rare_groups=(1, 2))
    assert row["precision"] == 1.0 and row["recall"] == 1.0
