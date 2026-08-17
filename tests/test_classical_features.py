import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.classical_features import otsu_segment, extract_region_features, summarise_features
from imaging_pipeline.data_prep import load_rgb, to_grayscale
from imaging_pipeline.config import TRAIN_DIR


def _load_gray(stem="train_062"):
    rgb = load_rgb(TRAIN_DIR / "images" / f"{stem}.png")
    return to_grayscale(rgb)


def test_otsu_segment_returns_boolean_mask():
    gray = _load_gray()
    mask = otsu_segment(gray)
    assert mask.dtype == bool
    assert mask.shape == gray.shape


def test_otsu_segment_detects_some_foreground_on_real_image():
    gray = _load_gray()
    mask = otsu_segment(gray)
    # train_062 has 27 nuclei -- should detect a non-trivial foreground
    # fraction, not everything and not nothing.
    frac = mask.mean()
    assert 0.001 < frac < 0.5


def test_extract_region_features_returns_expected_columns():
    gray = _load_gray()
    mask = otsu_segment(gray)
    df = extract_region_features(gray, mask)
    for col in ["label", "area", "perimeter", "eccentricity", "solidity", "mean_intensity", "extent"]:
        assert col in df.columns


def test_extract_region_features_empty_mask_returns_empty_df():
    gray = _load_gray()
    empty_mask = np.zeros_like(gray, dtype=bool)
    df = extract_region_features(gray, empty_mask)
    assert len(df) == 0


def test_summarise_features_empty_table():
    import pandas as pd
    summary = summarise_features(pd.DataFrame(), "test_image")
    assert "no objects" in summary.lower()


def test_summarise_features_is_deterministic():
    gray = _load_gray()
    mask = otsu_segment(gray)
    df = extract_region_features(gray, mask)
    s1 = summarise_features(df, "train_062")
    s2 = summarise_features(df, "train_062")
    assert s1 == s2


def test_summarise_features_mentions_object_count():
    gray = _load_gray()
    mask = otsu_segment(gray)
    df = extract_region_features(gray, mask)
    summary = summarise_features(df, "train_062")
    assert str(len(df)) in summary


def test_otsu_object_count_reasonably_close_to_ground_truth():
    # train_062 has 27 ground-truth nuclei (metadata.csv). Otsu + connected
    # components won't match exactly (touching nuclei can merge into one
    # blob) but should be in a plausible ballpark, not wildly off.
    gray = _load_gray()
    mask = otsu_segment(gray)
    df = extract_region_features(gray, mask)
    assert 5 <= len(df) <= 40
