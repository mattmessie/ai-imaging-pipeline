import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.data_prep import to_grayscale, resize_to_common_size, load_rgb, list_image_paths
from imaging_pipeline.config import TRAIN_DIR, IMAGE_SIZE


def test_list_image_paths_finds_all_train_images():
    paths = list_image_paths(TRAIN_DIR)
    assert len(paths) == 80
    assert all(p.suffix == ".png" for p in paths)


def test_load_rgb_returns_float_in_unit_range():
    paths = list_image_paths(TRAIN_DIR)
    img = load_rgb(paths[0])
    assert img.ndim == 3 and img.shape[-1] == 3
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_to_grayscale_reduces_to_single_channel():
    rgb = np.random.RandomState(0).rand(64, 64, 3).astype(np.float32)
    gray = to_grayscale(rgb)
    assert gray.shape == (64, 64)
    assert gray.min() >= 0.0 and gray.max() <= 1.0


def test_resize_to_common_size_is_noop_for_already_correct_size():
    img = np.random.RandomState(0).rand(*IMAGE_SIZE).astype(np.float32)
    resized = resize_to_common_size(img, IMAGE_SIZE)
    np.testing.assert_array_equal(resized, img)


def test_resize_to_common_size_actually_resizes_when_needed():
    img = np.random.RandomState(0).rand(128, 128).astype(np.float32)
    resized = resize_to_common_size(img, (256, 256))
    assert resized.shape == (256, 256)


def test_real_train_images_are_256x256_after_pipeline():
    paths = list_image_paths(TRAIN_DIR)[:5]
    for p in paths:
        rgb = load_rgb(p)
        resized = resize_to_common_size(rgb, IMAGE_SIZE)
        gray = to_grayscale(resized)
        assert gray.shape == IMAGE_SIZE
