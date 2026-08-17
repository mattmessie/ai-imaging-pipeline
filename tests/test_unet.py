import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.unet import UNet, DoubleConv
from imaging_pipeline.train_unet import (
    NucleiDataset, dice_loss, combined_loss, dice_coefficient, iou_score,
)
from imaging_pipeline.config import TRAIN_DIR, VAL_DIR


def test_unet_forward_pass_shape_matches_input():
    model = UNet(in_ch=1, out_ch=1, base=16)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    assert y.shape == x.shape


def test_unet_works_at_lab_original_resolution_too():
    # Fully convolutional -- architecture shouldn't be hardcoded to 256x256.
    model = UNet(in_ch=1, out_ch=1, base=16)
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    assert y.shape == x.shape


def test_unet_parameter_count_is_small():
    model = UNet(in_ch=1, out_ch=1, base=16)
    n_params = sum(p.numel() for p in model.parameters())
    # Lab notes ~500k params for this architecture -- confirm it's in that ballpark.
    assert 100_000 < n_params < 2_000_000


def test_double_conv_preserves_spatial_size():
    block = DoubleConv(3, 8)
    x = torch.randn(1, 3, 32, 32)
    y = block(x)
    assert y.shape == (1, 8, 32, 32)


def test_dice_coefficient_perfect_match_is_one():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    # logits that, after sigmoid+threshold, exactly reproduce target
    logits = (target * 20) - 10  # large positive where target=1, large negative where 0
    assert dice_coefficient(logits, target) == pytest.approx(1.0, abs=1e-3)


def test_dice_coefficient_no_overlap_is_zero():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 0:2, 0:2] = 1.0
    logits = torch.zeros(1, 1, 8, 8)
    logits[:, :, 6:8, 6:8] = 10.0  # predicts foreground in a disjoint region
    logits[:, :, 0:2, 0:2] = -10.0
    assert dice_coefficient(logits, target) == pytest.approx(0.0, abs=1e-3)


def test_iou_score_perfect_match_is_one():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    logits = (target * 20) - 10
    assert iou_score(logits, target) == pytest.approx(1.0, abs=1e-3)


def test_dice_loss_decreases_as_predictions_improve():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    bad_logits = torch.zeros(1, 1, 8, 8)  # sigmoid=0.5 everywhere, uninformative
    good_logits = (target * 20) - 10       # confidently correct
    assert dice_loss(good_logits, target) < dice_loss(bad_logits, target)


def test_combined_loss_is_bce_plus_dice():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    logits = torch.randn(1, 1, 8, 8)
    import torch.nn.functional as F
    expected = F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)
    assert combined_loss(logits, target).item() == pytest.approx(expected.item())


def test_nuclei_dataset_loads_correct_count_and_shapes():
    ds = NucleiDataset(TRAIN_DIR, augment=False)
    assert len(ds) == 80
    img, msk = ds[0]
    assert img.shape == (1, 256, 256)
    assert msk.shape == (1, 256, 256)
    assert img.dtype == torch.float32
    assert set(torch.unique(msk).tolist()).issubset({0.0, 1.0})


def test_nuclei_dataset_val_split_count():
    ds = NucleiDataset(VAL_DIR, augment=False)
    assert len(ds) == 20


def test_nuclei_dataset_augmentation_is_reproducible_per_seed():
    ds_a = NucleiDataset(TRAIN_DIR, augment=True, seed=0)
    ds_b = NucleiDataset(TRAIN_DIR, augment=True, seed=0)
    img_a, _ = ds_a[0]
    img_b, _ = ds_b[0]
    torch.testing.assert_close(img_a, img_b)
