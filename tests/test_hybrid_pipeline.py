import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.hybrid_pipeline import (
    build_hybrid_prompt, has_required_fields, prepare_pipeline_inputs,
    aggregate_records, REQUIRED_FIELDS,
)
from imaging_pipeline.unet import UNet
from imaging_pipeline.data_prep import load_rgb, to_grayscale
from imaging_pipeline.config import TEST_DIR


def test_module_imports_without_ollama_installed():
    import imaging_pipeline.hybrid_pipeline  # noqa: F401


def test_build_hybrid_prompt_includes_image_id_and_summary():
    prompt = build_hybrid_prompt("test_000", "detected 8 objects")
    assert "test_000" in prompt
    assert "detected 8 objects" in prompt


def test_build_hybrid_prompt_names_all_required_fields():
    prompt = build_hybrid_prompt("x", "y")
    for field in REQUIRED_FIELDS:
        assert field in prompt


def test_build_hybrid_prompt_clarifies_no_image_seen():
    prompt = build_hybrid_prompt("x", "y")
    assert "have not seen" in prompt.lower() or "not seen the image" in prompt.lower()


def test_has_required_fields_detects_missing():
    incomplete = {"image_id": "x", "n_objects": 5}
    assert not has_required_fields(incomplete)


def test_has_required_fields_all_present():
    complete = {f: "val" for f in REQUIRED_FIELDS}
    assert has_required_fields(complete)


def test_prepare_pipeline_inputs_on_real_test_image():
    # Untrained U-Net is fine here -- this test checks the pipeline wiring
    # (mask -> features -> summary -> prompt), not segmentation quality.
    device = torch.device("cpu")
    model = UNet(in_ch=1, out_ch=1, base=16).to(device)

    rgb = load_rgb(TEST_DIR / "images" / "test_000.png")
    gray = to_grayscale(rgb)

    result = prepare_pipeline_inputs("test_000", gray, model, device)
    assert result["image_id"] == "test_000"
    assert result["mask"].dtype == bool
    assert isinstance(result["features_df"], pd.DataFrame)
    assert "test_000" in result["summary_text"]
    assert "U-Net segmentation" in result["summary_text"]
    assert "test_000" in result["prompt"]


def test_aggregate_records_produces_dataframe_with_all_rows():
    records = [
        {"image_id": "a", "n_objects": 5, "mean_area": 100.0, "density_class": "sparse", "quality_flag": "ok"},
        {"image_id": "b", "n_objects": 20, "mean_area": 80.0, "density_class": "dense", "quality_flag": "ok"},
    ]
    df = aggregate_records(records)
    assert len(df) == 2
    assert list(df["image_id"]) == ["a", "b"]
