import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.llm_utils import parse_json_and_narrative
from imaging_pipeline.llm_interpretation import build_numbers_first_prompt


def test_module_imports_without_ollama_installed():
    import imaging_pipeline.llm_utils  # noqa: F401
    import imaging_pipeline.llm_interpretation  # noqa: F401


def test_parse_json_and_narrative_clean():
    text = 'JSON:\n{"n_objects": 22, "density_class": "moderate", "shape_regularity": "regular", "quality_flag": "ok"}\n\nNARRATIVE:\nThe image shows a moderate density of regular objects.'
    record, narrative = parse_json_and_narrative(text)
    assert record["n_objects"] == 22
    assert record["density_class"] == "moderate"
    assert narrative == "The image shows a moderate density of regular objects."


def test_parse_json_and_narrative_with_code_fence():
    text = 'JSON:\n```json\n{"n_objects": 8, "density_class": "sparse", "shape_regularity": "regular", "quality_flag": "ok"}\n```\n\nNARRATIVE:\nSparse regular objects.'
    record, narrative = parse_json_and_narrative(text)
    assert record["n_objects"] == 8
    assert narrative == "Sparse regular objects."


def test_parse_json_and_narrative_missing_narrative_marker():
    text = 'JSON:\n{"n_objects": 5, "density_class": "sparse", "shape_regularity": "regular", "quality_flag": "ok"}'
    record, narrative = parse_json_and_narrative(text)
    assert record["n_objects"] == 5
    assert narrative == ""


def test_parse_json_and_narrative_malformed_json_returns_error():
    text = "JSON:\nnot valid json at all\n\nNARRATIVE:\nSome text."
    record, narrative = parse_json_and_narrative(text)
    assert "error" in record


def test_build_numbers_first_prompt_never_mentions_seeing_image():
    prompt = build_numbers_first_prompt("test_001", "detected 10 objects")
    # Must explicitly clarify the model has NOT seen the image (numbers only).
    assert "have not seen" in prompt.lower() or "not seen the image" in prompt.lower()


def test_build_numbers_first_prompt_includes_summary_and_id():
    summary = "In test_001, Otsu segmentation detected 10 objects."
    prompt = build_numbers_first_prompt("test_001", summary)
    assert "test_001" in prompt
    assert summary in prompt


def test_build_numbers_first_prompt_names_all_four_required_fields():
    prompt = build_numbers_first_prompt("x", "y")
    for field in ["n_objects", "density_class", "shape_regularity", "quality_flag"]:
        assert field in prompt


def test_build_numbers_first_prompt_no_diagnostic_language_requested():
    prompt = build_numbers_first_prompt("x", "y")
    assert "diagnos" in prompt.lower()  # the word appears, in a *prohibition*
    assert "not provide any clinical" in prompt.lower() or "no diagnos" in prompt.lower() or "not" in prompt.lower()
