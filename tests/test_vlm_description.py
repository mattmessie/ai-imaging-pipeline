import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.vlm_description import (
    build_optimised_prompt, NAIVE_PROMPT, parse_json_response, has_required_fields,
)


def test_module_imports_without_ollama_installed():
    # query_vlm/run_prompt_comparison defer their `from ollama import chat`
    # to call-time specifically so this module can be imported and its
    # pure-logic functions tested on a machine without ollama installed.
    import imaging_pipeline.vlm_description  # noqa: F401


def test_optimised_prompt_anchors_descriptive_not_diagnostic():
    prompt = build_optimised_prompt()
    assert "not" in prompt.lower() and "diagnos" in prompt.lower()


def test_optimised_prompt_permits_uncertain():
    prompt = build_optimised_prompt()
    assert "uncertain" in prompt.lower()


def test_optimised_prompt_names_all_four_required_fields():
    prompt = build_optimised_prompt()
    for field in ["modality", "tissue_type", "notable_features", "image_quality"]:
        assert field in prompt


def test_naive_prompt_is_unstructured():
    # The naive prompt should NOT already contain the JSON schema or
    # safety framing -- it's supposed to be the unhelpful baseline.
    assert "json" not in NAIVE_PROMPT.lower()
    assert "uncertain" not in NAIVE_PROMPT.lower()


def test_parse_json_response_clean_json():
    text = '{"modality": "fluorescence microscopy", "tissue_type": "uncertain", "notable_features": "bright round objects", "image_quality": "good"}'
    record = parse_json_response(text)
    assert record["modality"] == "fluorescence microscopy"
    assert has_required_fields(record)


def test_parse_json_response_with_markdown_fence():
    text = '```json\n{"modality": "uncertain", "tissue_type": "uncertain", "notable_features": "dots", "image_quality": "moderate"}\n```'
    record = parse_json_response(text)
    assert record["image_quality"] == "moderate"
    assert has_required_fields(record)


def test_parse_json_response_with_leading_prose():
    text = 'Here is the JSON record:\n{"modality": "uncertain", "tissue_type": "uncertain", "notable_features": "several small objects", "image_quality": "good"}\nLet me know if you need anything else.'
    record = parse_json_response(text)
    assert has_required_fields(record)


def test_parse_json_response_malformed_returns_error_not_exception():
    text = "I cannot provide a diagnosis for this image."
    record = parse_json_response(text)
    assert "error" in record
    assert "raw" in record


def test_has_required_fields_detects_missing_field():
    incomplete = {"modality": "uncertain", "tissue_type": "uncertain"}
    assert not has_required_fields(incomplete)


def test_has_required_fields_all_present():
    complete = {
        "modality": "uncertain", "tissue_type": "uncertain",
        "notable_features": "x", "image_quality": "good",
    }
    assert has_required_fields(complete)
