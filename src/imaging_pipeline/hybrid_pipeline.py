"""Task 4: hybrid pipeline.

image -> U-Net mask -> regionprops feature table -> LLM structured JSON
record (image_id, n_objects, mean_area, density_class, quality_flag) ->
one-paragraph narrative.

Split into two stages deliberately:
- `prepare_pipeline_inputs()`: everything up to and including building the
  prompt -- U-Net inference, feature extraction, summary text. No Ollama
  needed, fully runnable/testable anywhere.
- `run_llm_stage()`: the actual LLM call + parsing. Needs a local Ollama
  server.

This split means the expensive/deterministic parts (U-Net inference on 12
test images) don't need to be repeated just because the LLM call has to
run on a different machine.
"""

import pandas as pd

from imaging_pipeline.classical_features import extract_region_features, summarise_features
from imaging_pipeline.llm_utils import query_text_llm, parse_json_and_narrative
from imaging_pipeline.train_unet import predict_mask
from imaging_pipeline.config import TEXT_LLM_MODEL

REQUIRED_FIELDS = ("image_id", "n_objects", "mean_area", "density_class", "quality_flag")


def build_hybrid_prompt(image_id: str, summary_text: str) -> str:
    return (
        "You are a scientific image-cataloguing assistant helping annotate "
        "microscopy images for a research dataset. You have NOT seen the "
        "image itself -- you are working ONLY from a numeric summary "
        "produced by U-Net segmentation. Do not claim to have observed "
        "anything beyond what the summary states, and do not provide any "
        "clinical or diagnostic interpretation.\n\n"
        f"Image ID: {image_id}\n"
        f"Summary: {summary_text}\n\n"
        "Based ONLY on this summary, produce:\n\n"
        "1. A JSON object with EXACTLY these five fields:\n"
        "{\n"
        f'  "image_id": "{image_id}",\n'
        '  "n_objects": <integer, from the summary>,\n'
        '  "mean_area": <number, the mean object area from the summary>,\n'
        '  "density_class": "<one of: sparse, moderate, dense>",\n'
        '  "quality_flag": "<one of: ok, review_recommended, fail>"\n'
        "}\n\n"
        "2. Then a one-paragraph description (3-4 sentences) suitable for a "
        "research report, grounded strictly in the summary's numbers.\n\n"
        "Format your response EXACTLY as:\n\n"
        "JSON:\n{...}\n\nNARRATIVE:\n<paragraph>"
    )


def has_required_fields(record: dict) -> bool:
    return all(field in record for field in REQUIRED_FIELDS)


def prepare_pipeline_inputs(image_id: str, gray_image, model, device) -> dict:
    """Stages 1-3: U-Net segmentation -> regionprops -> summary -> prompt.
    No LLM call here -- returns everything needed for `run_llm_stage`.
    """
    mask = predict_mask(model, gray_image, device)
    features_df = extract_region_features(gray_image, mask)
    summary_text = summarise_features(features_df, image_id, method="U-Net segmentation")
    prompt = build_hybrid_prompt(image_id, summary_text)

    return {
        "image_id": image_id,
        "mask": mask,
        "features_df": features_df,
        "summary_text": summary_text,
        "prompt": prompt,
    }


def run_llm_stage(prompt: str, model: str = TEXT_LLM_MODEL) -> dict:
    """Stage 4: the actual LLM call + parsing. Needs a local Ollama server."""
    raw_response = query_text_llm(prompt, model=model, temperature=0.0)
    record, narrative = parse_json_and_narrative(raw_response)
    return {
        "raw_response": raw_response,
        "record": record,
        "narrative": narrative,
        "record_valid": has_required_fields(record),
    }


def aggregate_records(records: list) -> pd.DataFrame:
    """Aggregate a list of per-image JSON records (dicts) into one
    DataFrame, ready to save as the Task 4 deliverable CSV.
    """
    return pd.DataFrame(records)
