"""Task 2 (part 2): numbers-first LLM interpretation.

The LLM receives ONLY the deterministic text summary from
classical_features.summarise_features() -- never the image itself. This
is the direct comparison point against Task 1's direct-image VLM
description (Part 9, Q1: "which description is more useful, and which is
more trustworthy... and why?").
"""

from imaging_pipeline.llm_utils import query_text_llm, parse_json_and_narrative
from imaging_pipeline.config import TEXT_LLM_MODEL


def build_numbers_first_prompt(image_id: str, summary_text: str) -> str:
    return (
        "You are a scientific image-cataloguing assistant. You have NOT seen "
        "the image itself -- you are working ONLY from a numeric summary "
        "produced by classical image segmentation (Otsu thresholding). Do "
        "not claim to have observed anything beyond what the summary states, "
        "and do not provide any clinical or diagnostic interpretation.\n\n"
        f"Image ID: {image_id}\n"
        f"Summary: {summary_text}\n\n"
        "Based ONLY on this summary, produce:\n\n"
        "1. A JSON object with EXACTLY these four fields:\n"
        "{\n"
        '  "n_objects": <integer, from the summary>,\n'
        '  "density_class": "<one of: sparse, moderate, dense>",\n'
        '  "shape_regularity": "<one of: highly irregular, irregular, '
        'regular, highly regular>",\n'
        '  "quality_flag": "<one of: ok, review_recommended, fail>"\n'
        "}\n\n"
        "2. Then a one-paragraph description (3-4 sentences) suitable for a "
        "research report, grounded strictly in the summary's numbers.\n\n"
        "Format your response EXACTLY as:\n\n"
        "JSON:\n{...}\n\nNARRATIVE:\n<paragraph>"
    )


def run_numbers_first_interpretation(image_id: str, summary_text: str, model: str = TEXT_LLM_MODEL) -> dict:
    """Run the numbers-first prompt and parse the result.

    Returns a dict with the prompt, raw response, parsed record, and
    narrative -- enough to save directly for the report.
    """
    prompt = build_numbers_first_prompt(image_id, summary_text)
    raw_response = query_text_llm(prompt, model=model, temperature=0.0)
    record, narrative = parse_json_and_narrative(raw_response)

    return {
        "image_id": image_id,
        "model": model,
        "summary_text": summary_text,
        "prompt": prompt,
        "raw_response": raw_response,
        "record": record,
        "narrative": narrative,
    }
