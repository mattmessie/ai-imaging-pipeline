"""Task 1 (part 2): multimodal LLM description via Ollama.

Sends a representative image to a local vision model (llama3.2-vision) and
compares two prompting strategies:

- NAIVE_PROMPT: an unstructured, open-ended prompt ("what is this?") --
  the kind of prompt a first attempt would use, with no safety framing,
  no forced structure, no permission to express uncertainty.
- build_optimised_prompt(): anchors the model as descriptive rather than
  diagnostic, forces a fixed-schema JSON record (modality, tissue_type,
  notable_features, image_quality), and explicitly permits "uncertain"
  for any field the model isn't confident about.

Also demonstrates that repeated calls to the same prompt are NOT
identical (unlike a temperature=0 call, which would be deterministic --
here we deliberately use the model's default sampling temperature to show
genuine run-to-run variability, since that's what the assignment asks to
be shown, not hidden).
"""

import json
from pathlib import Path

from imaging_pipeline.config import VLM_MODEL, MODALITY_NAME


NAIVE_PROMPT = "What is in this image?"


def build_optimised_prompt() -> str:
    """The engineered prompt: descriptive framing, forced JSON schema,
    explicit permission to say "uncertain". Kept as a function (not a
    bare constant) so the exact text used is easy to log/version if it
    gets iterated on further.
    """
    return (
        "You are a scientific image-cataloguing assistant helping annotate "
        "microscopy images for a research dataset.\n\n"
        "You are NOT a diagnostic tool. Do not provide any clinical, "
        "diagnostic, or disease-related interpretation, even if asked. "
        "Describe only what is visually present, in purely observational "
        "terms (shapes, brightness, spatial arrangement, apparent density, "
        "texture).\n\n"
        "Respond with a JSON object with EXACTLY these four fields, and "
        "nothing else:\n"
        "{\n"
        '  "modality": "<imaging modality if visually identifiable, else '
        '\\"uncertain\\">",\n'
        '  "tissue_type": "<general sample/tissue type if identifiable, '
        'else \\"uncertain\\">",\n'
        '  "notable_features": "<one short factual sentence on visually '
        'notable features>",\n'
        '  "image_quality": "<one of: good, moderate, poor>"\n'
        "}\n\n"
        "If you are not confident about a field, write \"uncertain\" for "
        "that field rather than guessing. Respond with ONLY the JSON "
        "object -- no preamble, no explanation, no markdown code fences."
    )


def query_vlm(image_path: Path, prompt: str, model: str = VLM_MODEL, temperature: float = None) -> str:
    """Send one image + prompt to a local Ollama vision model, return the
    raw response text.

    `temperature=None` uses Ollama's default sampling temperature (not
    deterministic) -- this is deliberate for the repeated-run-variability
    demonstration. Pass `temperature=0.0` explicitly for a reproducible
    single description.
    """
    from ollama import chat  # imported here so this module can be imported
    # and unit-tested (see tests/test_vlm_description.py) on a machine
    # without the `ollama` package/server available.

    options = {}
    if temperature is not None:
        options["temperature"] = temperature

    response = chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [str(image_path)],
        }],
        options=options,
    )
    return response["message"]["content"]


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response, tolerating markdown
    code fences and leading/trailing prose. Returns {"error": ..., "raw":
    text} on failure rather than raising, so a single bad response doesn't
    crash a batch run.
    """
    text = text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    # If there's leading/trailing prose around the JSON object, try to
    # isolate the outermost {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
    else:
        candidate = text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {"error": "could not parse JSON", "raw": text}


def has_required_fields(record: dict, required=("modality", "tissue_type", "notable_features", "image_quality")) -> bool:
    """Whether a parsed record has every field the optimised prompt asks for."""
    return all(field in record for field in required)


def run_prompt_comparison(image_path: Path, model: str = VLM_MODEL) -> dict:
    """Run both the naive and optimised prompts on one image, once each,
    at temperature=0 for a clean, reproducible side-by-side comparison.

    Returns a dict with the raw text and (for the optimised prompt) the
    parsed JSON record for both.
    """
    naive_raw = query_vlm(image_path, NAIVE_PROMPT, model=model, temperature=0.0)

    optimised_prompt = build_optimised_prompt()
    optimised_raw = query_vlm(image_path, optimised_prompt, model=model, temperature=0.0)
    optimised_record = parse_json_response(optimised_raw)

    return {
        "image_path": str(image_path),
        "model": model,
        "naive_prompt": NAIVE_PROMPT,
        "naive_response": naive_raw,
        "optimised_prompt": optimised_prompt,
        "optimised_response_raw": optimised_raw,
        "optimised_record": optimised_record,
        "optimised_record_valid": has_required_fields(optimised_record),
    }


def demonstrate_repeated_run_variability(image_path: Path, prompt: str, model: str = VLM_MODEL,
                                          n_runs: int = 3, temperature: float = 0.8) -> list:
    """Call the same prompt on the same image `n_runs` times at a
    non-zero temperature, to show responses are NOT identical run to run
    (the assignment explicitly asks this be demonstrated, not just
    asserted). Returns the list of raw responses.
    """
    return [query_vlm(image_path, prompt, model=model, temperature=temperature) for _ in range(n_runs)]
