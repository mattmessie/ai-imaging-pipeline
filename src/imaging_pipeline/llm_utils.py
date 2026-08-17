"""Shared utilities for text-only local LLM calls (used by Task 2's
numbers-first interpretation and Task 4's hybrid pipeline). Kept separate
from vlm_description.py since these calls never send an image -- the
model gets numbers/text only.
"""

import json

from imaging_pipeline.config import TEXT_LLM_MODEL


def query_text_llm(prompt: str, model: str = TEXT_LLM_MODEL, temperature: float = 0.0) -> str:
    """Send a text-only prompt to a local Ollama model, return the raw
    response text. No `images` key -- this never sends visual data.
    """
    from ollama import chat  # deferred import, see vlm_description.py for why

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def parse_json_and_narrative(text: str) -> tuple:
    """Split a combined 'JSON:\\n{...}\\n\\nNARRATIVE:\\n<paragraph>'
    response into (parsed_dict, narrative_str).

    Same pattern as the course's lab5 hybrid pipeline
    (parse_hybrid_response): tolerant of markdown code fences, and
    returns an {"error": ...} dict rather than raising if the JSON
    section doesn't parse, so one bad response never crashes a batch run.
    """
    text = text.strip()

    after_json = text.split("JSON:", 1)[1] if "JSON:" in text else text

    if "NARRATIVE:" in after_json:
        json_part, narrative_part = after_json.split("NARRATIVE:", 1)
    else:
        json_part, narrative_part = after_json, ""

    json_part = json_part.strip()
    if json_part.startswith("```"):
        json_part = json_part.split("\n", 1)[1] if "\n" in json_part else json_part
        if json_part.rstrip().endswith("```"):
            json_part = json_part.rstrip()[:-3]
        json_part = json_part.strip()

    try:
        record = json.loads(json_part)
    except json.JSONDecodeError:
        record = {"error": "could not parse JSON", "raw": json_part}

    return record, narrative_part.strip()
