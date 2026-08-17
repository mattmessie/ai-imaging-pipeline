"""
scripts/run_task2_llm.py

Task 2 (part 2): numbers-first LLM interpretation.

Runs LOCALLY on your Mac's regular Ollama install -- this only needs the
text model `llama3.2`, not the broken `llama3.2-vision`, so no Colab
workaround needed here.

Uses the same representative image as Task 1 (train_062) for direct
comparability, per the brief's explicit instruction to "compare this
numbers-first description against the direct image description from
Task 1."

Saves:
    outputs/records/task2_llm_interpretation.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.config import TRAIN_DIR, RECORDS_DIR, TEXT_LLM_MODEL
from imaging_pipeline.data_prep import load_rgb, to_grayscale
from imaging_pipeline.classical_features import otsu_segment, extract_region_features, summarise_features
from imaging_pipeline.llm_interpretation import run_numbers_first_interpretation

REPRESENTATIVE_IMAGE = "train_062"


def main():
    print(f"Recomputing Otsu features for {REPRESENTATIVE_IMAGE}...")
    rgb = load_rgb(TRAIN_DIR / "images" / f"{REPRESENTATIVE_IMAGE}.png")
    gray = to_grayscale(rgb)
    mask = otsu_segment(gray)
    df = extract_region_features(gray, mask)
    summary = summarise_features(df, REPRESENTATIVE_IMAGE)
    print(f"Summary (this is ALL the LLM will see -- no image): {summary}\n")

    print(f"Querying {TEXT_LLM_MODEL} with the numbers-first prompt...")
    result = run_numbers_first_interpretation(REPRESENTATIVE_IMAGE, summary, model=TEXT_LLM_MODEL)

    print("\n=== RAW RESPONSE ===")
    print(result["raw_response"])
    print("\n=== PARSED JSON RECORD ===")
    print(result["record"])
    print("\n=== NARRATIVE ===")
    print(result["narrative"])

    with open(RECORDS_DIR / "task2_llm_interpretation.txt", "w") as f:
        f.write(f"Image: {REPRESENTATIVE_IMAGE}\n")
        f.write(f"Model: {TEXT_LLM_MODEL}\n\n")
        f.write("--- Summary text passed to the LLM (numbers only, no image) ---\n")
        f.write(summary + "\n\n")
        f.write("--- Prompt ---\n")
        f.write(result["prompt"] + "\n\n")
        f.write("--- Raw response ---\n")
        f.write(result["raw_response"] + "\n\n")
        f.write("--- Parsed JSON record ---\n")
        f.write(str(result["record"]) + "\n\n")
        f.write("--- Narrative ---\n")
        f.write(result["narrative"] + "\n")

    print(f"\nSaved to {RECORDS_DIR / 'task2_llm_interpretation.txt'}")
    print(
        "\nNext step: send this file back so it can be compared against "
        "Task 1's direct VLM description for the report (Part 9, Q1)."
    )


if __name__ == "__main__":
    main()
