"""
scripts/run_task4_llm_batch.py

Task 4, stage 4: batch LLM calls for all 12 test images -- the only part
of Task 4 needing Ollama. Uses `llama3.2` (text-only), so this runs on
your regular local Mac Ollama install, same as Task 2.

Loads the prompts already built by run_task4_stage1.py (U-Net inference
+ regionprops + summary text -- run once, in the sandbox, on real data),
so this script only has to do the actual Ollama calls, not repeat any of
the deterministic pipeline stages.

Saves:
    outputs/metrics/task4_hybrid_results.csv       (the Task 4 deliverable)
    outputs/records/task4_full_records.txt         (raw responses + narratives)
"""

import sys
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.config import RECORDS_DIR, METRICS_DIR, TEXT_LLM_MODEL
from imaging_pipeline.hybrid_pipeline import run_llm_stage, aggregate_records


def main():
    with open(RECORDS_DIR / "task4_stage1_inputs.pkl", "rb") as f:
        stage1_inputs = pickle.load(f)

    print(f"Loaded stage 1-3 outputs for {len(stage1_inputs)} test images.")
    print(f"Running stage 4 (LLM call) with {TEXT_LLM_MODEL} for each...\n")

    records = []
    narratives = {}
    raw_responses = {}
    invalid = []

    t0 = time.time()
    for i, item in enumerate(stage1_inputs):
        image_id = item["image_id"]
        print(f"  [{i+1}/{len(stage1_inputs)}] {image_id}...", end=" ")
        result = run_llm_stage(item["prompt"], model=TEXT_LLM_MODEL)

        record = result["record"]
        if not result["record_valid"]:
            invalid.append(image_id)
            print("WARNING: missing required fields")
        else:
            print("ok")

        records.append(record)
        narratives[image_id] = result["narrative"]
        raw_responses[image_id] = result["raw_response"]

    print(f"\nAll {len(stage1_inputs)} images processed in {time.time()-t0:.1f}s")
    if invalid:
        print(f"WARNING: {len(invalid)} image(s) had invalid/incomplete JSON: {invalid}")

    df = aggregate_records(records)
    print(f"\n{df.to_string(index=False)}")

    df.to_csv(METRICS_DIR / "task4_hybrid_results.csv", index=False)
    print(f"\nSaved aggregated CSV to {METRICS_DIR / 'task4_hybrid_results.csv'}")

    with open(RECORDS_DIR / "task4_full_records.txt", "w") as f:
        for item in stage1_inputs:
            image_id = item["image_id"]
            f.write(f"{'='*70}\n{image_id}\n{'='*70}\n")
            f.write(f"Summary passed to LLM: {item['summary_text']}\n\n")
            f.write(f"Raw response:\n{raw_responses[image_id]}\n\n")
            f.write(f"Narrative:\n{narratives[image_id]}\n\n")
    print(f"Saved full records (raw responses + narratives) to "
          f"{RECORDS_DIR / 'task4_full_records.txt'}")

    print(
        "\nNext step: send back task4_hybrid_results.csv and "
        "task4_full_records.txt so Task 4's write-up can be finished."
    )


if __name__ == "__main__":
    main()
