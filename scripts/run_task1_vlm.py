"""
scripts/run_task1_vlm.py

Task 1 (part 2): multimodal LLM description via Ollama.

MUST BE RUN LOCALLY on a machine with Ollama running and `llama3.2-vision`
pulled. NOTE: if your Ollama version is 0.30.0 or later, this will fail
with "unknown model architecture: 'mllama'" -- a known upstream bug, see
README.md for the fix (downgrade Ollama, or use colab/Task1_VLM_Colab.ipynb).

Sends the representative training image (train_062: 27 nuclei, "normal"
density, closest to the median for that regime -- not cherry-picked) to
llama3.2-vision under two prompting strategies (naive vs. optimised),
and separately demonstrates that repeated calls to the same prompt are
NOT identical.

Saves:
    outputs/records/task1_prompt_comparison.txt   (for direct inclusion in report)
    outputs/records/task1_repeated_runs.txt
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imaging_pipeline.config import TRAIN_DIR, RECORDS_DIR, VLM_MODEL
from imaging_pipeline.vlm_description import (
    run_prompt_comparison, demonstrate_repeated_run_variability, build_optimised_prompt,
)

REPRESENTATIVE_IMAGE = TRAIN_DIR / "images" / "train_062.png"


def main():
    if not REPRESENTATIVE_IMAGE.exists():
        raise FileNotFoundError(f"Expected representative image at {REPRESENTATIVE_IMAGE}")

    print(f"Using representative image: {REPRESENTATIVE_IMAGE.name} "
          f"(27 nuclei, 'normal' density, closest to median for that regime)")
    print(f"Model: {VLM_MODEL}\n")

    # --- Naive vs optimised prompt comparison (temperature=0, reproducible) ---
    print("Running naive vs. optimised prompt comparison (temperature=0)...")
    t0 = time.time()
    comparison = run_prompt_comparison(REPRESENTATIVE_IMAGE, model=VLM_MODEL)
    print(f"Done in {time.time()-t0:.1f}s\n")

    print("=== NAIVE PROMPT RESPONSE ===")
    print(comparison["naive_response"])
    print("\n=== OPTIMISED PROMPT RESPONSE (raw) ===")
    print(comparison["optimised_response_raw"])
    print("\n=== OPTIMISED PROMPT: PARSED RECORD ===")
    print(comparison["optimised_record"])
    print(f"\nValid (has all 4 required fields): {comparison['optimised_record_valid']}")

    with open(RECORDS_DIR / "task1_prompt_comparison.txt", "w") as f:
        f.write(f"Representative image: {REPRESENTATIVE_IMAGE.name}\n")
        f.write(f"Model: {VLM_MODEL}\n\n")
        f.write("=" * 70 + "\nNAIVE PROMPT\n" + "=" * 70 + "\n")
        f.write(comparison["naive_prompt"] + "\n\n")
        f.write("--- Response ---\n")
        f.write(comparison["naive_response"] + "\n\n")
        f.write("=" * 70 + "\nOPTIMISED PROMPT\n" + "=" * 70 + "\n")
        f.write(comparison["optimised_prompt"] + "\n\n")
        f.write("--- Raw response ---\n")
        f.write(comparison["optimised_response_raw"] + "\n\n")
        f.write("--- Parsed JSON record ---\n")
        f.write(str(comparison["optimised_record"]) + "\n")
        f.write(f"\nValid (all 4 required fields present): {comparison['optimised_record_valid']}\n")
    print(f"\nSaved comparison to {RECORDS_DIR / 'task1_prompt_comparison.txt'}")

    # --- Repeated-run variability (non-zero temperature, deliberately) ---
    print("\n\nRunning repeated-run variability check (optimised prompt, "
          "temperature=0.8, 3 runs)...")
    optimised_prompt = build_optimised_prompt()
    t0 = time.time()
    runs = demonstrate_repeated_run_variability(
        REPRESENTATIVE_IMAGE, optimised_prompt, model=VLM_MODEL, n_runs=3, temperature=0.8
    )
    print(f"Done in {time.time()-t0:.1f}s\n")

    all_identical = len(set(runs)) == 1
    print(f"All 3 runs produced identical text: {all_identical}")
    for i, r in enumerate(runs):
        print(f"\n--- Run {i+1} ---")
        print(r)

    with open(RECORDS_DIR / "task1_repeated_runs.txt", "w") as f:
        f.write(f"Repeated-run variability check\n")
        f.write(f"Image: {REPRESENTATIVE_IMAGE.name}, model: {VLM_MODEL}, "
                f"temperature=0.8, n_runs={len(runs)}\n\n")
        f.write(f"All runs identical: {all_identical}\n\n")
        for i, r in enumerate(runs):
            f.write(f"--- Run {i+1} ---\n{r}\n\n")
    print(f"\nSaved repeated-run outputs to {RECORDS_DIR / 'task1_repeated_runs.txt'}")

    print(
        "\nNext step: send outputs/records/task1_prompt_comparison.txt and "
        "task1_repeated_runs.txt back so the Task 1 write-up can be finished."
    )


if __name__ == "__main__":
    main()
