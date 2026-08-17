"""
scripts/run_task4_stage1.py

Task 4, stages 1-3: run the trained U-Net on all 12 unseen test images,
extract regionprops features, build the deterministic summary + prompt
for each. No Ollama needed for this part -- fully runnable anywhere.

Saves everything the LLM stage needs as a pickle, so
scripts/run_task4_llm_batch.py (which DOES need Ollama, run on your Mac)
doesn't have to repeat any of this.

Saves:
    outputs/records/task4_stage1_inputs.pkl
    outputs/figures/task4_unet_masks_sample.png
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import torch

from imaging_pipeline.config import TEST_DIR, FIGURE_DIR, RECORDS_DIR, OUTPUT_DIR
from imaging_pipeline.data_prep import load_rgb, to_grayscale
from imaging_pipeline.unet import UNet
from imaging_pipeline.hybrid_pipeline import prepare_pipeline_inputs

MODEL_PATH = OUTPUT_DIR / "model_objects" / "unet_trained.pth"


def main():
    device = torch.device("cpu")
    model = UNet(in_ch=1, out_ch=1, base=16).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded trained U-Net from {MODEL_PATH}")

    test_images = sorted((TEST_DIR / "images").glob("*.png"))
    print(f"Running pipeline stages 1-3 on {len(test_images)} test images...")

    all_inputs = []
    for img_path in test_images:
        stem = img_path.stem
        rgb = load_rgb(img_path)
        gray = to_grayscale(rgb)
        result = prepare_pipeline_inputs(stem, gray, model, device)
        result["gray"] = gray  # keep for the sample visualization below
        all_inputs.append(result)
        print(f"  {stem}: {len(result['features_df'])} objects detected -> {result['summary_text']}")

    with open(RECORDS_DIR / "task4_stage1_inputs.pkl", "wb") as f:
        pickle.dump(all_inputs, f)
    print(f"\nSaved stage 1-3 outputs (prompts included) to {RECORDS_DIR / 'task4_stage1_inputs.pkl'}")

    # Sample visualization: 4 test images with their U-Net masks
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, result in enumerate(all_inputs[:4]):
        axes[0, i].imshow(result["gray"], cmap="gray")
        axes[0, i].set_title(result["image_id"])
        axes[0, i].axis("off")
        axes[1, i].imshow(result["mask"], cmap="gray")
        axes[1, i].set_title(f"{len(result['features_df'])} objects")
        axes[1, i].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "task4_unet_masks_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample figure to {FIGURE_DIR / 'task4_unet_masks_sample.png'}")

    print(
        "\nNext step: run scripts/run_task4_llm_batch.py locally (needs "
        "Ollama with llama3.2) to complete stage 4 for all 12 images."
    )


if __name__ == "__main__":
    main()
