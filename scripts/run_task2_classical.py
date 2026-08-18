"""
scripts/run_task2_classical.py

Task 2 (part 1): Otsu thresholding + morphological cleanup + regionprops
feature table, on the same representative image used in Task 1
(train_062: 27 nuclei, "normal" density).

Also runs Otsu across all 12 test images and saves per-image object counts
against ground truth (metadata.csv) -- useful evidence for Part 9 Q2
("did the U-Net improve on classical Otsu segmentation... give one example
image where each approach did better") once Task 3's U-Net exists.

Saves:
    outputs/figures/task2_otsu_segmentation.png
    outputs/metrics/task2_feature_table.csv
    outputs/metrics/task2_otsu_vs_groundtruth.csv
    outputs/records/task2_summary_text.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from imaging_pipeline.config import TRAIN_DIR, TEST_DIR, DATA_DIR, FIGURE_DIR, METRICS_DIR, RECORDS_DIR
from imaging_pipeline.data_prep import load_rgb, to_grayscale
from imaging_pipeline.classical_features import otsu_segment, extract_region_features, summarise_features

REPRESENTATIVE_IMAGE = "train_062"


def main():
    # --- Representative image: full walkthrough + visualization ---
    rgb = load_rgb(TRAIN_DIR / "images" / f"{REPRESENTATIVE_IMAGE}.png")
    gray = to_grayscale(rgb)

    mask = otsu_segment(gray)
    df = extract_region_features(gray, mask)
    summary = summarise_features(df, REPRESENTATIVE_IMAGE)

    print(f"Otsu segmentation on {REPRESENTATIVE_IMAGE}:")
    print(f"  Objects detected: {len(df)}")
    print(f"\nFeature table:\n{df}")
    print(f"\nSummary text (this is what the LLM sees, NOT the image):\n{summary}")

    df.to_csv(METRICS_DIR / "task2_feature_table.csv", index=False)
    with open(RECORDS_DIR / "task2_summary_text.txt", "w") as f:
        f.write(f"Image: {REPRESENTATIVE_IMAGE}\n\n{summary}\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(f"{REPRESENTATIVE_IMAGE} (grayscale)")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Otsu + cleanup ({len(df)} objects)")
    axes[1].axis("off")
    overlay = rgb.copy()
    overlay[mask] = [1, 0, 0]
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay on original")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "task2_otsu_segmentation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- All 12 test images: Otsu object count vs ground truth ---
    print("\n\nRunning Otsu across all 12 test images (for later U-Net comparison)...")
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    test_meta = metadata[metadata["split"] == "test"].set_index("image_id")

    rows = []
    for img_path in sorted((TEST_DIR / "images").glob("*.png")):
        stem = img_path.stem
        rgb_t = load_rgb(img_path)
        gray_t = to_grayscale(rgb_t)
        mask_t = otsu_segment(gray_t)
        df_t = extract_region_features(gray_t, mask_t)
        gt_n = int(test_meta.loc[stem, "n_objects"]) if stem in test_meta.index else None
        rows.append({
            "image_id": stem,
            "density": test_meta.loc[stem, "density"] if stem in test_meta.index else None,
            "ground_truth_n_objects": gt_n,
            "otsu_n_objects": len(df_t),
            "difference": len(df_t) - gt_n if gt_n is not None else None,
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(METRICS_DIR / "task2_otsu_vs_groundtruth.csv", index=False)
    print(f"\n{comparison.to_string(index=False)}")
    print(f"\nMean absolute difference (Otsu count - ground truth): "
          f"{comparison['difference'].abs().mean():.2f} objects")

    print(f"\nSaved figure to {FIGURE_DIR / 'task2_otsu_segmentation.png'}")
    print(f"Saved feature table to {METRICS_DIR / 'task2_feature_table.csv'}")
    print(f"Saved Otsu-vs-groundtruth comparison to {METRICS_DIR / 'task2_otsu_vs_groundtruth.csv'}")
    print(f"Saved summary text to {RECORDS_DIR / 'task2_summary_text.txt'}")


if __name__ == "__main__":
    main()
