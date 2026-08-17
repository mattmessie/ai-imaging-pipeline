"""
scripts/run_task1_eda.py

Task 1 (part 1): data preparation and EDA.

Loads all training images, converts to grayscale, confirms resize to a
common 256x256 size, and produces:
    outputs/figures/task1_sample_grid.png
    outputs/figures/task1_intensity_histogram.png
    outputs/metrics/task1_eda_summary.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from imaging_pipeline.config import TRAIN_DIR, FIGURE_DIR, METRICS_DIR, IMAGE_SIZE
from imaging_pipeline.data_prep import preprocess_split, plot_sample_grid, plot_intensity_histogram


def main():
    print(f"Loading and preprocessing training images from {TRAIN_DIR}...")
    records = preprocess_split(TRAIN_DIR)
    print(f"Loaded {len(records)} images.")

    shapes = {rec["gray"].shape for rec in records}
    print(f"Grayscale shapes after resize: {shapes} (target: {IMAGE_SIZE})")
    assert shapes == {IMAGE_SIZE}, "Not all images resized to the common target size!"

    print("\nGenerating sample grid...")
    plot_sample_grid(records, n=8, save_path=FIGURE_DIR / "task1_sample_grid.png")

    print("Generating intensity histogram...")
    plot_intensity_histogram(records, save_path=FIGURE_DIR / "task1_intensity_histogram.png")

    all_pixels = np.concatenate([rec["gray"].ravel() for rec in records])
    per_image_means = np.array([rec["gray"].mean() for rec in records])
    per_image_stds = np.array([rec["gray"].std() for rec in records])

    summary = (
        f"Task 1 EDA summary\n"
        f"===================\n"
        f"Number of training images: {len(records)}\n"
        f"Image size after resize: {IMAGE_SIZE}\n"
        f"\nPooled pixel intensity (grayscale, [0,1]):\n"
        f"  mean: {all_pixels.mean():.4f}\n"
        f"  std:  {all_pixels.std():.4f}\n"
        f"  min:  {all_pixels.min():.4f}\n"
        f"  max:  {all_pixels.max():.4f}\n"
        f"  median: {np.median(all_pixels):.4f}\n"
        f"\nPer-image mean intensity:\n"
        f"  mean of means: {per_image_means.mean():.4f}\n"
        f"  std of means:  {per_image_means.std():.4f}\n"
        f"  range: [{per_image_means.min():.4f}, {per_image_means.max():.4f}]\n"
        f"\nPer-image intensity std (contrast proxy):\n"
        f"  mean: {per_image_stds.mean():.4f}\n"
        f"  range: [{per_image_stds.min():.4f}, {per_image_stds.max():.4f}]\n"
    )
    print("\n" + summary)

    with open(METRICS_DIR / "task1_eda_summary.txt", "w") as f:
        f.write(summary)

    print(f"Saved figures to {FIGURE_DIR}")
    print(f"Saved summary to {METRICS_DIR / 'task1_eda_summary.txt'}")
    print(
        "\nNote for report: pooled intensity histogram is clearly bimodal "
        "(background spike near 0, valley ~0.15, foreground bump 0.2-0.35) "
        "-- good evidence Otsu thresholding (Task 2) is a reasonable choice "
        "for this modality."
    )


if __name__ == "__main__":
    main()
