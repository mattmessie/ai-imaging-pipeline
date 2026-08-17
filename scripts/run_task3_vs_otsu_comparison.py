"""
scripts/run_task3_vs_otsu_comparison.py

Direct comparison of the trained U-Net against Task 2's Otsu segmentation
on the SAME 12 test images, for Part 9 Q2 ("did the U-Net improve on
classical Otsu segmentation for your modality? Give one example image
where each approach did better").

Object counts are derived from the U-Net's predicted mask the same way as
Otsu's (connected-component labeling via skimage.measure.label), so the
two are directly comparable.

Saves:
    outputs/metrics/task3_unet_vs_otsu_vs_groundtruth.csv
    outputs/figures/task3_unet_vs_otsu_examples.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from skimage import measure

from imaging_pipeline.config import TEST_DIR, DATA_DIR, FIGURE_DIR, METRICS_DIR, OUTPUT_DIR
from imaging_pipeline.data_prep import load_rgb, to_grayscale
from imaging_pipeline.classical_features import otsu_segment
from imaging_pipeline.unet import UNet
from imaging_pipeline.train_unet import dice_coefficient, iou_score, predict_mask
from skimage import io as skio

MODEL_PATH = OUTPUT_DIR / "model_objects" / "unet_trained.pth"


def count_objects(binary_mask):
    labels = measure.label(binary_mask)
    return int(labels.max())


def main():
    device = torch.device("cpu")
    model = UNet(in_ch=1, out_ch=1, base=16).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded trained U-Net from {MODEL_PATH}")

    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    test_meta = metadata[metadata["split"] == "test"].set_index("image_id")

    otsu_prev = pd.read_csv(METRICS_DIR / "task2_otsu_vs_groundtruth.csv").set_index("image_id")

    rows = []
    masks_for_plot = {}
    for img_path in sorted((TEST_DIR / "images").glob("*.png")):
        stem = img_path.stem
        rgb = load_rgb(img_path)
        gray = to_grayscale(rgb)

        otsu_mask = otsu_segment(gray)
        unet_mask = predict_mask(model, gray, device)

        gt_n = int(test_meta.loc[stem, "n_objects"])
        otsu_n = int(otsu_prev.loc[stem, "otsu_n_objects"])
        unet_n = count_objects(unet_mask)

        # Pixel-level Dice/IoU against the real ground-truth binary mask --
        # the metric both models were actually trained/designed against,
        # as opposed to object counts (see note below on why counts alone
        # are a misleading comparison here).
        gt_mask = (skio.imread(TEST_DIR / "masks" / img_path.name) > 0)
        gt_t = torch.from_numpy(gt_mask.astype("float32")).unsqueeze(0).unsqueeze(0)
        otsu_t = torch.from_numpy(otsu_mask.astype("float32")).unsqueeze(0).unsqueeze(0)
        unet_t = torch.from_numpy(unet_mask.astype("float32")).unsqueeze(0).unsqueeze(0)
        # dice_coefficient/iou_score expect logits (they apply sigmoid+threshold
        # internally); feed pre-thresholded 0/1 masks scaled so sigmoid(x)>0.5
        # reproduces them exactly (large +/- logit trick).
        otsu_logits = (otsu_t * 20) - 10
        unet_logits = (unet_t * 20) - 10
        otsu_dice = dice_coefficient(otsu_logits, gt_t)
        otsu_iou = iou_score(otsu_logits, gt_t)
        unet_dice = dice_coefficient(unet_logits, gt_t)
        unet_iou = iou_score(unet_logits, gt_t)

        rows.append({
            "image_id": stem,
            "density": test_meta.loc[stem, "density"],
            "ground_truth_n_objects": gt_n,
            "otsu_n_objects": otsu_n,
            "otsu_error": abs(otsu_n - gt_n),
            "otsu_dice": otsu_dice,
            "otsu_iou": otsu_iou,
            "unet_n_objects": unet_n,
            "unet_error": abs(unet_n - gt_n),
            "unet_dice": unet_dice,
            "unet_iou": unet_iou,
        })
        masks_for_plot[stem] = (gray, otsu_mask, unet_mask)

    comparison = pd.DataFrame(rows)
    comparison["unet_better_count"] = comparison["unet_error"] < comparison["otsu_error"]
    comparison["unet_better_dice"] = comparison["unet_dice"] > comparison["otsu_dice"]
    comparison.to_csv(METRICS_DIR / "task3_unet_vs_otsu_vs_groundtruth.csv", index=False)

    print("\n" + comparison.to_string(index=False))
    print(f"\n--- Object counting (both derived by connected-component labeling) ---")
    print(f"Mean absolute error -- Otsu: {comparison['otsu_error'].mean():.2f}, "
          f"U-Net: {comparison['unet_error'].mean():.2f}")
    print(f"U-Net closer to ground truth on {comparison['unet_better_count'].sum()}/12 images")
    print(
        "\nNote: object counts are nearly identical between Otsu and U-Net because "
        "the ground-truth BINARY masks themselves merge touching nuclei into single "
        "connected components (only the separate instance-label masks distinguish "
        "them) -- the U-Net was trained on binary masks, so it faithfully reproduces "
        "the same merged-blob shapes Otsu also produces, not because it has failed, "
        "but because binary segmentation and instance separation are different tasks. "
        "This matches the course lecture's own 'Common Failure Modes' guidance: fixing "
        "this needs watershed post-processing or an instance-segmentation head, not a "
        "better binary segmenter."
    )
    print(f"\n--- Pixel-level accuracy (the metric each model actually optimises) ---")
    print(f"Mean Dice -- Otsu: {comparison['otsu_dice'].mean():.4f}, U-Net: {comparison['unet_dice'].mean():.4f}")
    print(f"Mean IoU  -- Otsu: {comparison['otsu_iou'].mean():.4f}, U-Net: {comparison['unet_iou'].mean():.4f}")
    print(f"U-Net better Dice on {comparison['unet_better_dice'].sum()}/12 images")

    # Pick one clear example of each, based on pixel-level Dice (the
    # meaningful comparison -- see note above on why object counts alone
    # aren't a fair comparison here).
    unet_wins = comparison.sort_values("otsu_dice", ascending=True).iloc[0]  # Otsu's worst Dice
    otsu_candidates = comparison[comparison["otsu_dice"] >= comparison["unet_dice"]]
    otsu_wins = otsu_candidates.sort_values("otsu_dice", ascending=False).iloc[0] if len(otsu_candidates) else comparison.sort_values("unet_dice").iloc[0]

    print(f"\nBiggest U-Net win (pixel Dice): {unet_wins['image_id']} "
          f"(Otsu Dice={unet_wins['otsu_dice']:.3f}, U-Net Dice={unet_wins['unet_dice']:.3f})")
    print(f"Best (or least-bad) Otsu case: {otsu_wins['image_id']} "
          f"(Otsu Dice={otsu_wins['otsu_dice']:.3f}, U-Net Dice={otsu_wins['unet_dice']:.3f})")

    # Visualize both example images: grayscale | Otsu mask | U-Net mask
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for row_idx, stem in enumerate([otsu_wins["image_id"], unet_wins["image_id"]]):
        gray, otsu_mask, unet_mask = masks_for_plot[stem]
        axes[row_idx, 0].imshow(gray, cmap="gray")
        axes[row_idx, 0].set_title(f"{stem} (input)")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(otsu_mask, cmap="gray")
        r = comparison.set_index("image_id").loc[stem]
        axes[row_idx, 1].set_title(f"Otsu (Dice={r['otsu_dice']:.3f})")
        axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(unet_mask, cmap="gray")
        axes[row_idx, 2].set_title(f"U-Net (Dice={r['unet_dice']:.3f})")
        axes[row_idx, 2].axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "task3_unet_vs_otsu_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved comparison table to {METRICS_DIR / 'task3_unet_vs_otsu_vs_groundtruth.csv'}")
    print(f"Saved example figure to {FIGURE_DIR / 'task3_unet_vs_otsu_examples.png'}")


if __name__ == "__main__":
    main()
