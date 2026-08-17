"""Task 1: data preparation and EDA.

Loads the nuclei dataset's RGB images, converts them to grayscale, resizes
to a common target size (a no-op here since the dataset is already
256x256, but implemented generically so the pipeline would still work on a
dataset with mixed original sizes), and produces summary EDA (a sample
grid and an intensity histogram).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, exposure

from imaging_pipeline.config import IMAGE_SIZE


def list_image_paths(split_dir: Path) -> list:
    """All image file paths in a split's images/ subfolder, sorted."""
    return sorted((split_dir / "images").glob("*.png"))


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float RGB in [0, 1]."""
    img = io.imread(path)
    if img.ndim == 2:
        img = color.gray2rgb(img)
    if img.shape[-1] == 4:  # drop alpha if present
        img = img[..., :3]
    return exposure.rescale_intensity(img.astype(np.float32), in_range="image", out_range=(0, 1)) \
        if img.max() > 1.0 else img.astype(np.float32)


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB float image in [0, 1] to grayscale, float in [0, 1].

    Uses skimage's standard luminosity weighting (0.2125 R + 0.7154 G +
    0.0721 B), which is appropriate even for a blue-dominant fluorescence
    channel -- it just means the blue channel contributes least to the
    weighted sum while still being the channel carrying almost all the
    signal here, so the result tracks the original blue intensity closely.
    """
    return color.rgb2gray(img_rgb)


def resize_to_common_size(img: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Resize (grayscale or RGB) image to `size`. No-op in practice for
    this dataset (already 256x256) but implemented generically."""
    if img.shape[:2] == size:
        return img
    return transform.resize(img, size, anti_aliasing=True, preserve_range=True).astype(img.dtype)


def preprocess_split(split_dir: Path) -> list:
    """Load every image in a split, convert to grayscale, resize.

    Returns a list of dicts: {"stem": str, "rgb": ndarray, "gray": ndarray}.
    """
    records = []
    for path in list_image_paths(split_dir):
        rgb = load_rgb(path)
        rgb_resized = resize_to_common_size(rgb, IMAGE_SIZE)
        gray = to_grayscale(rgb_resized)
        records.append({"stem": path.stem, "path": path, "rgb": rgb_resized, "gray": gray})
    return records


def plot_sample_grid(records: list, n: int = 8, save_path: Path = None, seed: int = 0):
    """Grid of n sample images (original RGB + grayscale pairs)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    idx.sort()

    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.6))
    for col, i in enumerate(idx):
        rec = records[i]
        axes[0, col].imshow(rec["rgb"])
        axes[0, col].set_title(rec["stem"], fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(rec["gray"], cmap="gray")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("RGB", fontsize=9)
    axes[1, 0].set_ylabel("Grayscale", fontsize=9)
    fig.suptitle("Sample images: original (top) vs grayscale (bottom)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_intensity_histogram(records: list, save_path: Path = None):
    """Histogram of grayscale pixel intensities, pooled across all images
    in `records`, plus per-image mean intensity distribution."""
    all_pixels = np.concatenate([rec["gray"].ravel() for rec in records])
    per_image_means = np.array([rec["gray"].mean() for rec in records])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(all_pixels, bins=60, color="steelblue", edgecolor="none")
    axes[0].set_title(f"Pooled pixel intensity histogram (n={len(records)} images)")
    axes[0].set_xlabel("Grayscale intensity [0, 1]")
    axes[0].set_ylabel("Pixel count")
    axes[0].set_yscale("log")

    axes[1].hist(per_image_means, bins=20, color="indianred", edgecolor="none")
    axes[1].set_title("Per-image mean intensity distribution")
    axes[1].set_xlabel("Mean grayscale intensity")
    axes[1].set_ylabel("Number of images")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
