"""Task 2 (part 1): classical image features via scikit-image.

Otsu thresholding + morphological cleanup -> connected-component labeling
-> regionprops feature table -> deterministic natural-language summary.

This is the classical baseline segmentation, distinct from the U-Net
(Task 3) -- used both as an LLM-interpretation input here and later as the
comparison point for "did the U-Net improve on classical Otsu segmentation
for your modality?" (Part 9, Q2).
"""

import numpy as np
import pandas as pd
from skimage import filters, morphology, measure
from skimage.measure import regionprops_table

REGIONPROPS = ("label", "area", "perimeter", "eccentricity", "solidity", "mean_intensity", "extent")


def otsu_segment(gray_image: np.ndarray, min_object_size: int = 20) -> np.ndarray:
    """Otsu threshold + morphological cleanup.

    Steps: Otsu threshold -> binary mask -> remove small specks
    (min_object_size, matches the U-Net-mask cleanup threshold used later
    in Task 4's pipeline, for a fair comparison) -> binary opening to
    smooth ragged edges from thresholding noise.
    """
    threshold = filters.threshold_otsu(gray_image)
    binary = gray_image > threshold
    # scikit-image renamed remove_small_objects()'s size argument from
    # min_size (pre-0.26) to max_size (0.26+) -- same practical effect
    # (removes small specks around the given pixel-count threshold), but
    # the exact keyword accepted depends on the installed version. Try
    # both so this works either way rather than pinning to one.
    try:
        binary = morphology.remove_small_objects(binary, max_size=min_object_size)
    except TypeError:
        binary = morphology.remove_small_objects(binary, min_size=min_object_size)
    try:
        binary = morphology.opening(binary, morphology.disk(1))
    except AttributeError:
        binary = morphology.binary_opening(binary, morphology.disk(1))
    return binary


def extract_region_features(gray_image: np.ndarray, binary_mask: np.ndarray) -> pd.DataFrame:
    """Label connected components in `binary_mask` and compute a
    per-object regionprops feature table, using `gray_image` for
    intensity-based properties.
    """
    labels = measure.label(binary_mask)
    if labels.max() == 0:
        return pd.DataFrame(columns=REGIONPROPS)
    props = regionprops_table(labels, intensity_image=gray_image, properties=REGIONPROPS)
    return pd.DataFrame(props)


def summarise_features(df: pd.DataFrame, image_name: str = "image", method: str = "Otsu segmentation") -> str:
    """Deterministic natural-language summary of a feature table -- the
    same table always yields the same text. This is what gets passed to
    the LLM (numbers only; the model never sees the image).

    `method` names which segmentation produced the mask this table came
    from (e.g. "Otsu segmentation" in Task 2, "U-Net segmentation" in
    Task 4) -- the underlying summary logic is identical either way, only
    the wording needs to stay accurate about the source.
    """
    n = len(df)
    if n == 0:
        return f"In {image_name}, no objects were detected by {method}."

    area, ecc, sol, intensity, extent = df["area"], df["eccentricity"], df["solidity"], df["mean_intensity"], df["extent"]
    return (
        f"In {image_name}, {method} detected {n} objects. "
        f"Object areas range from {area.min():.0f} to {area.max():.0f} pixels "
        f"(mean {area.mean():.0f}, median {area.median():.0f}). "
        f"Mean eccentricity is {ecc.mean():.2f} (0=circular, 1=elongated); "
        f"mean solidity is {sol.mean():.2f} (1=convex/solid boundary); "
        f"mean extent is {extent.mean():.2f} (area / bounding-box area). "
        f"Mean object intensity is {intensity.mean():.3f}."
    )
