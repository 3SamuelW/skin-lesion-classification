"""
features.py — Skin lesion feature extraction (V14b, full set)

Features are organised into groups, all controlled by config flags:

  use_color             Basic colour statistics (RGB / HSV / Lab per-channel mean/std/skew + HSV hist)
  use_advanced_color    Advanced colour (Blue-White Veil, variegation, entropy, centre-periphery diff, etc.)
  use_texture           LBP histogram + multi-distance GLCM (4-level quantisation)
  use_advanced_texture  Local entropy, multi-scale LBP, border GLCM
  use_shape             Basic shape (area ratio, perimeter, circularity, eccentricity, solidity, etc.)
  use_advanced_shape    Fractal dimension, colour asymmetry, radial distance CV, border jaggedness
  use_clinical          ABCD colour count, quadrant distribution, dark blob analysis, border gradient
  use_melanin_features  Melanin index, haemoglobin index, centre-periphery melanin ratio
  use_hog               64x64 HOG (off by default)
"""

import os
import tempfile
from pathlib import Path

# Suppress matplotlib config warnings in shared environments
_cache = Path(tempfile.gettempdir()) / "skin_cls_mpl"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation, uniform_filter
from scipy.stats import skew
from skimage import color, measure
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
from skimage.transform import resize
from tqdm import tqdm

from src.dataloader.preprocessing import crop_to_mask, load_image_and_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(image_path, mask_path, config):
    """Extract all enabled features for one image. Returns a flat dict."""
    image, mask = load_image_and_mask(image_path, mask_path, config)
    fc = config[config["model"]]["features"]
    features = {}

    if fc.get("use_color", True):
        features.update(_color_features(image, mask, fc))
        if fc.get("use_advanced_color", True):
            features.update(_advanced_color_features(image, mask))

    if fc.get("use_texture", True):
        features.update(_texture_features(image, mask, fc))
        if fc.get("use_advanced_texture", True):
            features.update(_advanced_texture_features(image, mask, fc))

    if fc.get("use_shape", True):
        features.update(_shape_features(mask))
        if fc.get("use_advanced_shape", True):
            features.update(_advanced_shape_features(image, mask))

    if fc.get("use_clinical", False):
        features.update(_clinical_features(image, mask))

    if fc.get("use_melanin_features", False):
        features.update(_melanin_features(image, mask))

    if fc.get("use_hog", False):
        features.update(_hog_features(image, mask))

    return features


def extract_feature_table(metadata_df, config):
    """Extract features for every row in metadata_df. Returns a DataFrame.

    Metadata columns (image_id, label, base_id, …) are preserved alongside
    the numeric feature columns so the result can be directly merged with
    split assignments.
    """
    rows = []
    for row in tqdm(metadata_df.itertuples(index=False),
                    total=len(metadata_df), desc="Extracting features"):
        entry = {
            "image_id": row.image_id,
            "label": row.label,
            "image_path": row.image_path,
            "mask_path": row.mask_path,
            "base_id": row.base_id,
            "is_augmented": row.is_augmented,
            "augmentation_id": row.augmentation_id,
        }
        entry.update(extract_features(row.image_path, row.mask_path, config))
        rows.append(entry)

    df = pd.DataFrame(rows)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Color features
# ---------------------------------------------------------------------------

def _channel_moments(values, prefix):
    result = {}
    for idx in range(values.shape[1]):
        ch = values[:, idx]
        result[f"{prefix}_{idx}_mean"] = float(np.mean(ch))
        result[f"{prefix}_{idx}_std"] = float(np.std(ch))
        result[f"{prefix}_{idx}_skew"] = (
            float(skew(ch, bias=False)) if len(ch) > 2 else 0.0
        )
    return result


def _color_features(image, mask, fc):
    lesion_rgb = image[mask]
    hsv = color.rgb2hsv(image)
    lesion_hsv = hsv[mask]
    lab = color.rgb2lab(image)
    lesion_lab = lab[mask]

    result = {}
    result.update(_channel_moments(lesion_rgb, "rgb"))
    result.update(_channel_moments(lesion_hsv, "hsv"))
    result.update(_channel_moments(lesion_lab, "lab"))

    bins = int(fc.get("hsv_hist_bins", 16))
    for ci, cn in enumerate(["h", "s", "v"]):
        hist, _ = np.histogram(
            lesion_hsv[:, ci], bins=bins, range=(0.0, 1.0), density=False
        )
        hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
        for bi, v in enumerate(hist):
            result[f"hsv_{cn}_hist_{bi}"] = float(v)

    result["dark_pixel_ratio"] = float(np.mean(lesion_hsv[:, 2] < 0.35))
    result["hue_std"] = float(np.std(lesion_hsv[:, 0]))
    result["saturation_mean"] = float(np.mean(lesion_hsv[:, 1]))
    return result


def _advanced_color_features(image, mask):
    hsv = color.rgb2hsv(image)
    lesion_hsv = hsv[mask]
    lab = color.rgb2lab(image)
    lesion_lab = lab[mask]
    lesion_rgb = image[mask]
    result = {}

    bw = (
        (lesion_lab[:, 0] > 70)
        & (lesion_lab[:, 1] > -15)
        & (lesion_lab[:, 1] < 15)
        & (lesion_lab[:, 2] > -20)
        & (lesion_lab[:, 2] < 20)
    )
    result["blue_white_veil_ratio"] = float(bw.mean())

    for idx, name in enumerate(["r", "g", "b"]):
        result[f"color_variegation_{name}"] = _local_std(image[:, :, idx], mask)

    for idx, name in enumerate(["h", "s", "v"]):
        hist, _ = np.histogram(
            lesion_hsv[:, idx], bins=16, range=(0, 1), density=True
        )
        result[f"color_entropy_{name}"] = float(
            -np.sum((hist + 1e-10) * np.log2(hist + 1e-10))
        )

    result["dark_spot_ratio"] = float(np.mean(lesion_hsv[:, 2] < 0.2))

    cm = _center_region(mask, 0.5)
    pm = mask & ~cm
    for idx, name in enumerate(["r", "g", "b"]):
        if pm.sum() > 50 and cm.sum() > 50:
            result[f"center_periphery_diff_{name}"] = float(
                abs(
                    image[:, :, idx][cm].mean()
                    - image[:, :, idx][pm].mean()
                )
            )
        else:
            result[f"center_periphery_diff_{name}"] = 0.0

    for idx, name in enumerate(["r", "g", "b"]):
        result[f"rgb_{name}_p90"] = float(np.percentile(lesion_rgb[:, idx], 90))
        result[f"rgb_{name}_p10"] = float(np.percentile(lesion_rgb[:, idx], 10))
        result[f"rgb_{name}_range"] = float(
            np.percentile(lesion_rgb[:, idx], 95)
            - np.percentile(lesion_rgb[:, idx], 5)
        )

    r, g, b = lesion_rgb[:, 0], lesion_rgb[:, 1], lesion_rgb[:, 2]
    result["rgb_rg_ratio_mean"] = float(np.mean(r / (g + 1e-8)))
    result["rgb_rb_ratio_mean"] = float(np.mean(r / (b + 1e-8)))
    result["rgb_gb_ratio_mean"] = float(np.mean(g / (b + 1e-8)))

    gl = color.rgb2gray(image)[mask]
    result["gray_skew"] = float(skew(gl, bias=False)) if len(gl) > 2 else 0.0
    result["gray_kurtosis"] = float(
        np.mean((gl - gl.mean()) ** 4) / (gl.std() ** 4 + 1e-8)
    )
    return result


# ---------------------------------------------------------------------------
# Texture features  (GLCM with configurable quantisation levels)
# ---------------------------------------------------------------------------

def _texture_features(image, mask, fc):
    """LBP histogram + multi-distance GLCM with configurable grey levels."""
    gray = color.rgb2gray(image)
    crop_gray, crop_mask = crop_to_mask(gray, mask)

    levels = int(fc.get("glcm_levels", 4))
    gray_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)
    divisor = max(256 // levels, 1)
    quantized = (gray_uint8 // divisor).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)

    lesion_values = gray_uint8[crop_mask]
    result = {
        "gray_mean": float(np.mean(lesion_values)),
        "gray_std": float(np.std(lesion_values)),
    }

    # LBP histogram
    pts = int(fc.get("lbp_points", 24))
    rad = int(fc.get("lbp_radius", 3))
    lbp = local_binary_pattern(gray_uint8, pts, rad, method="uniform")
    lbp_bins = pts + 2
    hist, _ = np.histogram(
        lbp[crop_mask], bins=lbp_bins, range=(0, lbp_bins), density=False
    )
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    for idx, v in enumerate(hist):
        result[f"lbp_hist_{idx}"] = float(v)

    # GLCM
    quantized = np.where(crop_mask, quantized, 0).astype(np.uint8)
    distances = fc.get("glcm_distances", [1, 2, 4])
    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "ASM"]:
        values = graycoprops(glcm, prop)
        result[f"glcm_{prop}_mean"] = float(np.mean(values))
        result[f"glcm_{prop}_std"] = float(np.std(values))
    return result


def _advanced_texture_features(image, mask, fc):
    gray = color.rgb2gray(image)
    gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    result = {}

    # Local entropy
    try:
        ent = local_entropy(gray_uint8, disk(5))[mask]
        result["local_entropy_mean"] = float(ent.mean())
        result["local_entropy_std"] = float(ent.std())
        result["local_entropy_p90"] = float(np.percentile(ent, 90))
    except Exception:
        result["local_entropy_mean"] = 0.0
        result["local_entropy_std"] = 0.0
        result["local_entropy_p90"] = 0.0

    # Multi-scale LBP
    crop_gray, crop_mask = crop_to_mask(gray, mask)
    crop_uint8 = np.clip(crop_gray * 255.0, 0, 255).astype(np.uint8)
    for r in fc.get("lbp_multi_radii", [1, 5, 7]):
        pts = int(max(8, round(24 * r / 3)))
        try:
            lbp = local_binary_pattern(crop_uint8, pts, r, method="uniform")
            vals = lbp[crop_mask]
            if len(vals) == 0:
                continue
            bins = pts + 2
            hist, _ = np.histogram(vals, bins=bins, range=(0, bins), density=False)
            hist = hist.astype(np.float32)
            s = hist.sum()
            if s > 0:
                hist /= s
            for idx, v in enumerate(hist):
                result[f"lbp_r{r}_{idx}"] = float(v)
        except Exception:
            pass

    # Border GLCM (8-level quantisation on erosion-dilation border ring)
    try:
        eroded = binary_erosion(mask, iterations=3)
        dilated = binary_dilation(mask, iterations=3)
        border = dilated & ~eroded
        if border.sum() >= 50:
            q = (gray_uint8 // 32).astype(np.uint8)
            q_border = np.where(border, q, 0).astype(np.uint8)
            glcm = graycomatrix(
                q_border,
                distances=[1, 2],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=8,
                symmetric=True,
                normed=True,
            )
            for prop in ["contrast", "dissimilarity", "homogeneity",
                         "energy", "correlation", "ASM"]:
                vals = graycoprops(glcm, prop)
                result[f"border_glcm_{prop}_mean"] = float(np.mean(vals))
                result[f"border_glcm_{prop}_std"] = float(np.std(vals))
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Shape features
# ---------------------------------------------------------------------------

def _shape_features(mask):
    label_img = measure.label(mask.astype(np.uint8))
    regions = measure.regionprops(label_img)
    if not regions:
        return {k: 0.0 for k in [
            "area_ratio", "perimeter", "circularity", "eccentricity",
            "major_axis_length", "minor_axis_length", "solidity", "extent",
            "bbox_aspect_ratio", "border_irregularity", "compactness",
            "convex_area_ratio", "horizontal_asymmetry", "vertical_asymmetry",
        ]}

    region = max(regions, key=lambda r: r.area)
    area = float(region.area)
    perim = float(measure.perimeter(mask))
    circ = 4.0 * np.pi * area / (perim ** 2 + 1e-8)
    minr, minc, maxr, maxc = region.bbox
    h = max(maxr - minr, 1)
    w = max(maxc - minc, 1)
    convex_regions = measure.regionprops(
        measure.label(region.image_convex.astype(np.uint8))
    )
    conv_perim = float(measure.perimeter(region.image_convex))

    return {
        "area_ratio": area / float(mask.size),
        "perimeter": perim,
        "circularity": float(circ),
        "eccentricity": float(region.eccentricity),
        "major_axis_length": float(region.axis_major_length),
        "minor_axis_length": float(region.axis_minor_length),
        "solidity": float(region.solidity),
        "extent": float(region.extent),
        "bbox_aspect_ratio": float(w / h),
        "border_irregularity": float(perim / (conv_perim + 1e-8)),
        "compactness": float(perim / (np.sqrt(area) + 1e-8)),
        "convex_area_ratio": float(
            area / (convex_regions[0].area + 1e-8) if convex_regions else 1.0
        ),
        "horizontal_asymmetry": _asymmetry(mask, 1),
        "vertical_asymmetry": _asymmetry(mask, 0),
    }


def _advanced_shape_features(image, mask):
    result = {"fractal_dimension": _fractal_dimension(mask)}
    gray = color.rgb2gray(image)
    result["color_asymmetry_h"] = _color_asymmetry(gray, mask, 1)
    result["color_asymmetry_v"] = _color_asymmetry(gray, mask, 0)
    result["radial_distance_cv"] = _radial_distance_cv(mask)
    result["border_jaggedness"] = _border_jaggedness(mask)

    regions = measure.regionprops(measure.label(mask.astype(np.uint8)))
    if regions:
        region = max(regions, key=lambda r: r.area)
        result["major_minor_ratio"] = float(
            region.axis_major_length / (region.axis_minor_length + 1e-8)
        )
    else:
        result["major_minor_ratio"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Clinical features
# ---------------------------------------------------------------------------

def _clinical_features(image, mask):
    from scipy.ndimage import label as nd_label

    result = {}
    lab = color.rgb2lab(image)
    lesion_lab = lab[mask]
    gray = color.rgb2gray(image)

    # --- Colour count (ABCD criteria) ---
    colors_present = 0
    if (lesion_lab[:, 0] < 20).mean() > 0.05:
        colors_present += 1  # black
    dk_brown = (
        (lesion_lab[:, 0] >= 20) & (lesion_lab[:, 0] < 50) & (lesion_lab[:, 1] > 2)
    )
    if dk_brown.mean() > 0.05:
        colors_present += 1
    lt_brown = (lesion_lab[:, 0] >= 50) & (lesion_lab[:, 0] < 70)
    if lt_brown.mean() > 0.05:
        colors_present += 1
    blue_gray = (
        (lesion_lab[:, 0] > 30) & (lesion_lab[:, 0] < 70)
        & (lesion_lab[:, 1] < -3) & (lesion_lab[:, 2] < -3)
    )
    if blue_gray.mean() > 0.05:
        colors_present += 1
    red = lesion_lab[:, 1] > 15
    if red.mean() > 0.05:
        colors_present += 1
    white = (
        (lesion_lab[:, 0] > 80)
        & (np.abs(lesion_lab[:, 1]) < 10)
        & (np.abs(lesion_lab[:, 2]) < 10)
    )
    if white.mean() > 0.05:
        colors_present += 1
    result["clinical_color_count"] = float(colors_present)

    class_sizes = [
        ("black", (lesion_lab[:, 0] < 20).mean()),
        ("dk_brown", dk_brown.mean()),
        ("lt_brown", lt_brown.mean()),
        ("blue_gray", blue_gray.mean()),
        ("red", red.mean()),
        ("white", white.mean()),
    ]
    dominant = max(class_sizes, key=lambda x: x[1])
    color_names = ["black", "dk_brown", "lt_brown", "blue_gray", "red", "white"]
    result["clinical_dominant_color"] = float(color_names.index(dominant[0]))
    result["clinical_dominant_color_ratio"] = float(dominant[1])

    # --- Quadrant colour distribution ---
    ys, xs = np.where(mask)
    cy = float(np.median(ys)) if len(ys) > 0 else 0.0
    cx = float(np.median(xs)) if len(xs) > 0 else 0.0
    if len(ys) > 50:
        def _qmean(ch, y0, y1, x0, x1):
            sm = np.zeros_like(mask, dtype=bool)
            sm[max(0, y0):min(mask.shape[0], y1),
               max(0, x0):min(mask.shape[1], x1)] = True
            sm &= mask
            return float(ch[sm].mean()) if sm.sum() >= 10 else 0.0

        for ci, cn in enumerate(["l", "a", "b"]):
            ch = lab[:, :, ci]
            means = np.array([
                _qmean(ch, 0, int(cy), 0, int(cx)),
                _qmean(ch, 0, int(cy), int(cx), mask.shape[1]),
                _qmean(ch, int(cy), mask.shape[0], 0, int(cx)),
                _qmean(ch, int(cy), mask.shape[0], int(cx), mask.shape[1]),
            ])
            result[f"clinical_quadrant_{cn}_var"] = float(np.var(means))
            result[f"clinical_quadrant_{cn}_range"] = float(np.ptp(means))

    # --- Dark blob analysis ---
    try:
        gl = gray[mask]
        dark_thresh = np.percentile(gl, 20) if len(gl) > 20 else 0.3
        dark_mask = (gray < dark_thresh) & mask
        if dark_mask.sum() > 10:
            labeled, n_blobs = nd_label(
                dark_mask, structure=np.ones((3, 3), dtype=bool)
            )
            blob_sizes = np.bincount(labeled.ravel())[1:]
            result["clinical_dark_blob_count"] = float(n_blobs)
            result["clinical_dark_blob_mean_size"] = (
                float(blob_sizes.mean()) if len(blob_sizes) else 0.0
            )
            result["clinical_dark_blob_size_std"] = (
                float(blob_sizes.std()) if len(blob_sizes) > 1 else 0.0
            )
            if n_blobs > 1:
                bc = np.array([
                    (
                        np.where(labeled == i)[0].mean(),
                        np.where(labeled == i)[1].mean(),
                    )
                    for i in range(1, n_blobs + 1)
                ])
                result["clinical_dark_blob_spread"] = float(
                    np.sqrt((bc[:, 0] - cy) ** 2 + (bc[:, 1] - cx) ** 2).std()
                )
            else:
                result["clinical_dark_blob_spread"] = 0.0
        else:
            for k in ["clinical_dark_blob_count", "clinical_dark_blob_mean_size",
                      "clinical_dark_blob_size_std", "clinical_dark_blob_spread"]:
                result[k] = 0.0
    except Exception:
        for k in ["clinical_dark_blob_count", "clinical_dark_blob_mean_size",
                  "clinical_dark_blob_size_std", "clinical_dark_blob_spread"]:
            result[k] = 0.0

    # --- Border gradient ---
    try:
        border = mask & ~binary_erosion(mask, iterations=2)
        if np.where(border)[0].size > 20:
            gy, gx = np.gradient(gray)
            grad_mag = np.sqrt(gy ** 2 + gx ** 2)
            bg = grad_mag[border]
            result["clinical_border_grad_mean"] = float(bg.mean())
            result["clinical_border_grad_std"] = float(bg.std())
            result["clinical_border_grad_p90"] = float(np.percentile(bg, 90))
            result["clinical_border_grad_max"] = float(bg.max())
            outer = binary_dilation(mask, iterations=3) & ~mask
            if outer.sum() > 20:
                og = grad_mag[outer]
                result["clinical_outer_grad_mean"] = float(og.mean())
                result["clinical_outer_grad_std"] = float(og.std())
            else:
                result["clinical_outer_grad_mean"] = 0.0
                result["clinical_outer_grad_std"] = 0.0
        else:
            for k in ["clinical_border_grad_mean", "clinical_border_grad_std",
                      "clinical_border_grad_p90", "clinical_border_grad_max",
                      "clinical_outer_grad_mean", "clinical_outer_grad_std"]:
                result[k] = 0.0
    except Exception:
        for k in ["clinical_border_grad_mean", "clinical_border_grad_std",
                  "clinical_border_grad_p90", "clinical_border_grad_max",
                  "clinical_outer_grad_mean", "clinical_outer_grad_std"]:
            result[k] = 0.0
    return result


# ---------------------------------------------------------------------------
# Melanin / haemoglobin features
# ---------------------------------------------------------------------------

def _melanin_features(image, mask):
    result = {}
    lesion_rgb = image[mask]

    mel = -np.log(np.maximum(lesion_rgb[:, 1], 1e-8))
    result["melanin_index_mean"] = float(mel.mean())
    result["melanin_index_std"] = float(mel.std())
    result["melanin_index_skew"] = (
        float(skew(mel, bias=False)) if len(mel) > 2 else 0.0
    )
    result["melanin_index_p90"] = float(np.percentile(mel, 90))
    result["melanin_index_range"] = float(
        np.percentile(mel, 95) - np.percentile(mel, 5)
    )

    hemo = np.log(
        (lesion_rgb[:, 0] + 1e-8) / (lesion_rgb[:, 1] + 1e-8)
    )
    result["hemoglobin_index_mean"] = float(hemo.mean())
    result["hemoglobin_index_std"] = float(hemo.std())
    result["hemoglobin_index_range"] = float(
        np.percentile(hemo, 95) - np.percentile(hemo, 5)
    )

    try:
        center = _center_region(mask, 0.4)
        peripheral = mask & ~_center_region(mask, 0.6)
        if peripheral.sum() > 50 and center.sum() > 50:
            mm = -np.log(np.maximum(image[:, :, 1].astype(np.float64), 1e-8))
            result["peri_center_melanin_ratio"] = float(
                mm[peripheral].mean() / (mm[center].mean() + 1e-8)
            )
            gray = color.rgb2gray(image)
            dt = np.percentile(gray[mask], 30)
            result["peripheral_dark_ratio"] = float(
                (gray[peripheral] < dt).mean()
            )
            result["center_dark_ratio"] = float((gray[center] < dt).mean())
            result["peri_center_dark_ratio"] = float(
                result["peripheral_dark_ratio"]
                / (result["center_dark_ratio"] + 1e-8)
            )
        else:
            for k in ["peri_center_melanin_ratio", "peripheral_dark_ratio",
                      "center_dark_ratio", "peri_center_dark_ratio"]:
                result[k] = 0.0
    except Exception:
        for k in ["peri_center_melanin_ratio", "peripheral_dark_ratio",
                  "center_dark_ratio", "peri_center_dark_ratio"]:
            result[k] = 0.0

    try:
        gray = color.rgb2gray(image)
        result["melanin_asymmetry_h"] = _color_asymmetry(gray, mask, 1)
        result["melanin_asymmetry_v"] = _color_asymmetry(gray, mask, 0)
    except Exception:
        result["melanin_asymmetry_h"] = 0.0
        result["melanin_asymmetry_v"] = 0.0
    return result


# ---------------------------------------------------------------------------
# HOG features (optional)
# ---------------------------------------------------------------------------

def _hog_features(image, mask):
    ci, cm = crop_to_mask(image, mask)
    ci = resize(ci, (64, 64, 3), order=1, preserve_range=True, anti_aliasing=True)
    cm = (
        resize(cm.astype(float), (64, 64), order=0,
               preserve_range=True, anti_aliasing=False) > 0.5
    )
    gray = np.where(cm, color.rgb2gray(ci), 0.0)
    vals = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return {f"hog_{i}": float(v) for i, v in enumerate(vals)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _local_std(image, mask, window_size=9):
    mean = uniform_filter(image, size=window_size)
    mean_sq = uniform_filter(image ** 2, size=window_size)
    return float(np.sqrt(np.maximum(mean_sq - mean ** 2, 0))[mask].mean())


def _center_region(mask, shrink_ratio=0.5):
    regions = measure.regionprops(measure.label(mask.astype(np.uint8)))
    if not regions:
        return mask.astype(bool)
    region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = region.bbox
    cy = (minr + maxr) / 2.0
    cx = (minc + maxc) / 2.0
    hh = (maxr - minr) * shrink_ratio / 2.0
    hw = (maxc - minc) * shrink_ratio / 2.0
    center = np.zeros_like(mask, dtype=bool)
    center[
        max(0, int(round(cy - hh))): min(mask.shape[0], int(round(cy + hh))),
        max(0, int(round(cx - hw))): min(mask.shape[1], int(round(cx + hw))),
    ] = True
    return center & mask


def _asymmetry(mask, axis):
    flipped = np.flip(mask, axis=axis)
    union = np.logical_or(mask, flipped).sum()
    return float(np.logical_xor(mask, flipped).sum() / union) if union else 0.0


def _fractal_dimension(mask):
    boundary = mask & ~binary_erosion(mask)
    if boundary.sum() < 20:
        return 0.0
    points = np.column_stack(np.where(boundary))
    sizes = np.array([2, 4, 8, 16, 32, 64])
    counts = []
    for s in sizes:
        stride = max(1, len(points) // 3000)
        boxes = set((p[0] // s, p[1] // s) for p in points[::stride])
        if boxes:
            counts.append(len(boxes))
    if len(counts) < 3:
        return 0.0
    return float(-np.polyfit(np.log(sizes[: len(counts)]), np.log(counts), 1)[0])


def _color_asymmetry(ch, mask, axis):
    fm = np.flip(mask, axis=axis)
    fi = np.flip(ch, axis=axis)
    overlap = mask & fm
    if not overlap.sum():
        return 0.0
    return float(np.abs(ch * mask - fi * fm)[overlap].mean())


def _radial_distance_cv(mask):
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    bys, bxs = np.where(mask & ~binary_erosion(mask))
    if len(bys) < 10:
        return 0.0
    d = np.sqrt((bys - cy) ** 2 + (bxs - cx) ** 2)
    m = d.mean()
    return float(d.std() / m) if m > 0 else 0.0


def _border_jaggedness(mask):
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    bys, bxs = np.where(mask & ~binary_erosion(mask))
    if len(bys) < 20:
        return 0.0
    angles = np.arctan2(bys - cy, bxs - cx)
    dist = np.sqrt((bys - cy) ** 2 + (bxs - cx) ** 2)
    sorted_dist = dist[np.argsort(angles)]
    w = max(3, len(sorted_dist) // 20)
    smoothed = np.convolve(sorted_dist, np.ones(w) / w, mode="same")
    return float(np.sum(np.diff(np.sign(np.diff(smoothed))) != 0))
