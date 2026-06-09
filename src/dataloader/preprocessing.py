"""
Preprocessing pipeline (V14b)

Processing steps in order:
  1. Load RGB image and greyscale mask
  2. Morphological denoising (optional, off by default)
  3. Gamma correction (default gamma=1.06)
  4. CLAHE on the L channel of Lab colour space (on by default)
  5. Skin-only Shades-of-Gray colour normalisation
     — illuminant estimated only from healthy skin pixels (excluding lesion and
       black border), preventing lesion colour from biasing the estimate and
       improving cross-device colour consistency
  6. Normalise to [0,1] float32

Config key path: config[config["model"]]["preprocessing"]
"""

import numpy as np
from PIL import Image
from skimage import color as sk_color, exposure, morphology


def load_image_and_mask(image_path, mask_path, config):
    """Load image+mask, apply preprocessing, return (image_float32, binary_mask).

    image_float32: H×W×3 float32 in [0,1]
    binary_mask:   H×W bool, True = lesion region
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch: {image_path} {image.shape[:2]} "
            f"vs {mask_path} {mask.shape[:2]}"
        )

    model_key = config["model"]
    fc = config[model_key]["features"]
    pc = config[model_key]["preprocessing"]

    binary_mask = mask > fc.get("mask_threshold", 127)
    if pc.get("morphology", False):
        fp = morphology.square(3)
        binary_mask = morphology.binary_opening(binary_mask, fp)
        binary_mask = morphology.binary_closing(binary_mask, fp)
    if not binary_mask.any():
        raise ValueError(f"Empty mask after thresholding: {mask_path}")

    # Step 1: Gamma correction
    if pc.get("gamma_correction", True):
        image = exposure.adjust_gamma(
            image, gamma=float(pc.get("gamma_value", 1.06))
        )

    # Step 2: CLAHE on Lab L-channel
    if pc.get("clahe", False):
        lab = sk_color.rgb2lab(image)
        L = lab[:, :, 0] / 100.0
        kernel = pc.get("clahe_kernel", 7)
        L_eq = exposure.equalize_adapthist(
            L,
            kernel_size=(kernel, kernel),
            clip_limit=pc.get("clahe_clip", 0.03),
        )
        lab[:, :, 0] = L_eq * 100.0
        image = np.clip(sk_color.lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)

    # Step 3: Skin-only Shades-of-Gray color normalization
    if pc.get("color_normalize", False):
        image_f = image.astype(np.float32)
        # Healthy skin = NOT lesion AND NOT black border
        is_black_border = (
            (image[:, :, 0] < 15)
            & (image[:, :, 1] < 15)
            & (image[:, :, 2] < 15)
        )
        skin_valid = ~binary_mask & ~is_black_border
        if skin_valid.sum() < 100:
            skin_valid = ~binary_mask  # fallback

        p = float(pc.get("shades_of_gray_p", 6))
        illum = []
        for c in range(3):
            vals = image_f[:, :, c][skin_valid].astype(np.float64)
            if len(vals) > 0:
                L_c = np.power(np.mean(np.power(vals, p)), 1.0 / p)
            else:
                L_c = np.power(
                    np.mean(np.power(image_f[:, :, c].astype(np.float64), p)),
                    1.0 / p,
                )
            illum.append(L_c)
        im = np.mean(illum)
        for c in range(3):
            if illum[c] > 0:
                image_f[:, :, c] = np.clip(
                    image_f[:, :, c] / illum[c] * im, 0, 255
                )
        image = image_f.astype(np.uint8)

    # Step 4: Normalize to [0,1]
    if pc.get("normalize", True):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    return image, binary_mask


def crop_to_mask(image, mask, padding=4):
    """Crop image/mask to bounding box of the lesion region."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return image, mask
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, mask.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, mask.shape[1])
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
