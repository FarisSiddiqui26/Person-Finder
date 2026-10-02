"""Clothing-colour matcher — pixel-ratio matching over HSV ranges (pattern-aware)."""

import cv2
import numpy as np

from .config import (
    COLORS, VALID_COLORS, COLOR_FAMILIES,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    FAMILY_MISMATCH_CAP,
)

# Neutrals are defined by saturation/value, not hue — skip the hue fallback for them.
_NEUTRAL_FAMILIES = {"white", "black", "gray", "beige"}
# Below this pixel ratio the "dominant" colour is just noise → report "unknown".
MIN_DOMINANT_RATIO = 0.08

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)


def _expand_to_shades(color_name):
    # Family name ("blue") → its member shades; a shade name → itself; else [].
    if color_name in COLOR_FAMILIES:
        return COLOR_FAMILIES[color_name]
    if color_name in COLORS:
        return [color_name]
    return []


def _family_of(color):
    if color in COLOR_FAMILIES:
        return color
    for fam, shades in COLOR_FAMILIES.items():
        if color in shades:
            return fam
    return None


def _same_family(color_a, color_b):
    if not color_a or not color_b:
        return False
    if color_a == color_b:
        return True
    fam_a = _family_of(color_a)
    fam_b = _family_of(color_b)
    return fam_a is not None and fam_a == fam_b


def _is_neutral(color):
    return _family_of(color) in _NEUTRAL_FAMILIES


def normalize_lighting(bgr_image):
    # CLAHE on the L channel evens out lighting without shifting colour.
    if bgr_image is None or bgr_image.size == 0:
        return bgr_image

    h, w = bgr_image.shape[:2]
    tile_w, tile_h = CLAHE_TILE_SIZE
    if h < tile_h or w < tile_w:          # too small for CLAHE's tile grid
        return bgr_image

    try:
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_normalized = _clahe.apply(l_channel)
        lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    except cv2.error:
        return bgr_image


def extract_body_regions(person_crop):
    # Shirt = 18-50% height, pants = 50-82%, sampled from the CENTRE 25-75% of
    # the box only (the edges are background). No CLAHE here — it would wash dark
    # garments (a brown kurta) into "gray".
    if person_crop is None:
        return None, None

    h, w = person_crop.shape[:2]
    if h < 40 or w < 20:
        return None, None

    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    upper_bgr = person_crop[int(h * 0.18): int(h * 0.50), cx0:cx1]
    lower_bgr = person_crop[int(h * 0.50): int(h * 0.82), cx0:cx1]
    if upper_bgr.size == 0 or lower_bgr.size == 0:
        return None, None

    upper_bgr = cv2.GaussianBlur(upper_bgr, (5, 5), 0)
    lower_bgr = cv2.GaussianBlur(lower_bgr, (5, 5), 0)
    upper_hsv = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2HSV)
    return upper_hsv, lower_hsv


def count_color_pixels(hsv_region, color_name):
    # OR every HSV sub-range of the colour/family, then open to drop noise pixels.
    shades = _expand_to_shades(color_name)
    if not shades:
        return 0

    total_mask = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
    for shade in shades:
        for lower, upper in COLORS.get(shade, []):
            mask = cv2.inRange(hsv_region,
                               np.array(lower, dtype=np.uint8),
                               np.array(upper, dtype=np.uint8))
            total_mask = cv2.bitwise_or(total_mask, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel)
    return int(np.count_nonzero(total_mask))


def detect_dominant_color(hsv_region):
    if hsv_region is None or hsv_region.size == 0:
        return "unknown", 0.0
    total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
    if total_pixels == 0:
        return "unknown", 0.0

    best_color, best_ratio = "unknown", 0.0
    for color_name in VALID_COLORS:
        ratio = count_color_pixels(hsv_region, color_name) / total_pixels
        if ratio > best_ratio:
            best_ratio, best_color = ratio, color_name

    if best_ratio < MIN_DOMINANT_RATIO:
        return "unknown", best_ratio
    return best_color, best_ratio


def match_clothing_color(hsv_region, target_color):
    # Score by what fraction of pixels fall in the target colour's HSV range,
    # then cap the score if the region's dominant colour is a different family.
    if hsv_region is None or hsv_region.size == 0:
        return 0.0, "unknown"
    if target_color not in COLORS and target_color not in COLOR_FAMILIES:
        return 0.0, "unknown"
    total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
    if total_pixels == 0:
        return 0.0, "unknown"

    target_ratio = count_color_pixels(hsv_region, target_color) / total_pixels
    dominant_color, _ = detect_dominant_color(hsv_region)

    if target_ratio > 0.40:
        score = 0.95
    elif target_ratio > 0.25:
        score = 0.85
    elif target_ratio > 0.15:
        score = 0.70
    elif target_ratio > 0.08:
        score = 0.50
    elif _is_neutral(target_color):
        score = 0.0                       # neutrals have no meaningful hue fallback
    else:
        median_hsv = np.median(hsv_region.reshape(-1, 3), axis=0)
        score = _hsv_similarity(median_hsv, _compute_color_center(target_color)) * 0.45

    if not _same_family(dominant_color, target_color):
        score = min(score, FAMILY_MISMATCH_CAP)
    return score, dominant_color


def _compute_color_center(color_name):
    shades = _expand_to_shades(color_name)
    if not shades:
        return np.array([0, 0, 0], dtype=np.float32)
    all_centers = []
    for shade in shades:
        for lower, upper in COLORS.get(shade, []):
            all_centers.append([(lower[i] + upper[i]) / 2.0 for i in range(3)])
    if not all_centers:
        return np.array([0, 0, 0], dtype=np.float32)
    return np.mean(all_centers, axis=0).astype(np.float32)


def _hsv_similarity(hsv1, hsv2):
    # Weighted hue/sat/value distance → similarity (hue is circular).
    if hsv1 is None or hsv2 is None:
        return 0.0
    h1, s1, v1 = float(hsv1[0]), float(hsv1[1]), float(hsv1[2])
    h2, s2, v2 = float(hsv2[0]), float(hsv2[1]), float(hsv2[2])
    h_diff = min(abs(h1 - h2), 180.0 - abs(h1 - h2)) / 90.0
    s_diff = abs(s1 - s2) / 255.0
    v_diff = abs(v1 - v2) / 255.0
    return max(0.0, 1.0 - (0.5 * h_diff + 0.25 * s_diff + 0.25 * v_diff))
