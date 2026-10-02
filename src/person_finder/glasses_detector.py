"""Glasses detector — eye-band vs cheek-band edge ratio (>1 means glasses)."""

import cv2
import numpy as np

from .config import GLASSES_RATIO_PIVOT, GLASSES_RATIO_SCALE


def _band_edges(gray, y0, y1, x0, x1):
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None, None
    h_strength = float(np.mean(np.abs(cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3))))
    edges = cv2.Canny(band, 40, 120)
    density = float(np.count_nonzero(edges)) / edges.size if edges.size else 0.0
    return h_strength, density


def detect_glasses(face_crop):
    # Glasses add frame/bridge edges to the eye band; the ratio to the smooth
    # cheek band cancels out crop resolution. Returns 0..1 (0.5 = can't tell).
    if face_crop is None:
        return 0.5

    h, w = face_crop.shape[:2]
    if h < 24 or w < 24:
        return 0.5

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    x0, x1 = int(w * 0.10), int(w * 0.90)
    eye_h, eye_d = _band_edges(gray, int(h * 0.28), int(h * 0.50), x0, x1)
    cheek_h, cheek_d = _band_edges(gray, int(h * 0.55), int(h * 0.78), x0, x1)
    if eye_h is None or cheek_h is None:
        return 0.5

    eps = 1e-6
    avg_ratio = 0.5 * (eye_h / (cheek_h + eps)) + 0.5 * (eye_d / (cheek_d + eps))
    confidence = 0.5 + (avg_ratio - GLASSES_RATIO_PIVOT) * GLASSES_RATIO_SCALE
    return float(np.clip(confidence, 0.0, 1.0))


def score_glasses_match(glasses_confidence, target_wearing_glasses):
    # High confidence is good when the target wears glasses, and the reverse if not.
    if target_wearing_glasses:
        return float(glasses_confidence)
    return float(1.0 - glasses_confidence)
