"""Matching engine — cascaded scoring: cheap colour pre-filter, then face/glasses."""

from .color_engine import extract_body_regions, match_clothing_color
from .glasses_detector import detect_glasses, score_glasses_match
from .config import (
    COLOR_MATCH_THRESHOLD,
    FACE_SIMILARITY_THRESHOLD,
    NO_FACE_SCORE_FACTOR,
    GLASSES_MISMATCH_PENALTY,
    GLASSES_CONFIDENCE_HIGH,
    GLASSES_CONFIDENCE_LOW,
    WEIGHT_FACE, WEIGHT_SHIRT, WEIGHT_PANTS, WEIGHT_GLASSES,
    WEIGHT_SHIRT_NO_FACE, WEIGHT_PANTS_NO_FACE, WEIGHT_GLASSES_NO_FACE,
)


class MatchResult:
    """All scores for one detection; flows through the pipeline and into the log."""

    __slots__ = [
        "shirt_score", "shirt_detected",
        "pants_score", "pants_detected",
        "face_score", "face_confidence",
        "glasses_score", "glasses_confidence",
        "passed_color_filter", "overall",
        "face_detected",
    ]

    def __init__(self):
        self.shirt_score = 0.0
        self.shirt_detected = "unknown"
        self.pants_score = 0.0
        self.pants_detected = "unknown"
        self.face_score = 0.0
        self.face_confidence = 0.0
        self.glasses_score = 0.5
        self.glasses_confidence = 0.5
        self.passed_color_filter = False
        self.overall = 0.0
        self.face_detected = False

    def to_dict(self):
        return {
            "shirt_score":        round(self.shirt_score, 4),
            "shirt_detected":     self.shirt_detected,
            "pants_score":        round(self.pants_score, 4),
            "pants_detected":     self.pants_detected,
            "face_score":         round(self.face_score, 4),
            "face_confidence":    round(self.face_confidence, 4),
            "glasses_score":      round(self.glasses_score, 4),
            "glasses_confidence": round(self.glasses_confidence, 4),
            "passed_color_filter": self.passed_color_filter,
            "overall":            round(self.overall, 4),
        }


def color_prefilter(person_crop, target_shirt, target_pants):
    # Cheap gate run on every detection. SHIRT-PRIMARY: the shirt must match;
    # pants is a bonus (legs are often hidden). A wrong shirt still rejects.
    result = MatchResult()
    upper_hsv, lower_hsv = extract_body_regions(person_crop)

    if upper_hsv is not None and target_shirt:
        result.shirt_score, result.shirt_detected = match_clothing_color(upper_hsv, target_shirt)
    if lower_hsv is not None and target_pants:
        result.pants_score, result.pants_detected = match_clothing_color(lower_hsv, target_pants)

    if target_shirt:
        result.passed_color_filter = result.shirt_score >= COLOR_MATCH_THRESHOLD
    elif target_pants:
        result.passed_color_filter = result.pants_score >= COLOR_MATCH_THRESHOLD
    else:
        result.passed_color_filter = True   # no colour given → rely on the face
    return result


def full_match(person_crop, face_engine, known_embedding, color_result,
               target_wearing_glasses):
    # Only run on colour survivors: face recognition + glasses, then final score.
    if known_embedding is not None:
        face_data = face_engine.detect_and_compare(person_crop, known_embedding)
        color_result.face_score      = face_data["face_score"]
        color_result.face_confidence = face_data["face_confidence"]
        color_result.face_detected   = face_data["face_confidence"] > 0.0

        if face_data["face_crop"] is not None:
            color_result.glasses_confidence = detect_glasses(face_data["face_crop"])
            if target_wearing_glasses is None:
                color_result.glasses_score = 0.5
            else:
                color_result.glasses_score = score_glasses_match(
                    color_result.glasses_confidence, target_wearing_glasses
                )

    color_result.overall = compute_overall_score(color_result)
    return color_result


def compute_overall_score(result):
    # CLOTHING-PRIMARY: clothing is the reliable signal; a CONFIRMED face only
    # ever raises the score (an unconfirmed face never drags it down).
    clothing_baseline = _clothing_only_score(result)

    face_confirmed = result.face_detected and result.face_score >= FACE_SIMILARITY_THRESHOLD
    if not face_confirmed:
        return clothing_baseline

    components = [
        (WEIGHT_FACE,    result.face_score),
        (WEIGHT_SHIRT,   result.shirt_score),
        (WEIGHT_PANTS,   result.pants_score),
        (WEIGHT_GLASSES, result.glasses_score),
    ]
    active = [(w, v) for w, v in components if v > 0.0]
    if not active:
        return clothing_baseline
    face_aware = sum(w * v for w, v in active) / sum(w for w, _ in active)
    return max(clothing_baseline, face_aware)


def _clothing_only_score(result):
    # A garment counts only when it actually matches (>= threshold); a hidden or
    # wrong garment is neutral, never a penalty. Glasses excluded here.
    components = [
        (WEIGHT_SHIRT_NO_FACE, result.shirt_score),
        (WEIGHT_PANTS_NO_FACE, result.pants_score),
    ]
    active = [(w, v) for w, v in components if v >= COLOR_MATCH_THRESHOLD]
    if not active:
        return 0.0
    score = sum(w * v for w, v in active) / sum(w for w, _ in active)
    return score * NO_FACE_SCORE_FACTOR


def apply_glasses_filter(candidates, target_wearing_glasses):
    # Tie-break between same-clothing people by glasses. Hard-reject only when
    # the (noisy) detector can actually discriminate in this frame; otherwise a
    # lone candidate just gets a small soft penalty so we never drop the target.
    if target_wearing_glasses is None or not candidates:
        return candidates

    strong_clothing = [c for c in candidates
                       if c["result"].shirt_score >= COLOR_MATCH_THRESHOLD]

    def _clearly_mismatched(glasses_conf):
        if target_wearing_glasses:
            return glasses_conf <= GLASSES_CONFIDENCE_LOW
        return glasses_conf >= GLASSES_CONFIDENCE_HIGH

    def _clearly_matched(glasses_conf):
        if target_wearing_glasses:
            return glasses_conf >= GLASSES_CONFIDENCE_HIGH
        return glasses_conf <= GLASSES_CONFIDENCE_LOW

    if len(strong_clothing) >= 2:
        discriminating = any(_clearly_matched(c["result"].glasses_confidence)
                             for c in strong_clothing)
        if discriminating:
            return [c for c in candidates
                    if not _clearly_mismatched(c["result"].glasses_confidence)]
        return candidates

    for c in candidates:
        if _clearly_mismatched(c["result"].glasses_confidence):
            c["result"].overall = max(0.0, c["result"].overall - GLASSES_MISMATCH_PENALTY)
    return candidates
