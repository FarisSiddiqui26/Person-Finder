import cv2
import numpy as np
import pytest

from person_finder.color_engine import (
    extract_body_regions, match_clothing_color, normalize_lighting,
)
from person_finder.config import FAMILY_MISMATCH_CAP

from conftest import BLUE_BGR, BROWN_BGR, build_person


def test_solid_colors_match(blue_black_person):
    upper, lower = extract_body_regions(blue_black_person)
    assert upper is not None and lower is not None
    shirt_score, _ = match_clothing_color(upper, "blue")
    pants_score, _ = match_clothing_color(lower, "black")
    assert shirt_score >= 0.50
    assert pants_score >= 0.50


def test_striped_pattern_matches_dominant_color():
    striped = np.zeros((200, 80, 3), dtype=np.uint8)
    for i in range(36, 100, 10):
        striped[i:i + 5, :] = BLUE_BGR
        if i + 5 < 100:
            striped[i + 5:i + 10, :] = (255, 255, 255)
    striped[100:164, :] = (20, 20, 20)

    upper, _ = extract_body_regions(striped)
    assert upper is not None
    score, _ = match_clothing_color(upper, "blue")
    assert score >= 0.50


def test_wrong_color_rejected(blue_black_person):
    upper, _ = extract_body_regions(blue_black_person)
    score, _ = match_clothing_color(upper, "red")
    assert score < 0.50


def test_clahe_brightens_dark_image():
    dark = np.full((100, 80, 3), 25, dtype=np.uint8)
    assert normalize_lighting(dark).mean() > dark.mean()


@pytest.mark.parametrize("upper_bgr", [
    (80, 30, 10),      # navy
    (235, 200, 130),   # sky blue
    (200, 80, 30),     # royal blue
])
def test_blue_family_covers_all_shades(upper_bgr):
    upper, _ = extract_body_regions(build_person(upper_bgr))
    score, _ = match_clothing_color(upper, "blue")
    assert score >= 0.70


def test_family_mismatch_cap_blocks_incidental_pixels():
    # Mostly-blue shirt with ~20% brown-ish pixels must not score high as "brown".
    region = np.zeros((100, 80, 3), dtype=np.uint8)
    region[:80, :] = BLUE_BGR
    region[80:, :] = BROWN_BGR
    score, dominant = match_clothing_color(cv2.cvtColor(region, cv2.COLOR_BGR2HSV), "brown")
    assert dominant == "blue"
    assert score <= FAMILY_MISMATCH_CAP + 1e-6


def test_same_family_keeps_high_score():
    region = np.full((100, 80, 3), (80, 30, 10), dtype=np.uint8)   # navy
    score, dominant = match_clothing_color(cv2.cvtColor(region, cv2.COLOR_BGR2HSV), "blue")
    assert dominant == "blue"
    assert score >= 0.70
