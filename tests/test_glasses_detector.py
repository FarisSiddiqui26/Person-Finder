import numpy as np

from person_finder.glasses_detector import detect_glasses, score_glasses_match


def test_scores_valid_and_complementary():
    rng = np.random.default_rng(0)
    face = rng.integers(50, 200, (100, 80, 3), dtype=np.uint8)
    conf = detect_glasses(face)
    assert 0.0 <= conf <= 1.0
    assert abs(score_glasses_match(conf, True) + score_glasses_match(conf, False) - 1.0) < 0.01


def test_unknown_when_no_face():
    assert detect_glasses(None) == 0.5
