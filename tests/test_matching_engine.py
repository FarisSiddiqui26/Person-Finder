import pytest

from person_finder.config import GLASSES_MISMATCH_PENALTY
from person_finder.matching_engine import MatchResult, apply_glasses_filter, color_prefilter

from conftest import BLUE_BGR, BROWN_BGR, build_person


def test_match_result_serialization():
    mr = MatchResult()
    mr.shirt_score = 0.85
    mr.face_score = 0.72
    mr.pants_detected = "black"
    d = mr.to_dict()
    assert len(d) == 10
    assert d["shirt_score"] == 0.85
    assert d["pants_detected"] == "black"


def test_prefilter_passes_correct_candidate(blue_black_person):
    assert color_prefilter(blue_black_person, "blue", "black").passed_color_filter


@pytest.mark.parametrize("shirt, pants, expected", [
    (BLUE_BGR,  BROWN_BGR, False),   # wrong shirt rejects even with right pants
    (BROWN_BGR, BLUE_BGR,  True),    # right shirt passes; pants is only a bonus
    (BROWN_BGR, BROWN_BGR, True),    # full match
])
def test_prefilter_is_shirt_primary(shirt, pants, expected):
    result = color_prefilter(build_person(shirt, pants), "brown", "brown")
    assert result.passed_color_filter is expected


def _cand(shirt_score, glasses_conf, label):
    r = MatchResult()
    r.shirt_score = shirt_score
    r.glasses_confidence = glasses_conf
    r.overall = 0.60
    return {"bbox": (0, 0, 10, 10), "result": r, "person_crop": None,
            "fresh_face_data": None, "label": label}


def test_glasses_filter_drops_clear_mismatch_when_ambiguous():
    cands = [_cand(0.70, 0.80, "keep"), _cand(0.70, 0.10, "drop")]
    survivors = apply_glasses_filter(cands, target_wearing_glasses=True)
    assert [c["label"] for c in survivors] == ["keep"]


def test_glasses_filter_soft_penalty_for_lone_candidate():
    lone = [_cand(0.70, 0.10, "lone")]
    survivors = apply_glasses_filter(lone, target_wearing_glasses=True)
    assert len(survivors) == 1
    assert survivors[0]["result"].overall == pytest.approx(0.60 - GLASSES_MISMATCH_PENALTY)


def test_glasses_filter_bypassed_when_unknown():
    cands = [_cand(0.70, 0.80, "a"), _cand(0.70, 0.10, "b")]
    survivors = apply_glasses_filter(cands, target_wearing_glasses=None)
    assert len(survivors) == 2
    assert all(c["result"].overall == 0.60 for c in survivors)
