import time

from person_finder.tracker import PersonTracker


def test_tracks_and_alerts_after_consistent_frames():
    tracker = PersonTracker()
    for _ in range(6):
        tid = tracker.update((100, 100, 200, 300), 0.65)
    assert tracker.get_track(tid).frame_count == 6
    assert tracker.should_alert(tid)


def test_stale_tracks_are_removed():
    tracker = PersonTracker()
    tid = tracker.update((50, 50, 100, 200), 0.5)
    tracker.tracks[tid].last_seen = time.time() - 10
    tracker.cleanup_stale(max_age_sec=5.0)
    assert len(tracker.tracks) == 0


def test_face_cache_refresh_logic():
    tracker = PersonTracker()
    tid = tracker.update((100, 100, 200, 300), 0.5)
    track = tracker.get_track(tid)

    assert track.needs_face_refresh((100, 100, 200, 300))           # empty cache

    tracker.update((100, 100, 200, 300), 0.5, face_data={
        "face_score": 0.80, "face_conf": 0.95, "face_detected": True,
        "glasses_score": 0.5, "glasses_conf": 0.5,
    })
    assert not track.needs_face_refresh((101, 100, 201, 300))       # fresh

    for _ in range(track.POSITIVE_CACHE_FRAMES + 1):
        tracker.update((100, 100, 200, 300), 0.5)
    assert track.needs_face_refresh((100, 100, 200, 300))           # aged out

    track.face_cache_age = 0
    assert track.needs_face_refresh((300, 300, 400, 500))           # big jump
