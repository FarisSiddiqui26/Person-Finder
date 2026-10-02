"""Person tracker — links detections across frames (IoU) and gates alerts."""

import time
import numpy as np

from .config import MIN_CONSECUTIVE_FRAMES, ALERT_COOLDOWN_SEC, COMBINED_THRESHOLD


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class TrackedPerson:
    """One tracked person: rolling scores, best snapshot, and a face-result cache."""

    __slots__ = [
        "bbox", "scores", "frame_count", "last_alert_time",
        "last_seen", "best_score", "best_snapshot",
        "cached_face_score", "cached_face_conf", "cached_face_detected",
        "cached_glasses_score", "cached_glasses_conf",
        "face_cache_age", "last_bbox_center",
    ]

    # Face cache lifetimes (in processed frames) — a face check is the CPU bottleneck.
    POSITIVE_CACHE_FRAMES = 25
    NEGATIVE_CACHE_FRAMES = 10
    MOVE_REFRESH_FRACTION = 0.40

    def __init__(self, bbox, score):
        self.bbox = bbox
        self.scores = [float(score)]
        self.frame_count = 1
        self.last_alert_time = 0.0
        self.last_seen = time.time()
        self.best_score = float(score)
        self.best_snapshot = None
        self.cached_face_score = None
        self.cached_face_conf = None
        self.cached_face_detected = None
        self.cached_glasses_score = None
        self.cached_glasses_conf = None
        self.face_cache_age = 0
        self.last_bbox_center = _bbox_center(bbox)

    def update(self, bbox, score, snapshot=None, face_data=None):
        self.bbox = bbox
        self.scores.append(float(score))
        self.frame_count += 1
        self.last_seen = time.time()
        self.last_bbox_center = _bbox_center(bbox)

        if len(self.scores) > 15:           # rolling window of recent scores
            self.scores.pop(0)

        if score > self.best_score:
            self.best_score = float(score)
            if snapshot is not None:
                self.best_snapshot = snapshot.copy()

        if face_data is not None:           # fresh face result → refresh cache
            self.cached_face_score    = face_data.get("face_score", 0.0)
            self.cached_face_conf     = face_data.get("face_conf", 0.0)
            self.cached_face_detected = face_data.get("face_detected", False)
            self.cached_glasses_score = face_data.get("glasses_score", 0.5)
            self.cached_glasses_conf  = face_data.get("glasses_conf", 0.5)
            self.face_cache_age = 0
        else:
            self.face_cache_age += 1

    def needs_face_refresh(self, bbox):
        # Reuse the cached face unless it's stale, or the box jumped (likely a new person).
        if self.cached_face_detected is None:
            return True

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        movement = ((cx - self.last_bbox_center[0]) ** 2 +
                    (cy - self.last_bbox_center[1]) ** 2) ** 0.5
        bw, bh = (x2 - x1), (y2 - y1)
        diag = (bw * bw + bh * bh) ** 0.5
        if diag > 0 and (movement / diag) > self.MOVE_REFRESH_FRACTION:
            return True

        if self.cached_face_detected:
            return self.face_cache_age >= self.POSITIVE_CACHE_FRAMES
        return self.face_cache_age >= self.NEGATIVE_CACHE_FRAMES

    @property
    def avg_score(self):
        return float(np.mean(self.scores)) if self.scores else 0.0


class PersonTracker:
    """IoU tracker: assigns each detection to a track and decides when to alert."""

    def __init__(self):
        self.tracks = {}
        self._next_id = 0

    def update(self, bbox, overall_score, snapshot=None, face_data=None):
        track_id = self._match_by_iou(bbox)

        if track_id is None:                # new person → new track
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = TrackedPerson(bbox, overall_score)
            if snapshot is not None:
                self.tracks[track_id].best_snapshot = snapshot.copy()
            if face_data is not None:       # seed the cache if we already ran a face check
                t = self.tracks[track_id]
                t.cached_face_score    = face_data.get("face_score", 0.0)
                t.cached_face_conf     = face_data.get("face_conf", 0.0)
                t.cached_face_detected = face_data.get("face_detected", False)
                t.cached_glasses_score = face_data.get("glasses_score", 0.5)
                t.cached_glasses_conf  = face_data.get("glasses_conf", 0.5)
                t.face_cache_age = 0
        else:
            self.tracks[track_id].update(bbox, overall_score, snapshot, face_data)

        return track_id

    def find_track_id(self, bbox):
        return self._match_by_iou(bbox)

    def should_alert(self, track_id):
        # Fire only after enough frames, a high enough average, and the cooldown.
        track = self.tracks.get(track_id)
        if track is None:
            return False
        if track.frame_count < MIN_CONSECUTIVE_FRAMES:
            return False
        if track.avg_score < COMBINED_THRESHOLD:
            return False
        if (time.time() - track.last_alert_time) < ALERT_COOLDOWN_SEC:
            return False
        return True

    def mark_alerted(self, track_id):
        track = self.tracks.get(track_id)
        if track:
            track.last_alert_time = time.time()

    def get_track(self, track_id):
        return self.tracks.get(track_id)

    def cleanup_stale(self, max_age_sec=5.0):
        now = time.time()
        stale = [tid for tid, t in self.tracks.items()
                 if (now - t.last_seen) > max_age_sec]
        for tid in stale:
            del self.tracks[tid]

    def _match_by_iou(self, bbox, min_iou=0.3):
        x1, y1, x2, y2 = bbox
        best_iou = min_iou
        best_id = None
        for tid, track in self.tracks.items():
            tx1, ty1, tx2, ty2 = track.bbox
            inter = (max(0, min(x2, tx2) - max(x1, tx1)) *
                     max(0, min(y2, ty2) - max(y1, ty1)))
            union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        return best_id
