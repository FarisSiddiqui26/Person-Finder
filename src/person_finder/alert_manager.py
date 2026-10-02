"""Alert manager — saves snapshots, appends a JSON-lines record to alerts.log, prints to console."""

import cv2
import json
import os
import threading
from datetime import datetime

from .config import ALERTS_FOLDER, LOG_FILE


class AlertManager:
    # A lock serializes the whole alert across camera threads so alert_number,
    # snapshot filenames and the log line never collide or interleave.

    def __init__(self):
        self.alert_count = 0
        self._lock = threading.Lock()
        os.makedirs(ALERTS_FOLDER, exist_ok=True)

    def trigger_alert(self, camera_name, target_name, track, match_result,
                      person_crop, full_frame, bbox, frame_number,
                      match_type="clothing"):
        with self._lock:
            self.alert_count += 1
            timestamp = datetime.now()
            crop_path, frame_path = self._save_snapshots(
                camera_name, target_name, person_crop, full_frame, bbox,
                timestamp, match_type
            )
            self._log_alert(
                camera_name, target_name, track, match_result,
                crop_path, frame_path, frame_number, timestamp, match_type
            )
            self._print_alert(
                camera_name, target_name, track, match_result,
                crop_path, timestamp, match_type
            )
            return crop_path

    def _save_snapshots(self, camera_name, target_name, person_crop,
                        full_frame, bbox, timestamp, match_type="clothing"):
        ts = timestamp.strftime("%Y%m%d_%H%M%S")
        # alert_count in the name keeps two same-second alerts from overwriting each other.
        base_name = f"{ts}_{camera_name}_{target_name}_a{self.alert_count}"
        crop_path  = os.path.join(ALERTS_FOLDER, f"{base_name}_crop.jpg")
        frame_path = os.path.join(ALERTS_FOLDER, f"{base_name}_frame.jpg")

        try:
            if person_crop is not None and person_crop.size > 0:
                cv2.imwrite(crop_path, person_crop)
            else:
                crop_path = "save_failed"
        except Exception as e:
            print(f"  [AlertManager] Warning: Could not save crop: {e}")
            crop_path = "save_failed"

        try:
            if full_frame is not None:
                annotated = full_frame.copy()
                x1, y1, x2, y2 = bbox
                box_color = (0, 0, 255) if match_type == "face" else (0, 140, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

                label = (f"FOUND (FACE): {target_name.upper()}"
                         if match_type == "face"
                         else f"CLOTHING MATCH: {target_name.upper()}")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                ts_label = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(annotated, ts_label, (10, annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imwrite(frame_path, annotated)
            else:
                frame_path = "save_failed"
        except Exception as e:
            print(f"  [AlertManager] Warning: Could not save frame: {e}")
            frame_path = "save_failed"

        return crop_path, frame_path

    def _log_alert(self, camera_name, target_name, track, match_result,
                   crop_path, frame_path, frame_number, timestamp,
                   match_type="clothing"):
        # JSON-lines, one object per line. The backend depends on this exact schema.
        log_entry = {
            "alert_number":   self.alert_count,
            "timestamp":      timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "camera":         camera_name,
            "person":         target_name,
            "confidence":     round(track.avg_score, 4),
            "scores":         match_result.to_dict(),
            "snapshot_crop":  crop_path,
            "snapshot_frame": frame_path,
            "frame_number":   frame_number,
        }
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [AlertManager] Warning: Could not write log: {e}")

    def _print_alert(self, camera_name, target_name, track, match_result,
                     crop_path, timestamp, match_type="clothing"):
        border = "!" * 62
        time_str = timestamp.strftime("%H:%M:%S")
        date_str = timestamp.strftime("%Y-%m-%d")

        if match_type == "face":
            header = f"ALERT #{self.alert_count} — FACE CONFIRMED MATCH"
        else:
            header = f"ALERT #{self.alert_count} — CLOTHING MATCH (face not confirmed)"

        glasses_str = (f"conf={match_result.glasses_confidence:.0%}, "
                       f"match={match_result.glasses_score:.0%}")
        face_str = (f"{match_result.face_score:.1%} (det: {match_result.face_confidence:.1%})"
                    if match_result.face_detected
                    else "not detected (back/side view)")

        print(f"\n{border}")
        print(f"!!  {header}")
        print(f"{border}")
        print(f"!!")
        print(f"!!  Camera:      {camera_name}")
        print(f"!!  Person:      {target_name.upper()}")
        print(f"!!  Confidence:  {track.avg_score:.1%}")
        print(f"!!  Date:        {date_str}")
        print(f"!!  Time:        {time_str}")
        print(f"!!")
        print(f"!!  ── Score Breakdown ──")
        print(f"!!  Face Match:  {face_str}")
        print(f"!!  Shirt:       {match_result.shirt_detected} "
              f"({match_result.shirt_score:.1%})")
        print(f"!!  Pants:       {match_result.pants_detected} "
              f"({match_result.pants_score:.1%})")
        print(f"!!  Glasses:     {glasses_str}")
        print(f"!!")
        print(f"!!  Snapshot:    {crop_path}")
        print(f"{border}\n")
