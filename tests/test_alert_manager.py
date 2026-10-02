import json

import numpy as np

from person_finder import alert_manager
from person_finder.matching_engine import MatchResult
from person_finder.tracker import TrackedPerson


def test_alert_writes_snapshots_and_json_line(tmp_path, monkeypatch):
    alerts_dir = tmp_path / "alerts"
    log_file = tmp_path / "alerts.log"
    monkeypatch.setattr(alert_manager, "ALERTS_FOLDER", str(alerts_dir))
    monkeypatch.setattr(alert_manager, "LOG_FILE", str(log_file))

    am = alert_manager.AlertManager()
    assert alerts_dir.is_dir()

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    am.trigger_alert(
        camera_name="CAM_1", target_name="test", track=TrackedPerson((10, 10, 50, 100), 0.8),
        match_result=MatchResult(), person_crop=frame[10:100, 10:50], full_frame=frame,
        bbox=(10, 30, 50, 100), frame_number=1,
    )

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["person"] == "test"
    assert len(list(alerts_dir.glob("*.jpg"))) == 2
