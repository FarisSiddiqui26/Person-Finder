# AI-Powered Missing Person Surveillance System

**BSCS Final Year Project**

A real-time, CPU-only system that finds a described person across up to 4 CCTV
feeds using a cascaded pipeline: **YOLOv8 (person) → clothing-colour filter →
glasses → FaceNet**. The operator gives a reference photo, a clothing
description (shirt/pants colour) and whether the person wears glasses; the
system watches the feeds and raises an alert (snapshot + JSON log) on a match.

---

## How it works

The design is **clothing-primary**: at CCTV distance a face is rarely
confirmable, so clothing colour is the main signal and a *confirmed* face only
ever raises the score (it never blocks a clothing match).

```
CCTV frame
   │
   ▼  YOLOv8  — detect people (person-first: conf ≥ 0.55, so objects/plants are ignored)
   ▼  Colour  — match shirt (gate) + pants (bonus) on the CENTRE of each box
   ▼  Face    — MTCNN + FaceNet on the colour survivors (budgeted; ~130 ms each)
   ▼  Glasses — eye-band vs cheek-band edge ratio; tie-breaks same-clothing people
   ▼  Track   — IoU tracking + rolling average; alert after a few consistent frames
   ▼  ALERT   — save crop + annotated full-res frame, append JSON to alerts.log
```

Key behaviours:
- **Shirt-primary colour gate** — the shirt must match; pants is a bonus
  (legs are often hidden). A wrong shirt still rejects the person.
- **Centre-column colour sampling** — only the central torso is read, so a
  beige wall behind a person can't be mistaken for their clothing.
- **Glasses as a tie-breaker** — between two people in the same clothing, the
  one whose glasses status matches is preferred.
- **Per-camera face-check budget + face cache** — keeps the CPU real-time.

---

## Project structure

```
person_finder/
├── src/person_finder/        # the installable package
│   ├── app.py                # entry point: interactive setup + camera threads + GUI
│   ├── config.py             # all tunable parameters + paths
│   ├── color_engine.py       # pattern-aware clothing-colour matching (HSV pixel ratios)
│   ├── face_engine.py        # MTCNN face detection + FaceNet embeddings (thread-safe)
│   ├── glasses_detector.py   # eye-band/cheek-band edge-ratio glasses detector
│   ├── matching_engine.py    # cascade orchestration + score combination
│   ├── tracker.py            # IoU tracking, rolling scores, alert gating
│   └── alert_manager.py      # snapshots + JSON-lines logging + console alerts
├── tests/                    # pytest suite (synthetic images, no data needed)
├── scripts/
│   └── evaluate_accuracy.py  # regenerates docs/ACCURACY.md + charts
├── docs/                     # accuracy report, charts, design deep-dive
├── data/known_faces/         # reference photos (git-ignored)
├── models/                   # YOLO weights (git-ignored, auto-downloaded)
└── output/                   # alerts/ snapshots + alerts.log (git-ignored, created at runtime)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # or: pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: tests + accuracy report
```

Put a clear, frontal reference photo in `data/known_faces/` (e.g.
`data/known_faces/affan.jpeg`). The filename (without extension) is the name you
type during setup. `models/yolov8n.pt` is downloaded automatically on first run.

## Run

```bash
person-finder                      # or: python -m person_finder
```

It loads the models, asks for the person's name, shirt colour, pants colour and
glasses, then asks for cameras (video-file paths or RTSP/HTTP URLs, up to 4). Press
**Q** in any window to stop.

Available colours (each covers all its shades):
`beige, black, blue, brown, gray, green, orange, pink, purple, red, teal, white, yellow`

### Environment variables (optional)

| Variable | Default | Meaning |
|---|---|---|
| `PERSON_FINDER_HOME` | project root | base dir the paths below resolve from |
| `PERSON_FINDER_KNOWN_FACES` | `data/known_faces` | reference photos folder |
| `PERSON_FINDER_YOLO_MODEL` | `models/yolov8n.pt` | YOLO weights path |
| `PERSON_FINDER_OUTPUT` | `output` | where `alerts/` and `alerts.log` are written |
| `PERSON_FINDER_DEVICE` | `cpu` | `cuda` for an NVIDIA GPU |

## Tests

```bash
pytest
```

---

## Accuracy

Run `python scripts/evaluate_accuracy.py` to (re)generate
**[`docs/ACCURACY.md`](docs/ACCURACY.md)** and the charts in `docs/accuracy_charts/`.

**Overall pipeline accuracy ≈ 86%**, weighted by the same weights the engine
uses at runtime:

| Component | Accuracy | Evidence |
|---|---|---|
| FaceNet recognition | 86.7% | ROC AUC = 1.000 (perfect identity separation) |
| Clothing-colour matcher | 82.1% (93.3% on colours actually used) | HSV swatch classification |
| Glasses detector | 100% | labelled reference faces |

---

## Configuration (`src/person_finder/config.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `YOLO_CONFIDENCE` | 0.55 | min person confidence (rejects objects/plants) |
| `COLOR_MATCH_THRESHOLD` | 0.55 | min shirt colour score to pass the gate |
| `FACE_SIMILARITY_THRESHOLD` | 0.55 | FaceNet cosine to confirm identity |
| `COMBINED_THRESHOLD` | 0.52 | overall score needed to alert |
| `MIN_CONSECUTIVE_FRAMES` | 3 | times tracked before alerting |
| `ALERT_COOLDOWN_SEC` | 10 | seconds between repeat alerts for one person |
| `TRACK_MAX_AGE_SEC` | 20 | how long a track survives without a new detection |
| `MAX_FACE_CHECKS_PER_FRAME` | 5 | per-frame budget of new face checks |
| `DEBUG_SCORES` | False | True prints every per-frame score (slows multi-cam) |

Resolution, frame-skip and YOLO size adapt to the camera count
(`ADAPTIVE_*` dicts). The `COLORS` / `COLOR_FAMILIES` dicts define every colour.

**Tuning tips:** too many false alerts → raise `COMBINED_THRESHOLD`; missing a
colour in your lighting → widen that colour's HSV range; too slow on 4 cameras →
keep `DEBUG_SCORES=False` and consider downscaling source videos to ~720p.

---

## Output for the backend

Alerts are appended to **`output/alerts.log`** as JSON-lines (one object per line):

```json
{
  "alert_number": 1,
  "timestamp": "2026-05-30 14:30:25",
  "camera": "VIDEO_1",
  "person": "affan",
  "confidence": 0.8075,
  "scores": {
    "shirt_score": 0.95, "shirt_detected": "brown",
    "pants_score": 0.95, "pants_detected": "brown",
    "face_score": 0.0,   "face_confidence": 0.0,
    "glasses_score": 0.5, "glasses_confidence": 0.5,
    "passed_color_filter": true, "overall": 0.8075
  },
  "snapshot_crop": ".../output/alerts/..._crop.jpg",
  "snapshot_frame": ".../output/alerts/..._frame.jpg",
  "frame_number": 452
}
```

Parse it like:

```python
import json
with open("output/alerts.log") as f:
    alerts = [json.loads(line) for line in f if line.strip()]
```

Each alert also saves a person-crop and an annotated full-resolution frame in `output/alerts/`.

---

## Notes / limitations

- CPU-only (no CUDA needed). 4× 2560×1440 video is heavy — keep `DEBUG_SCORES`
  off and downscale very-high-res footage for the smoothest multi-camera runs.
- Glasses and very-similar warm colours (e.g. a brown kurta vs a dusty-pink
  shirt) are tuned for the project's footage; new lighting may need a small HSV
  nudge in `config.py`.
- `pytest` runs the component tests and never writes to `output/`.
