"""All tunable parameters for the surveillance pipeline (thresholds, colours, weights)."""

import os
from pathlib import Path

# ── Processing ──
PROCESS_WIDTH  = 640
PROCESS_HEIGHT = 360
SKIP_FRAMES = 3
DEVICE = os.getenv("PERSON_FINDER_DEVICE", "cpu")   # "cuda" if you have an NVIDIA GPU

# Adaptive per camera count (more cameras → cheaper processing to stay real-time).
ADAPTIVE_SKIP = {1: 3, 2: 4, 3: 5, 4: 6}
ADAPTIVE_RESOLUTIONS = {1: (640, 360), 2: (640, 360), 3: (512, 288), 4: (384, 216)}
ADAPTIVE_YOLO_IMGSZ = {1: 640, 2: 512, 3: 512, 4: 384}
YOLO_MAX_DETECTIONS = 25
DISPLAY_QUEUE_BACKPRESSURE = 8          # file workers skip-decode if this far ahead of the GUI

# ── Detection thresholds ──
# Person-first gate: below this, objects (e.g. a plant pot) get mislabelled as
# people. Real people score ~0.66-0.78, so this cleanly rejects non-people.
YOLO_CONFIDENCE = 0.55
FACE_SIMILARITY_THRESHOLD = 0.55        # FaceNet cosine: same person if >= this
COLOR_MATCH_THRESHOLD = 0.55            # min colour score to pass the pre-filter
FAMILY_MISMATCH_CAP = 0.40             # cap a colour score when the dominant colour is a different family
COMBINED_THRESHOLD = 0.52              # overall score needed to alert

# Glasses detector: conf = 0.5 + (eye/cheek edge ratio - PIVOT) * SCALE.
GLASSES_RATIO_PIVOT = 1.40
GLASSES_RATIO_SCALE = 0.90

NO_FACE_SCORE_FACTOR = 0.85            # slight penalty on clothing-only (no face) scores

# Glasses tie-break between same-clothing people.
GLASSES_MISMATCH_PENALTY = 0.25
GLASSES_CONFIDENCE_HIGH = 0.55
GLASSES_CONFIDENCE_LOW  = 0.45

# Crowd mode caps expensive face checks per frame to protect frame rate.
CROWD_THRESHOLD = 10
MAX_FACE_CHECKS_PER_FRAME = 5
CROWD_MAX_FACE_CHECKS     = 3

DEBUG_SCORES = False                    # True prints every per-frame score (slows multi-cam)

# ── Small-face handling ──
MIN_CROP_HEIGHT_FOR_FACE = 100          # upscale person crops shorter than this before MTCNN
FACE_UPSCALE_FACTOR = 2.0

# ── Alerts ──
MIN_CONSECUTIVE_FRAMES = 3              # times tracked before alerting
ALERT_COOLDOWN_SEC = 10                 # seconds between repeat alerts for the same person
# Track lifetime without a new detection. Must exceed the wall-clock gap between
# processed frames on a slow camera, or tracks die between detections and never alert.
TRACK_MAX_AGE_SEC = 20.0

# ── Paths ──
# Everything resolves from one base dir (the project root by default), so the
# app behaves the same whatever directory it's launched from. Each path can be
# overridden with an environment variable.
BASE_DIR = Path(os.getenv("PERSON_FINDER_HOME", Path(__file__).resolve().parents[2]))
KNOWN_FACES_FOLDER = os.getenv("PERSON_FINDER_KNOWN_FACES", str(BASE_DIR / "data" / "known_faces"))
YOLO_MODEL_PATH    = os.getenv("PERSON_FINDER_YOLO_MODEL", str(BASE_DIR / "models" / "yolov8n.pt"))
OUTPUT_DIR         = os.getenv("PERSON_FINDER_OUTPUT", str(BASE_DIR / "output"))
ALERTS_FOLDER      = os.path.join(OUTPUT_DIR, "alerts")
LOG_FILE           = os.path.join(OUTPUT_DIR, "alerts.log")

CAMERA_LABELS = {0: "CAMERA_01", 1: "CAMERA_02", 2: "CAMERA_03", 3: "CAMERA_04"}

# ── HSV colour ranges: name → list of (lower, upper). OpenCV hue is 0-179.
# Some colours wrap around hue 0/180 so they need two ranges.
COLORS = {
    "red":         [((0,   70,  50),  (10,  255, 255)),
                    ((170, 70,  50),  (180, 255, 255))],
    "dark_red":    [((0,   80,  30),  (8,   255, 110)),
                    ((172, 80,  30),  (180, 255, 110))],
    "crimson":     [((0,   100, 60),  (8,   255, 200)),
                    ((172, 100, 60),  (180, 255, 200))],
    # maroon = dark red-brown; value-capped so it stays brown, not dim pink (see "mauve").
    "maroon":      [((0,   40,  20),  (12,  66,  115)),
                    ((167, 40,  20),  (180, 66,  115))],

    "orange":      [((10,  80,  60),  (22,  255, 255))],
    "amber":       [((12,  150, 150), (22,  255, 255))],

    "yellow":      [((22,  80,  90),  (32,  255, 255))],
    "gold":        [((22,  120, 130), (32,  255, 255))],

    # Pink, lightest → deepest (widened to catch washed-out / dim pinks).
    "pink":        [((140, 30,  80),  (174, 255, 255))],
    "hot_pink":    [((150, 100, 130), (173, 255, 255))],
    "fuchsia":     [((144, 150, 130), (168, 255, 255))],
    "deep_pink":   [((143, 120, 70),  (171, 255, 175))],
    "rose":        [((140, 30,  85),  (173, 180, 205))],
    "light_pink":  [((143, 20,  178), (177, 130, 255))],
    "pale_pink":   [((141, 12,  198), (179, 80,  255))],
    "salmon_pink": [((0,   45,  165), (9,   160, 255)),
                    ((170, 45,  165), (179, 160, 255))],
    # mauve = dim/dusty pink: pink (not brown) when more saturated OR brighter than maroon.
    "mauve":       [((153, 66,  60),  (179, 150, 210)),
                    ((153, 30,  116), (179, 150, 255))],

    # Brown is dark (value capped at 110) so bright beige walls/floors don't score as brown.
    "brown":       [((8,   40,  20),  (22,  200, 110))],
    "tan":         [((15,  50,  130), (25,  150, 220))],
    "dark_brown":  [((8,   50,  20),  (20,  200, 90))],
    "chocolate":   [((10,  60,  30),  (18,  200, 80))],
    "coffee":      [((10,  40,  20),  (18,  150, 65))],
    "chestnut":    [((5,   80,  40),  (12,  200, 100))],
    "sienna":      [((6,   120, 60),  (12,  220, 140))],
    "rust":        [((8,   140, 60),  (15,  230, 140))],
    "terracotta":  [((8,   120, 90),  (14,  220, 160))],
    "caramel":     [((15,  80,  120), (22,  180, 200))],
    "burgundy":    [((0,   100, 25),  (8,   220, 90)),
                    ((172, 100, 25),  (180, 220, 90))],
    "wine":        [((0,   120, 35),  (6,   230, 100)),
                    ((174, 120, 35),  (180, 230, 100))],

    "blue":        [((100, 60,  40),  (130, 255, 255))],
    "navy":        [((100, 60,  20),  (130, 255, 120))],
    "sky_blue":    [((95,  30,  150), (115, 180, 255))],
    "light_blue":  [((90,  25,  180), (115, 100, 255))],
    "royal_blue":  [((105, 150, 120), (125, 255, 230))],

    "green":       [((35,  40,  40),  (85,  255, 255))],
    "dark_green":  [((40,  60,  20),  (75,  255, 90))],
    "light_green": [((40,  40,  150), (75,  180, 255))],
    "forest_green":[((45,  100, 30),  (80,  255, 100))],
    "lime":        [((35,  100, 150), (50,  255, 255))],
    "olive":       [((25,  30,  30),  (50,  200, 150))],

    "teal":        [((80,  40,  40),  (100, 255, 255))],
    "cyan":        [((85,  60,  60),  (100, 255, 255))],

    "purple":      [((125, 40,  40),  (160, 255, 255))],
    "violet":      [((125, 60,  60),  (145, 255, 230))],
    "lavender":    [((125, 20,  170), (155, 100, 255))],

    # Neutrals — defined by saturation/value, not hue.
    "white":       [((0,   0,   170), (180, 50,  255))],
    "black":       [((0,   0,   0),   (180, 255, 55))],
    "gray":        [((0,   0,   55),  (180, 45,  175))],
    "silver":      [((0,   0,   130), (180, 30,  200))],
    "beige":       [((15,  15,  140), (30,  80,  230))],
}

# Operator picks a family name; the matcher ORs every shade in it. VALID_COLORS = the 13 families.
COLOR_FAMILIES = {
    "blue":   ["blue", "navy", "sky_blue", "light_blue", "royal_blue"],
    "green":  ["green", "dark_green", "light_green", "forest_green", "lime", "olive"],
    "red":    ["red", "dark_red", "crimson"],
    "yellow": ["yellow", "gold"],
    "purple": ["purple", "violet", "lavender"],
    "gray":   ["gray", "silver"],
    "orange": ["orange", "amber"],
    "brown":  ["brown", "tan", "dark_brown", "maroon",
               "chocolate", "coffee", "chestnut", "sienna", "rust",
               "terracotta", "caramel", "burgundy", "wine"],
    "pink":   ["pink", "hot_pink", "fuchsia", "deep_pink", "rose",
               "light_pink", "pale_pink", "salmon_pink", "mauve"],
    "teal":   ["teal", "cyan"],
    "white":  ["white"],
    "black":  ["black"],
    "beige":  ["beige"],
}
VALID_COLORS = sorted(COLOR_FAMILIES.keys())

# ── Scoring weights (re-normalised over the active signals at runtime) ──
WEIGHT_FACE    = 0.50
WEIGHT_SHIRT   = 0.25
WEIGHT_PANTS   = 0.15
WEIGHT_GLASSES = 0.10
WEIGHT_SHIRT_NO_FACE   = 0.45
WEIGHT_PANTS_NO_FACE   = 0.35
WEIGHT_GLASSES_NO_FACE = 0.20

# ── CLAHE lighting normalisation (used for face crops) ──
CLAHE_CLIP_LIMIT   = 2.5
CLAHE_TILE_SIZE    = (8, 8)
CLAHE_FOR_FACES    = True
