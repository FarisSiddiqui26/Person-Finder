# AI-Powered Missing Person Surveillance System — FYP Deep Dive

*Prepared for final evaluation. This document explains what the system does, how every component works internally, why each technical choice was made, and the measured accuracy — so you can defend the project in front of academic and industry evaluators.*

---

## 1. Problem Statement

Given a **reference photo** of a missing person plus a description of their **current clothing** (shirt/pants colour) and whether they wear **glasses**, the system watches up to **4 CCTV feeds simultaneously** (video files or RTSP/HTTP live streams) and raises an alert — with a saved snapshot and a JSON log entry — the moment it spots someone who matches, in real time, on **CPU only** (no GPU required).

This mirrors a real operational constraint: most CCTV control rooms don't have GPU servers, and at typical camera distance a face is often too small, angled, or low-resolution for face recognition alone to be reliable. The system is built around that constraint rather than around an idealized high-resolution face-recognition problem.

---

## 2. High-Level Architecture

The system is a **cascaded pipeline**: each stage is a cheap filter that only lets promising candidates through to the next, more expensive stage. This is a classic "coarse-to-fine" or "cascade classifier" pattern (the same idea behind Viola–Jones face detection), applied here to keep the total cost low enough to run 4 camera feeds live on a CPU.

```
CCTV frame
   │
   ▼  YOLOv8n (person detector)      — reject everything that isn't a person, conf ≥ 0.55
   ▼  Colour pre-filter (HSV)        — shirt must match (gate); pants is a bonus signal
   ▼  Face check (MTCNN + FaceNet)   — only run on colour survivors, budgeted per frame
   ▼  Glasses detector               — tie-breaks people wearing visually identical clothing
   ▼  IoU Tracker                    — links detections into tracks across frames, smooths noise
   ▼  Alert decision                 — fires after N consistent frames above threshold
   ▼  Alert Manager                  — saves crop + annotated frame, appends JSON-lines log
```

Each camera runs on its **own thread** with its own YOLO model instance and its own tracker, but all cameras **share** one or two `FaceEngine` instances (MTCNN+FaceNet are the expensive, memory-heavy models, so they're not duplicated per camera). A background main thread owns the OpenCV GUI, because `cv2.imshow`/`waitKey` must run on the main thread — worker threads hand it frames through a bounded queue.

---

## 3. Technology Stack and Why Each Piece Was Chosen

| Component | Technology | Why this, not the alternatives |
|---|---|---|
| Person detection | **YOLOv8-nano** (Ultralytics) | Real-time on CPU (the "nano" variant trades some accuracy for speed); single-stage detector avoids the two-pass cost of Faster R-CNN; pre-trained on COCO so "person" (class 0) needs no custom training data. |
| Face detection | **MTCNN** (facenet-pytorch) | Purpose-built for small/angled faces typical of CCTV; returns bounding box + confidence + landmarks in one pass; a 3-stage cascade (P-Net/R-Net/O-Net) itself, so it's cheap when it rejects early. |
| Face recognition | **FaceNet (InceptionResnetV1, pretrained on VGGFace2)** | Produces a 512-D embedding where cosine similarity directly measures identity similarity — no classifier training needed for new identities (one-shot: a single reference photo is enough). This is essential because the system must recognise *any* missing person from *one* photo, not a fixed enrolled set. |
| Clothing colour | **Custom HSV pixel-ratio matcher** (OpenCV) | Deep-learning colour/attribute classifiers need labelled training data and are overkill for a well-understood, low-dimensional problem; HSV separates colour (hue) from lighting (value) far better than RGB, so a classical thresholding approach is both cheaper and more robust to CCTV lighting than a CNN would be without a large annotated dataset. |
| Glasses detection | **Custom edge-ratio heuristic** (Sobel + Canny) | Same reasoning — a lightweight, explainable, training-free signal that only needs to *discriminate*, not classify with high absolute accuracy, because it's used purely as a tie-breaker between already-similar candidates. |
| Tracking | **IoU-based tracker** (custom) | Cheaper than DeepSORT/ByteTrack (no re-ID network), and sufficient because identity re-confirmation is already handled by the face/colour cascade — the tracker's only job is temporal smoothing and alert gating, not long-term re-identification. |
| Runtime | **CPU-only, PyTorch backend** | Matches the deployment reality (commodity CCTV control-room hardware, Intel Iris Xe in this build) — the whole design (frame skipping, adaptive resolution, face-check budgets, caching) exists specifically to make CPU-only real-time performance possible. |

---

## 4. Module-by-Module Deep Dive

### 4.1 `main.py` — Orchestration
Interactive setup collects: reference photo → face embedding, shirt colour, pants colour, glasses yes/no/unknown. Then up to 4 camera sources are configured (video file path or `rtsp://`/`http(s)://` URL). Each camera gets its own thread (`camera_worker`) running the full pipeline; a shutdown `threading.Event` and OpenCV's `Q` key stop everything cleanly. Live streams auto-reconnect on drop; video files loop from the start when they end (useful for demoing).

Per-camera settings **adapt to camera count** (`ADAPTIVE_*` dicts in `config.py`): with 4 cameras running at once, resolution drops to 384×216, frame-skip rises to every 6th frame, and YOLO's input size drops to 384 — trading a bit of accuracy for the CPU budget needed to stay real-time across all feeds simultaneously. With 3+ cameras a second `FaceEngine` is loaded to halve lock contention (only one thread can use a given `FaceEngine` at a time — MTCNN/FaceNet aren't thread-safe).

### 4.2 `color_engine.py` — Clothing Colour Matching
- **Region extraction**: from each person bounding box, the **shirt region** is the 18–50% height band and the **pants region** is 50–82%, sampled only from the **centre 25–75% width** — this avoids picking up background pixels at the box edges, which is the single biggest source of false colour reads in real CCTV footage.
- **Colour space**: HSV, not RGB — hue captures "what colour" independent of "how bright", which matters enormously outdoors/indoors under varying lighting.
- **Matching**: rather than a single centroid distance, the matcher counts what **fraction of pixels** in the region fall inside the target colour's HSV range(s) (a `cv2.inRange` mask, morphologically opened to remove single-pixel noise). This makes it **pattern-aware for free** — a striped or two-tone shirt where 40%+ of pixels are "blue" still scores as blue, without any special-case code.
- **Colour families**: 45 fine-grained HSV shades (navy, sky_blue, royal_blue, maroon, wine, …) are grouped into **13 operator-facing families** (e.g. "blue" covers 5 shades). The operator only ever picks a family; the matcher ORs every shade's mask.
- **Family-mismatch cap**: if the *dominant* colour in the region belongs to a different family than the target, the score is capped at `FAMILY_MISMATCH_CAP = 0.40` even if a coincidental minority of pixels matched — this stops a blue shirt with a small brownish shadow from scoring as a false "brown match".
- **Shirt-primary gating**: the shirt colour is the hard gate; pants is a bonus-only signal, because legs are very often occluded (sitting, tables, crowd) or out of frame. A correct shirt with wrong/hidden pants still passes; a wrong shirt is always rejected even with correct pants.

### 4.3 `face_engine.py` — Face Detection + Recognition
- One shared `FaceEngine` per 1–2 cameras, guarded by a `threading.Lock` because neither MTCNN nor FaceNet is thread-safe and concurrent CPU inference would just thrash the same cores anyway.
- **Small-face handling**: CCTV person crops are often too small for MTCNN's default 20px minimum face size, so crops shorter than 100px are **upscaled 2×** (bicubic) before detection.
- **Best-face selection**: MTCNN can detect multiple faces in one crop (bystanders); the engine keeps the **largest** confident (≥0.5 prob) face, since a bystander is usually smaller/farther than the tracked person.
- **Lighting normalisation**: CLAHE (Contrast-Limited Adaptive Histogram Equalisation) is applied to the L-channel of the face crop in LAB colour space before embedding — evens out CCTV exposure without distorting colour, improving embedding stability.
- **Embedding**: the face crop is resized to 160×160, normalised to `[-1, 1]`, and passed through `InceptionResnetV1` (pretrained on VGGFace2) to get a 512-D embedding. Identity match = cosine similarity between this and the reference embedding, thresholded at `FACE_SIMILARITY_THRESHOLD = 0.55`.
- A detection is only trusted if the MTCNN confidence is **≥ 0.70** — below that, the "face" is likely a false positive from motion blur or occlusion.

### 4.4 `glasses_detector.py`
Purely classical: compares Sobel-gradient strength and Canny edge density in the **eye band** (28–50% of face height) versus the **cheek band** (55–78%). Glasses frames/bridges add edges to the eye band that skin doesn't; the ratio of the two bands cancels out resolution/scale, so it works across crop sizes without calibration per camera. The ratio is linearly mapped to a 0–1 confidence around a pivot of 1.40 (empirically the ratio for "no glasses" faces). This is deliberately **not** a CNN classifier: training one would need a labelled glasses dataset the project doesn't have, and the discriminative task here is easy enough that a hand-tuned edge heuristic reaches 100% accuracy on the reference set (see §6).

### 4.5 `matching_engine.py` — Score Fusion
This is the most important file to be able to explain, because it encodes the project's central design philosophy: **clothing-primary, face-boosts-only**.

- `color_prefilter()` — cheap gate run on *every* detected box before any face/glasses work happens.
- `full_match()` — only runs face + glasses checks on colour survivors.
- `_compute_overall_score()` — computes two numbers and takes the **max**:
  1. `_clothing_only_score()` — weighted average of shirt/pants scores (only garments that actually cleared the colour threshold count; a hidden/wrong garment is neutral, never a penalty), scaled down slightly (`NO_FACE_SCORE_FACTOR = 0.85`) since clothing alone is weaker evidence than a confirmed face.
  2. A face-aware weighted score (face 50%, shirt 25%, pants 15%, glasses 10%) — but **only computed at all if the face is "confirmed"** (detected AND similarity ≥ threshold).
  
  Taking the **max** of the two means: an unconfirmed/absent face (side profile, occlusion, too far away — extremely common in real CCTV) never drags the score down below what clothing alone earned; but a strong face match can only ever push the score *up*. This directly reflects the operational reality that faces are the exception, not the rule, at CCTV range.
- `apply_glasses_filter()` — only engages when **2+ people share matching clothing** in the same frame (a genuine ambiguous case). If the glasses reading clearly discriminates between them, the mismatched one is hard-rejected; if there's only one candidate, a mismatch gets a small soft penalty instead of an outright rejection, because the heuristic detector is too noisy to be the sole reason to drop the only lead.

### 4.6 `tracker.py` — Temporal Consistency
An **IoU (Intersection-over-Union) tracker**: each new detection is matched to the existing track with highest IoU ≥ 0.3, or spawns a new track. Each track keeps a **rolling window of the last 15 scores** and alerts only when: (a) tracked for ≥3 consecutive detections, (b) the rolling average score ≥ `COMBINED_THRESHOLD` (0.52), and (c) at least 10 seconds since the last alert for that track (cooldown, avoids alert spam). Stale tracks (not seen for 20s) are pruned every detection cycle.

A key efficiency feature lives here too: **face-result caching per track**. Running MTCNN+FaceNet on every person, every processed frame, is the CPU bottleneck (~130ms per call). Instead, once a track has a face result, it's reused for **25 frames if positive / 10 frames if negative**, unless the box has moved more than 40% of its own diagonal (likely a different person/occlusion swap) — in which case the cache is invalidated immediately regardless of age.

### 4.7 `alert_manager.py`
Thread-safe (single lock, since multiple camera threads can alert concurrently): saves the best snapshot crop, an annotated full-resolution frame (box + label + timestamp burned in), appends a structured JSON-lines record to `alerts.log` (camera, person, confidence, full score breakdown, snapshot paths, frame number), and prints a formatted console alert. The JSON-lines format is a deliberate integration choice — it's a stable, line-oriented schema a backend/dashboard can tail and parse incrementally without re-reading the whole file.

### 4.8 Crowd handling
If a frame has ≥10 detected people (`CROWD_THRESHOLD`), the system enters **crowd mode**: the per-frame face-check budget drops from 5 to 3, and boxes are processed **largest-area first** (closer/bigger people are more likely to be identifiable and are prioritised for the expensive face check under budget pressure).

---

## 5. Key Design Decisions — the "why" behind the non-obvious choices

1. **Why cascade instead of running everything on every detection?** Face recognition (MTCNN+FaceNet) dominates the CPU budget. Gating it behind a near-free colour check means the expensive model only runs on plausible candidates — this is what makes 4 concurrent camera feeds on CPU feasible at all.
2. **Why clothing-primary rather than face-primary?** At real CCTV distance/angle, a confirmable frontal face is the exception. Designing the score so clothing alone can reach the alert threshold — and a face can only add confidence, never subtract it — matches how a human operator would actually reason ("I can't see their face, but that's definitely the shirt").
3. **Why per-track face caching with movement-based invalidation?** A naive per-frame face check would make multi-camera real-time impossible on CPU; a time-based-only cache risks stale identity data if two people swap places within a box. Combining a frame-based TTL with an IoU/movement re-check gets both speed and correctness.
4. **Why HSV pixel-ratio matching instead of a trained classifier for colour?** No labelled clothing-colour dataset exists for this project's footage; HSV thresholding is deterministic, instantly explainable to an evaluator, and (per the accuracy report) performs strongly on the colours actually used.
5. **Why max(clothing_score, face_aware_score) instead of a fixed weighted sum?** A fixed weighted sum would always penalize the (very common) no-face case, since the face weight's "slot" would just go unfilled. Taking the max removes that penalty while still rewarding a confirmed face.

---

## 6. Measured Accuracy (from `accuracy.py` / `ACCURACY.md`)

Each of the three scored components is evaluated independently with standard ML metrics, then combined using the **same fusion weights the live system uses** (face 0.50, clothing 0.40, glasses 0.10):

| Component | Method | Result |
|---|---|---|
| **FaceNet identity recognition** | Cosine similarity across all same/different-identity pairs from `known_faces/` (+ brightness/dim/blur augmentations); threshold 0.55 | **ROC AUC = 1.000**, Accuracy = 100%, Precision/Recall/F1 = 100%/100%/1.000 (40 same-identity pairs, 80 different-identity pairs, 0 false positives/negatives) |
| **Clothing-colour matcher** | Top-1 classification of synthetic HSV swatches (3 per colour family, with jitter) across all 13 families, using the exact `match_clothing_color()` function the live system calls | **82.1%** top-1 over all 13 families; **93.3%** on the 5 colours actually used operationally (brown, pink, teal, blue, black) |
| **Glasses detector** | Edge-ratio confidence on labelled reference faces (glasses/no-glasses), thresholded at 0.55 | **100%** accuracy, Precision/Recall/F1 = 100%/100%/1.000 (12 labelled samples) |
| **Overall (weighted)** | `0.50·face_acc + 0.40·colour_acc + 0.10·glasses_acc` | **92.8%** |

Where the errors come from: colour misclassifications are concentrated between visually adjacent families (a very dark brown swatch reading as black; purple reading as pink) — exactly the kind of confusion a human would also make from a small swatch, and it doesn't affect the colours this project's demo actually uses.

`accuracy.py` is fully reproducible (`python accuracy.py`) — it regenerates `ACCURACY.md` and 5 charts (overall bar chart, FaceNet ROC curve, FaceNet similarity distribution histogram, colour confusion matrix, glasses confusion matrix) and never touches the live `alerts.log`.

---

## 7. Testing

`test_pipeline.py` is a standalone functional test suite (18 checks, run with `python test_pipeline.py`) covering: solid-colour and striped-pattern colour matching, wrong-colour rejection, CLAHE brightening, glasses-score validity/complementarity, IoU tracking + alert gating, stale-track cleanup, `MatchResult` serialization, the shirt-primary colour pre-filter (including the specific edge cases: wrong shirt rejects even with right pants; right shirt passes even with wrong/hidden pants), the family-mismatch cap, and the glasses tie-break filter's three behaviours (hard-reject when ambiguous and discriminating, soft-penalty when it's the lone candidate, and full bypass when the target's glasses status is unknown). It uses synthetic images built with NumPy rather than real footage, so it runs anywhere with no data dependency and never writes to `alerts.log`.

---

## 8. Known Limitations (be upfront about these — evaluators respect honesty here)

- **CPU-only by design** — no GPU acceleration; scales down (resolution/frame-skip/YOLO size) as camera count rises, which is a deliberate accuracy-for-throughput tradeoff at 3–4 simultaneous feeds.
- **HSV colour ranges and the glasses pivot are hand-tuned** on this project's footage/lighting; a materially different lighting setup may need a small manual adjustment in `config.py` rather than being fully self-calibrating.
- **The colour matcher's accuracy numbers are measured on synthetic HSV swatches**, not on messy real-world clothing photos with fabric texture, folds, and mixed lighting — the 93.3% figure on operationally-used colours is the most representative number, but it's still a controlled evaluation, not an in-the-wild benchmark.
- **The IoU tracker has no re-identification** across a full occlusion/exit-reentry — if a person leaves the frame and comes back, they become a new track and must re-accumulate the 3 consecutive frames before alerting again. This is a reasonable trade for its simplicity/speed but is a real limitation.
- **Single reference photo per identity** (one-shot recognition) — no enrollment of multiple photos per person to average out pose/lighting variance, though FaceNet's embedding space is robust enough that this wasn't needed to hit the measured accuracy.

---

## 9. One-paragraph summary if asked to explain the whole project in 60 seconds

"It's a real-time, CPU-only surveillance system that finds a described person across up to four camera feeds. YOLOv8 detects people; a cheap HSV colour check on their shirt (with pants as a bonus signal) filters out almost everyone before the expensive step; only colour survivors get checked with MTCNN+FaceNet for face recognition, with glasses used as a tie-breaker between visually identical people. Because a confirmable face is rare at CCTV distance, the scoring is clothing-primary — a face, when detected and confident, can only raise the score, never lower it. An IoU tracker smooths detections across frames and gates alerts on consistency and a cooldown, and every alert is logged as JSON plus a saved snapshot. On our accuracy evaluation the fused pipeline scores 92.8%, driven by essentially perfect face recognition (ROC AUC 1.000) and a colour matcher that's over 93% accurate on the colours we actually use."
