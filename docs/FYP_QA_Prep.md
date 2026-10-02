# Evaluator Q&A Prep — Missing Person Surveillance System

Grouped by theme. Answer in your own words in the room, but the substance below is accurate to the code.

## Architecture / Design

**Q: Why a cascade instead of a single end-to-end model?**
A single deep model (e.g. a person re-identification network) would need labelled training data we don't have, would be much more expensive to run on CPU across 4 feeds, and would be a black box. The cascade (YOLO → colour → face → glasses) lets each stage be cheap, explainable, and independently tunable/testable, and only pays the expensive FaceNet cost on candidates that already survived a near-free colour check.

**Q: Why is clothing weighted more than face in some cases?**
It isn't fixed-weighted — it's `max(clothing_score, face_aware_score)`. At CCTV range/angle, a usable face is the exception; if the system required a face to alert, it would miss most real detections. Clothing alone can cross the alert threshold; a confirmed face only ever adds confidence on top, never subtracts.

**Q: What happens if two people wear identical clothing?**
The glasses detector acts as a tie-breaker: if ≥2 candidates share matching clothing in the same frame, and the glasses reading can actually discriminate between them, the one whose glasses status doesn't match the target is rejected. If only one ambiguous candidate exists, it gets a soft score penalty instead of being dropped outright, since the heuristic isn't reliable enough to be the sole rejection reason.

**Q: Why IoU tracking instead of DeepSORT/ByteTrack?**
Those trackers exist mainly to solve long-term re-identification with a learned appearance embedding — but this system already has a purpose-built re-identification signal (face + clothing matching). The tracker's only remaining job is short-term temporal smoothing and alert gating, which plain IoU handles at near-zero cost.

**Q: What's the false-positive control?**
Multiple layers: (1) YOLO confidence ≥ 0.55 filters non-person detections, (2) the colour pre-filter must pass before any face check runs, (3) an alert requires ≥3 consecutive frames tracked (not a single lucky frame), (4) the rolling-average score must clear `COMBINED_THRESHOLD` (0.52), (5) a 10-second cooldown prevents the same track from spamming repeat alerts.

## Algorithms / Math

**Q: What is FaceNet actually computing?**
It's `InceptionResnetV1` pretrained on VGGFace2, mapping a 160×160 face crop to a 512-dimensional embedding vector such that embeddings of the same identity are close together and different identities are far apart in cosine distance. Identity is decided as `1 - cosine_distance(embedding_a, embedding_b) ≥ 0.55`.

**Q: Why HSV over RGB for colour matching?**
RGB conflates brightness and colour, so the same shirt looks like different RGB values under different lighting. HSV separates Hue (colour identity) from Saturation and Value (intensity/lighting), so the hue channel alone is largely lighting-invariant — critical for CCTV footage with inconsistent exposure.

**Q: How is a colour "score" computed exactly?**
The percentage of pixels in the sampled shirt/pants region falling inside the target colour's HSV range (`cv2.inRange`, morphologically opened to drop noise) is converted to a score by fixed bands (>40%→0.95, >25%→0.85, >15%→0.70, >8%→0.50, else a weighted HSV-distance fallback for non-neutral colours). If the *dominant* colour in the region is a different family than the target, the score is capped at 0.40 regardless, to stop a minority-pixel coincidence from producing a false match.

**Q: What does the glasses detector actually measure?**
The ratio of edge strength (Sobel vertical gradient + Canny edge density) in the eye-band of the face versus the cheek-band. Glasses frames and bridges add high-frequency edges near the eyes that bare skin lacks; taking a ratio (not an absolute count) makes it robust to crop resolution/scale. It's not a trained classifier — it's a hand-tuned heuristic, deliberately, because the discrimination task is simple and it avoids needing a labelled glasses dataset.

**Q: Why cosine similarity and not Euclidean distance for face matching?**
Cosine similarity measures the angle between embeddings regardless of magnitude, which is the standard choice for FaceNet-style embeddings — the model was trained with a loss (triplet loss on VGGFace2) that shapes the embedding space for angular/cosine comparison, so it's the metric consistent with how the network was trained.

## Evaluation / Metrics

**Q: How was accuracy measured — is it inflated?**
`accuracy.py` evaluates each component with standard ML metrics (accuracy, precision, recall, F1, ROC AUC) against ground truth, not eyeballed. FaceNet: cosine similarity across every same/different-identity pair from the reference photos (with brightness/dim/blur augmentation) — 120 pairs total, threshold 0.55, ROC AUC 1.000. Colour: top-1 classification of reproducible HSV swatches per colour family — 82.1% over all 13, 93.3% on the 5 colours actually used. Glasses: 12 labelled reference faces, 100% accuracy. These are run with the *exact same functions* the live pipeline calls, so it's not a separate/idealized model.

**Q: Why is colour accuracy lower (82%) than face (100%)?**
The 13-family test intentionally includes visually adjacent colours (e.g. dark brown vs black, purple vs pink) that are genuinely ambiguous even to a human from a small swatch. The overall figure is dragged down by families the operator doesn't actually search for in this deployment; restricted to the colours actually used, accuracy is 93.3%.

**Q: What's ROC AUC and why report it for face recognition?**
Area under the ROC curve measures how well the score separates the two classes (same-identity vs different-identity) across *all* possible thresholds, not just the one chosen operationally. AUC = 1.000 means the embeddings perfectly rank every same-identity pair above every different-identity pair in this evaluation — a stronger claim than "accuracy at one threshold," which is why the operating threshold (0.55) can be conservative without missing anything in this test set.

**Q: Is 92.8% the accuracy of the whole system finding a person in a video?**
No — it's a **weighted combination of the three component-level metrics** (face recognition accuracy, colour matcher accuracy, glasses accuracy), weighted by the same fusion weights the live pipeline uses. It's a proxy for pipeline quality, not an end-to-end "percentage of missing persons found in real footage" statistic — that end-to-end number isn't reported because it isn't measured. Being upfront about this distinction, if asked, is the honest answer.

## Engineering / Systems

**Q: Why 4 camera feeds max, and why does quality degrade as you add more?**
CPU-only hardware has a finite compute budget shared across all camera threads. Resolution, frame-skip interval, and YOLO input size are deliberately reduced as camera count increases (`ADAPTIVE_*` tables in `config.py`) so the system stays real-time rather than falling behind and building unbounded latency. It's an explicit throughput/accuracy tradeoff, not an accident.

**Q: How do you avoid re-running FaceNet on every single frame?**
Per-track face-result caching: once a track has a face result, it's reused for 25 frames if positive (a real face was found) or 10 frames if negative, unless the box moves more than 40% of its own diagonal between frames (signals a probable different person or serious occlusion change), which forces an immediate re-check regardless of cache age.

**Q: What happens under a crowd (many people in frame)?**
Crowd mode triggers at ≥10 detected people: the per-frame face-check budget drops from 5 to 3, and candidates are processed largest-bounding-box-first, since closer/larger people are both more likely to be identifiable and more likely to be the actual target of interest.

**Q: Is this thread-safe? How do you handle 4 camera threads writing alerts concurrently?**
Each camera runs its own thread with its own YOLO model and tracker (no shared mutable state there). The two genuinely shared, non-thread-safe resources are the `FaceEngine` (MTCNN+FaceNet) — guarded by a lock, with a second engine loaded at 3+ cameras to reduce contention — and the `AlertManager`, which uses a single lock so alert numbering, snapshot filenames, and the JSON log line never interleave or collide across threads.

**Q: How is an alert consumed downstream?**
`alerts.log` is JSON-lines — one JSON object per line, containing camera, person, confidence, the full score breakdown, and paths to a saved crop and an annotated full-frame snapshot. It's an append-only, line-oriented, streaming-friendly format a backend/dashboard can tail (`for line in f: json.loads(line)`), rather than a single JSON array that would need to be fully rewritten on every alert.

## Limitations / Critique (be ready — evaluators probe here)

**Q: What's the weakest part of the system?**
Honestly, the clothing-colour matcher is the least sophisticated component (rule-based HSV thresholding vs. learned models for the other two), and it's also the one whose accuracy is measured on synthetic swatches rather than real cluttered clothing photos — its 93.3%-on-used-colours figure is representative but not an in-the-wild benchmark.

**Q: Could this be fooled by someone changing clothes?**
Yes — the colour gate is a strength for finding someone quickly across cameras in a short time window (clothes usually don't change in that window), but it's not designed for long-horizon tracking across a change of clothes. Face-only matching would still work in that scenario if the face is visible, since a confirmed face doesn't depend on clothing at all — but the clothing pre-filter has to be turned off or the search will miss them.

**Q: Why not use a commercial/cloud face-recognition API?**
Two reasons: (1) privacy/deployment — CCTV footage of missing-person searches is sensitive, and an on-device, offline pipeline avoids sending footage to a third party; (2) the project's constraint was CPU-only, offline, real-time processing, which an open pretrained model (FaceNet/VGGFace2) run locally satisfies without per-call cost or network dependency.

**Q: What would you do differently with more time?**
Likely candidates: a small learned re-identification embedding to survive full occlusion/re-entry (current IoU tracker treats a re-entry as a new track); a labelled real-world dataset for the colour matcher to move past synthetic swatch evaluation; multi-photo enrollment per identity to average out pose variance in the reference embedding.
