"""Entry point — registers a target, runs 1-4 camera threads, displays + alerts.

Pipeline per camera: YOLOv8 (person) -> colour pre-filter -> glasses -> FaceNet.
Camera workers push annotated frames to a queue; the main thread does the GUI.
"""

import cv2
import gc
import os
import signal
import sys
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from typing import Optional

import numpy as np
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

from .config import (
    PROCESS_WIDTH, PROCESS_HEIGHT, SKIP_FRAMES, DEVICE,
    ADAPTIVE_SKIP, ADAPTIVE_RESOLUTIONS, DISPLAY_QUEUE_BACKPRESSURE,
    ADAPTIVE_YOLO_IMGSZ, YOLO_MAX_DETECTIONS,
    YOLO_CONFIDENCE, COMBINED_THRESHOLD, COLOR_MATCH_THRESHOLD,
    FACE_SIMILARITY_THRESHOLD, CROWD_THRESHOLD, CROWD_MAX_FACE_CHECKS,
    MAX_FACE_CHECKS_PER_FRAME, TRACK_MAX_AGE_SEC,
    KNOWN_FACES_FOLDER, ALERTS_FOLDER, LOG_FILE, YOLO_MODEL_PATH,
    VALID_COLORS, COLOR_FAMILIES,
    DEBUG_SCORES,
)
from .face_engine import FaceEngine
from .matching_engine import (
    color_prefilter, full_match, apply_glasses_filter, compute_overall_score,
)
from .tracker import PersonTracker
from .alert_manager import AlertManager


class MissingPerson:
    """The target: name, face embedding, shirt/pants colour, glasses (True/False/None)."""

    def __init__(self, name):
        self.name = name
        self.face_embedding = None
        self.shirt_color = ""
        self.pants_color = ""
        self.wearing_glasses: Optional[bool] = None


# imshow() must run on the main thread, so workers hand frames over this queue.
display_queue  = Queue(maxsize=12)
shutdown_event = threading.Event()
# Per-camera count of queued frames so each worker applies backpressure to its OWN backlog.
_pending_frames      = {}
_pending_frames_lock = threading.Lock()


def camera_worker(camera_id, camera_name, target, face_engine, alert_manager,
                  process_w, process_h, skip_frames, yolo_imgsz=640):
    # One thread per camera: own YOLO + tracker, a shared (locked) FaceEngine.
    print(f"  [{camera_name}] Starting at {process_w}x{process_h}, "
          f"detection every {skip_frames} frames...")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"  [{camera_name}] ERROR: Cannot open source: {camera_id}")
        return

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [{camera_name}] Source: {src_w}x{src_h} @ {src_fps:.0f}fps"
          f" ({total} frames)" if total > 0 else
          f"  [{camera_name}] Source: {src_w}x{src_h} @ {src_fps:.0f}fps (live)")

    yolo         = YOLO(YOLO_MODEL_PATH, verbose=False)
    tracker      = PersonTracker()
    frame_count  = 0
    local_alerts = 0
    last_boxes = []                         # (bbox, label, colour, thickness) drawn every frame

    is_live_stream = isinstance(camera_id, str) and camera_id.lower().startswith(
        ("rtsp://", "http://", "https://")
    )

    try:
        while not shutdown_event.is_set():
            # Backpressure: for files, skip-decode (cap.grab) if we're ahead of the GUI.
            if not is_live_stream:
                while True:
                    with _pending_frames_lock:
                        pending = _pending_frames.get(camera_name, 0)
                    if pending <= DISPLAY_QUEUE_BACKPRESSURE:
                        break
                    if not cap.grab():
                        break
                    frame_count += 1

            ret, frame = cap.read()
            if not ret:
                for _ in range(2):          # small retry on transient FFmpeg hiccups
                    time.sleep(0.01)
                    ret, frame = cap.read()
                    if ret:
                        break

            if not ret:                     # end of video / stream dropout
                if is_live_stream:
                    print(f"  [{camera_name}] Stream lost. Reconnecting in 3s...")
                    cap.release()
                    time.sleep(3)
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        print(f"  [{camera_name}] Reconnect failed. Retrying...")
                        time.sleep(5)
                        cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        print(f"  [{camera_name}] Cannot reconnect. Stopping.")
                        break
                    print(f"  [{camera_name}] Reconnected!")
                    continue
                else:                       # video file → loop from the start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"  [{camera_name}] Video unreadable. Stopping.")
                        break
                    frame_count = 0
                    tracker = PersonTracker()
                    last_boxes = []

            frame_count += 1

            # Keep the full-res frame for crops (faces need the source pixels);
            # YOLO runs on the downscaled frame for speed.
            orig_frame = frame
            orig_h, orig_w = orig_frame.shape[:2]
            frame   = cv2.resize(frame, (process_w, process_h))
            display = frame.copy()
            scale_x = orig_w / float(process_w)
            scale_y = orig_h / float(process_h)

            if frame_count % skip_frames == 0:
                last_boxes = []
                tracker.cleanup_stale(max_age_sec=TRACK_MAX_AGE_SEC)

                results = yolo(frame, verbose=False, conf=YOLO_CONFIDENCE,
                               classes=[0], imgsz=yolo_imgsz, max_det=YOLO_MAX_DETECTIONS)
                if shutdown_event.is_set():
                    break

                boxes = results[0].boxes
                person_count = len(boxes) if boxes is not None else 0

                crowd_mode = person_count >= CROWD_THRESHOLD
                if crowd_mode:
                    print(f"  [{camera_name}] CROWD MODE: {person_count} persons "
                          f"detected — face checks capped at "
                          f"{CROWD_MAX_FACE_CHECKS}/frame")

                # PHASE A — colour pre-filter every box (no tracker writes yet).
                color_fail_boxes = []
                passed = []
                if boxes is not None:
                    for box in boxes:
                      try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(process_w, x2)
                        y2 = min(process_h, y2)

                        # Crop from the full-res source (faces need the pixels).
                        ox1 = max(0, int(x1 * scale_x))
                        oy1 = max(0, int(y1 * scale_y))
                        ox2 = min(orig_w,  int(x2 * scale_x))
                        oy2 = min(orig_h,  int(y2 * scale_y))
                        person_crop = orig_frame[oy1:oy2, ox1:ox2]
                        if person_crop.size == 0:
                            continue

                        bbox = (x1, y1, x2, y2)   # process coords (tracker + log)
                        color_result = color_prefilter(
                            person_crop, target.shirt_color, target.pants_color
                        )
                        if not color_result.passed_color_filter:
                            color_fail_boxes.append(bbox)
                            continue

                        passed.append({
                            "bbox": bbox,
                            "person_crop": person_crop,
                            "color_result": color_result,
                            "area": (ox2 - ox1) * (oy2 - oy1),
                        })
                      except Exception as box_err:
                        print(f"  [{camera_name}] skipped one detection: {box_err}")
                        continue

                # PHASE B — face-check survivors, capped per frame (cache hits are free).
                passed.sort(key=lambda c: c["area"], reverse=True)
                face_budget = (CROWD_MAX_FACE_CHECKS if crowd_mode
                               else MAX_FACE_CHECKS_PER_FRAME)
                face_checks_used = 0

                frame_candidates = []
                for cand in passed:
                    bbox         = cand["bbox"]
                    person_crop  = cand["person_crop"]
                    color_result = cand["color_result"]

                    existing_id = tracker.find_track_id(bbox)
                    existing_track = (tracker.get_track(existing_id)
                                      if existing_id is not None else None)

                    if existing_track and not existing_track.needs_face_refresh(bbox):
                        color_result.face_score      = existing_track.cached_face_score or 0.0
                        color_result.face_confidence = existing_track.cached_face_conf or 0.0
                        color_result.face_detected   = bool(existing_track.cached_face_detected)
                        color_result.glasses_score      = existing_track.cached_glasses_score or 0.5
                        color_result.glasses_confidence = existing_track.cached_glasses_conf or 0.5
                        color_result.overall = compute_overall_score(color_result)
                        result = color_result
                        fresh_face_data = None
                    elif face_checks_used < face_budget:
                        result = full_match(person_crop, face_engine,
                                            target.face_embedding, color_result,
                                            target.wearing_glasses)
                        face_checks_used += 1
                        fresh_face_data = {
                            "face_score":     result.face_score,
                            "face_conf":      result.face_confidence,
                            "face_detected":  result.face_detected,
                            "glasses_score":  result.glasses_score,
                            "glasses_conf":   result.glasses_confidence,
                        }
                    else:
                        # Budget spent → clothing-only score this frame; face deferred.
                        color_result.overall = compute_overall_score(color_result)
                        result = color_result
                        fresh_face_data = None

                    frame_candidates.append({
                        "bbox": bbox,
                        "result": result,
                        "person_crop": person_crop,
                        "fresh_face_data": fresh_face_data,
                    })

                    if DEBUG_SCORES:
                        print(f"  [{camera_name} f{frame_count}] bbox={bbox} "
                              f"shirt={result.shirt_score:.2f}/{result.shirt_detected} "
                              f"pants={result.pants_score:.2f}/{result.pants_detected} "
                              f"face={result.face_score:.2f} (det={result.face_detected}) "
                              f"glasses={result.glasses_confidence:.2f} "
                              f"overall={result.overall:.2f}")

                # Glasses tie-break (drops clear mismatches when ≥2 share clothing).
                rejected_count = len(frame_candidates)
                frame_candidates = apply_glasses_filter(frame_candidates, target.wearing_glasses)
                rejected_count -= len(frame_candidates)
                if rejected_count > 0:
                    print(f"  [{camera_name}] Glasses filter dropped "
                          f"{rejected_count} ambiguous-clothing candidate(s)")

                # PASS 2 — all tracker writes + alert decisions live here.
                for bbox in color_fail_boxes:
                    last_boxes.append((bbox, "", (0, 160, 0), 1))
                    tracker.update(bbox, 0.0)

                for cand in frame_candidates:
                    bbox        = cand["bbox"]
                    result      = cand["result"]
                    person_crop = cand["person_crop"]
                    fresh_face_data = cand["fresh_face_data"]

                    track_id = tracker.update(bbox, result.overall,
                                              snapshot=person_crop, face_data=fresh_face_data)
                    face_confirmed = (result.face_detected
                                      and result.face_score >= FACE_SIMILARITY_THRESHOLD)

                    if tracker.should_alert(track_id):
                        track = tracker.get_track(track_id)
                        local_alerts += 1
                        tracker.mark_alerted(track_id)
                        snapshot = (track.best_snapshot
                                    if track.best_snapshot is not None else person_crop)
                        match_type = "face" if face_confirmed else "clothing"

                        # Save the evidence frame full-res, so map bbox back to source coords.
                        bx1, by1, bx2, by2 = bbox
                        orig_bbox = (max(0, int(bx1 * scale_x)),
                                     max(0, int(by1 * scale_y)),
                                     min(orig_w, int(bx2 * scale_x)),
                                     min(orig_h, int(by2 * scale_y)))

                        alert_manager.trigger_alert(
                            camera_name=camera_name, target_name=target.name,
                            track=track, match_result=result, person_crop=snapshot,
                            full_frame=orig_frame, bbox=orig_bbox,
                            frame_number=frame_count, match_type=match_type,
                        )

                        if face_confirmed:
                            label = (f"FOUND: {target.name.upper()} "
                                     f"[FACE {result.face_score:.0%}] ({track.avg_score:.0%})")
                            last_boxes.append((bbox, label, (0, 0, 255), 3))
                        else:
                            label = f"CLOTHING MATCH: {target.name.upper()} ({track.avg_score:.0%})"
                            last_boxes.append((bbox, label, (0, 140, 255), 3))

                    elif result.face_detected and result.face_score >= FACE_SIMILARITY_THRESHOLD:
                        label = f"Face {result.face_score:.0%} — verifying ({result.overall:.0%})"
                        last_boxes.append((bbox, label, (0, 200, 100), 2))
                    elif result.overall > 0.35:
                        face_info = (f"face {result.face_score:.0%}"
                                     if result.face_detected else "no face")
                        label = f"Checking... {result.overall:.0%} ({face_info})"
                        last_boxes.append((bbox, label, (0, 220, 255), 2))
                    else:
                        last_boxes.append((bbox, "", (180, 180, 0), 1))

            # Draw persistent boxes every frame (no flicker between processed frames).
            for bbox, label, color, thickness in last_boxes:
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(display, (bx1, by1), (bx2, by2), color, thickness)
                if label:
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                    cv2.rectangle(display, (bx1, by1 - th - 10), (bx1 + tw + 6, by1), color, -1)
                    cv2.putText(display, label, (bx1 + 3, by1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            hud = f"{camera_name} | Frame:{frame_count} | Alerts:{local_alerts}"
            cv2.putText(display, hud, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            try:
                display_queue.put_nowait((camera_name, display))
                with _pending_frames_lock:
                    _pending_frames[camera_name] = _pending_frames.get(camera_name, 0) + 1
            except Exception:
                pass                        # drop frame if queue is full

            if frame_count % 300 == 0:
                gc.collect()                # release model intermediate buffers

    except Exception as e:
        print(f"  [{camera_name}] Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        try:
            cv2.destroyWindow(camera_name)
        except Exception:
            pass
        print(f"  [{camera_name}] Stopped. Total alerts: {local_alerts}")


def load_missing_person(face_engine):
    # Interactive setup: pick a reference photo, embed it, describe the clothing.
    print()
    print("  " + "=" * 58)
    print("   MISSING PERSON REGISTRATION")
    print("  " + "=" * 58)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)

    available = set()
    for f in os.listdir(KNOWN_FACES_FOLDER):
        name, ext = os.path.splitext(f)
        if ext.lower() in extensions:
            available.add(name)

    if available:
        print(f"\n  Reference photos in '{KNOWN_FACES_FOLDER}/':")
        for name in sorted(available):
            print(f"    - {name}")
    else:
        print(f"\n  No photos found in '{KNOWN_FACES_FOLDER}/'.")
        print(f"  Please add a photo: {KNOWN_FACES_FOLDER}/PersonName.jpg")

    print()
    person_name = input("  Enter the name of the missing person: ").strip()
    if not person_name:
        return None

    photo_path = None
    for ext in extensions:
        candidate = os.path.join(KNOWN_FACES_FOLDER, f"{person_name}{ext}")
        if os.path.exists(candidate):
            photo_path = candidate
            break
    if not photo_path:                      # case-insensitive fallback
        for f in os.listdir(KNOWN_FACES_FOLDER):
            name, ext = os.path.splitext(f)
            if name.lower() == person_name.lower() and ext.lower() in extensions:
                photo_path = os.path.join(KNOWN_FACES_FOLDER, f)
                break

    if not photo_path:
        print(f"\n  ERROR: No photo found for '{person_name}'.")
        print(f"  Save a clear frontal photo as: {KNOWN_FACES_FOLDER}/{person_name}.jpg")
        return None

    img = cv2.imread(photo_path)
    if img is None:
        print(f"  ERROR: Could not read image file: {photo_path}")
        return None
    print(f"  Photo loaded: {photo_path}")

    person = MissingPerson(person_name)
    print("  Extracting face embedding...")
    embedding, confidence = face_engine.extract_embedding_from_photo(img)
    if embedding is not None:
        person.face_embedding = embedding
        print(f"  Face extracted successfully (confidence: {confidence:.2f})")
    else:
        print("  WARNING: No face detected in the reference photo!")
        print("  The system will rely on clothing description only.")
        print("  For best results, use a clear frontal photo with good lighting.")

    print()
    print("  " + "-" * 58)
    print("   DESCRIBE CURRENT APPEARANCE")
    print("  " + "-" * 58)
    print(f"  Available colors: {', '.join(VALID_COLORS)}")
    print(f"  (Each color matches all its shades — 'blue' covers sky,")
    print(f"   navy, royal, light blue; 'green' covers light, dark,")
    print(f"   forest, lime. Patterns/stripes work automatically.)")
    print()

    while True:
        shirt = input("  Shirt/top color: ").strip().lower()
        if shirt == "":
            print("  (skipped — no shirt color filter)")
            break
        if shirt in VALID_COLORS:
            person.shirt_color = shirt
            break
        print(f"  '{shirt}' not recognized. Choose from: {', '.join(VALID_COLORS)}")

    while True:
        pants = input("  Pants/bottom color: ").strip().lower()
        if pants == "":
            print("  (skipped — no pants color filter)")
            break
        if pants in VALID_COLORS:
            person.pants_color = pants
            break
        print(f"  '{pants}' not recognized. Choose from: {', '.join(VALID_COLORS)}")

    # Glasses is tri-state; "unknown" turns the disambiguation filter off.
    glasses = input("  Wearing glasses? (yes/no/unknown): ").strip().lower()
    if glasses in ("yes", "y"):
        person.wearing_glasses = True
    elif glasses in ("no", "n"):
        person.wearing_glasses = False
    else:
        person.wearing_glasses = None

    print()
    print("  " + "=" * 58)
    print("   TARGET SUMMARY")
    print("  " + "=" * 58)
    print(f"  Name:     {person.name}")
    print(f"  Shirt:    {person.shirt_color or '(any color)'}")
    print(f"  Pants:    {person.pants_color or '(any color)'}")
    if person.wearing_glasses is None:
        glasses_str = "(unknown — filter off)"
    else:
        glasses_str = "Yes" if person.wearing_glasses else "No"
    print(f"  Glasses:  {glasses_str}")
    print(f"  Face:     {'Loaded (' + f'{confidence:.0%} conf)' if person.face_embedding is not None else 'NOT AVAILABLE'}")
    print("  " + "=" * 58)
    return person


def configure_cameras():
    # Ask for the video files (or RTSP/HTTP stream URLs) to watch, up to 4.
    print()
    print("  " + "=" * 58)
    print("   CAMERA CONFIGURATION")
    print("  " + "=" * 58)
    print("  Enter a video file path (or RTSP/HTTP stream URL) for each camera.")
    print("  Leave blank and press Enter to finish.")
    print()

    cameras = {}
    for i in range(4):
        cam_input = input(f"  Camera {i + 1}: ").strip()
        if not cam_input:
            if i == 0:
                print("  At least one video is required. Try again.")
                cam_input = input(f"  Camera {i + 1}: ").strip()
                if not cam_input:
                    break
            else:
                break

        if cam_input.lower().startswith(("rtsp://", "http://", "https://")):
            label = f"IP_CAM_{i + 1}"
            cameras[cam_input] = label
            print(f"    -> Added IP camera: {label} ({cam_input})")
        elif os.path.isfile(cam_input):
            label = f"VIDEO_{i + 1}"
            cameras[cam_input] = label
            print(f"    -> Added video: {label} ({cam_input})")
        else:
            print(f"    -> Not found: {cam_input} — skipped")

    return cameras


def main():
    print()
    print("=" * 64)
    print("   AI-POWERED MISSING PERSON SURVEILLANCE SYSTEM")
    print("=" * 64)
    print("   Mode:     CPU-Only (Intel Iris Xe)")
    print("   Pipeline: YOLOv8 -> Color Filter -> Glasses -> FaceNet")
    print("   Cameras:  Up to 4 simultaneous feeds")
    print("=" * 64)

    print()
    print("  Loading AI models (this may take 15-30 seconds)...")
    print()

    try:
        yolo_test = YOLO(YOLO_MODEL_PATH, verbose=False)
        print("  [OK] YOLOv8-Nano loaded (person detection)")
        del yolo_test
    except Exception as e:
        print(f"  FATAL: Cannot load YOLOv8: {e}")
        print("  Make sure ultralytics is installed: pip install ultralytics")
        return

    try:
        face_engine = FaceEngine()
        print("  [OK] MTCNN + FaceNet loaded (face recognition + landmarks)")
    except Exception as e:
        print(f"  FATAL: Cannot load face models: {e}")
        print("  Make sure facenet-pytorch is installed: pip install facenet-pytorch")
        return

    alert_manager = AlertManager()
    print("  [OK] Alert manager ready (snapshots + JSON logging)")
    print()
    print("  All models loaded successfully.")

    target = load_missing_person(face_engine)
    if target is None:
        print("\n  No target registered. Exiting.")
        return

    cameras = configure_cameras()
    if not cameras:
        print("  No cameras configured. Exiting.")
        return

    print()
    print("=" * 64)
    print("   SURVEILLANCE ACTIVE")
    print("=" * 64)
    print(f"  Target:     {target.name}")
    print(f"  Shirt:      {target.shirt_color or '(any)'}")
    print(f"  Pants:      {target.pants_color or '(any)'}")
    print(f"  Glasses:    {('Yes' if target.wearing_glasses else 'No') if target.wearing_glasses is not None else 'Unknown (filter off)'}")
    print(f"  Face:       {'Loaded' if target.face_embedding is not None else 'N/A'}")
    print(f"  Cameras:    {len(cameras)}")
    for cid, cname in cameras.items():
        print(f"    - {cname} -> {cid}")
    print()
    print("  Press 'Q' in any camera window to stop surveillance.")
    print(f"  Snapshots will be saved to: {ALERTS_FOLDER}")
    print("=" * 64)
    print()

    n_cams = len(cameras)
    pw, ph = ADAPTIVE_RESOLUTIONS.get(n_cams, (PROCESS_WIDTH, PROCESS_HEIGHT))
    skip   = ADAPTIVE_SKIP.get(n_cams, SKIP_FRAMES)
    yolo_imgsz = ADAPTIVE_YOLO_IMGSZ.get(n_cams, 640)

    # Cap intra-op threads so concurrent model calls across cameras don't thrash the cores.
    try:
        import torch
        n_threads = max(2, (os.cpu_count() or 4) // 2) if n_cams <= 1 else 3
        torch.set_num_threads(n_threads)
        print(f"  CPU math threads per inference call: {n_threads}")
    except Exception as _e:
        print(f"  [WARN] Could not set torch threads: {_e}")

    # 3+ cameras get a 2nd FaceEngine to halve lock contention.
    if n_cams >= 3:
        try:
            face_engine_b = FaceEngine()
            print(f"  [OK] Loaded 2nd FaceEngine to halve lock contention "
                  f"(camera count: {n_cams})")
            face_engines = [face_engine, face_engine_b]
        except Exception as e:
            print(f"  [WARN] Could not load 2nd FaceEngine ({e}); "
                  f"continuing with single engine")
            face_engines = [face_engine]
    else:
        face_engines = [face_engine]

    print(f"  Processing: {pw}x{ph}, detection every {skip} frame(s), "
          f"face engines: {len(face_engines)}")

    threads = []
    for i, (cam_id, cam_name) in enumerate(cameras.items()):
        engine_for_this_cam = face_engines[i % len(face_engines)]
        t = threading.Thread(
            target=camera_worker,
            args=(cam_id, cam_name, target, engine_for_this_cam, alert_manager,
                  pw, ph, skip, yolo_imgsz),
            name=f"cam-{cam_name}",
            daemon=True,                    # so Q exits even if a worker is stuck in a model call
        )
        threads.append(t)
        t.start()

    # Main thread owns the GUI (imshow/waitKey).
    try:
        while not shutdown_event.is_set():
            try:
                cam_name, frame = display_queue.get(timeout=0.1)
                with _pending_frames_lock:
                    _pending_frames[cam_name] = max(0, _pending_frames.get(cam_name, 0) - 1)
                cv2.imshow(cam_name, frame)
            except Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("\n  Shutdown requested by user (Q pressed)...")
                shutdown_event.set()
                break

            if all(not t.is_alive() for t in threads):
                print("\n  All camera threads have stopped.")
                break

    except KeyboardInterrupt:
        print("\n  Ctrl+C received — shutting down...")
        shutdown_event.set()

    shutdown_event.set()

    while True:                             # drain queue so workers unblock
        try:
            display_queue.get_nowait()
        except Empty:
            break

    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)

    for t in threads:
        t.join(timeout=2)                   # daemon threads won't hang the exit

    print()
    print("=" * 64)
    print(f"  System stopped cleanly.")
    print(f"  Total alerts triggered: {alert_manager.alert_count}")
    print(f"  Alert snapshots saved:  {os.path.abspath(ALERTS_FOLDER)}")
    print(f"  JSON log file:          {os.path.abspath(LOG_FILE)}")
    print("=" * 64)
    print()


def run():
    """Console entry point (`person-finder` / `python -m person_finder`)."""
    signal.signal(signal.SIGINT, lambda *_: shutdown_event.set())   # graceful Ctrl+C
    try:
        main()
    except Exception as e:
        print(f"\n  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
