"""Face engine — MTCNN face detection + FaceNet embeddings, shared across camera threads."""

import cv2
import numpy as np
import threading

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine

from .config import (
    DEVICE,
    MIN_CROP_HEIGHT_FOR_FACE,
    FACE_UPSCALE_FACTOR,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    CLAHE_FOR_FACES,
)

_face_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)


class FaceEngine:
    def __init__(self):
        # One lock guards MTCNN AND FaceNet — neither is thread-safe, and on CPU
        # concurrent calls just thrash the cores (3-4 cams get a 2nd engine).
        self._lock = threading.Lock()
        self.mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=20,
            thresholds=[0.6, 0.7, 0.8], keep_all=True,
            device=DEVICE, post_process=False,
        )
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    def extract_embedding_from_photo(self, bgr_image):
        # Used at setup to embed the reference photo. Returns (embedding|None, confidence).
        face_data = self._detect_best_face(bgr_image)
        if face_data is None:
            return None, 0.0
        face_crop, confidence, _ = face_data
        return self._compute_embedding(face_crop), confidence

    def detect_and_compare(self, person_crop, known_embedding):
        # Detect a face in the crop and cosine-compare it to the reference.
        result = {"face_score": 0.0, "face_crop": None,
                  "face_confidence": 0.0, "landmarks": None}
        if person_crop is None or known_embedding is None:
            return result

        crop_for_face = self._upscale_if_needed(person_crop)

        with self._lock:                       # lock covers MTCNN + FaceNet
            face_data = self._detect_best_face(crop_for_face)
            if face_data is None:
                return result
            face_crop, confidence, landmarks = face_data
            if confidence < 0.70:
                return result
            candidate_emb = self._compute_embedding(face_crop)

        result["face_crop"] = face_crop
        result["face_confidence"] = confidence
        result["landmarks"] = landmarks
        if candidate_emb is None:
            return result

        similarity = 1.0 - cosine(candidate_emb, known_embedding)
        result["face_score"] = max(0.0, float(similarity))
        return result

    def _upscale_if_needed(self, crop):
        # Tiny CCTV crops give faces too small for MTCNN; upscale them first.
        h, w = crop.shape[:2]
        if h < MIN_CROP_HEIGHT_FOR_FACE:
            new_h, new_w = int(h * FACE_UPSCALE_FACTOR), int(w * FACE_UPSCALE_FACTOR)
            return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return crop

    def _detect_best_face(self, bgr_image):
        # Return the LARGEST confident face (a bystander's face is usually smaller).
        try:
            h, w = bgr_image.shape[:2]
            if h < 20 or w < 20:
                return None

            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)
            if boxes is None or len(boxes) == 0:
                return None

            confident = [i for i in range(len(probs))
                         if probs[i] is not None and probs[i] >= 0.5]
            if not confident:
                return None

            def _box_area(i):
                bx1, by1, bx2, by2 = boxes[i]
                return max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

            best_idx = max(confident, key=_box_area)
            x1, y1, x2, y2 = boxes[best_idx].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 12 or (y2 - y1) < 12:
                return None

            face_crop = bgr_image[y1:y2, x1:x2]
            face_landmarks = (landmarks[best_idx]
                              if landmarks is not None and len(landmarks) > best_idx
                              else None)
            return face_crop, float(probs[best_idx]), face_landmarks
        except Exception:
            return None

    def _compute_embedding(self, face_crop):
        # Resize to 160, optional CLAHE, BGR→RGB, normalise to [-1,1], run FaceNet.
        try:
            face = cv2.resize(face_crop, (160, 160))
            if CLAHE_FOR_FACES:
                lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_ch = _face_clahe.apply(l_ch)
                face = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
            tensor = (tensor - 127.5) / 128.0
            tensor = tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                embedding = self.facenet(tensor).cpu().numpy()[0]
            return embedding
        except Exception:
            return None
