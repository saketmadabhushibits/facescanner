"""Face landmark extraction using MediaPipe FaceMesh.

MediaPipe's Face Mesh model returns 468 3D landmarks per face. We expose a
small helper that accepts an image (numpy BGR/RGB array) and returns the
landmarks for the most prominent face along with the bounding box of that
face - or ``None`` if no face is detected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - helpful error at import time
    raise ImportError(
        "mediapipe is required for landmark detection. "
        "Install it with `pip install mediapipe`."
    ) from exc


# Subset of MediaPipe FaceMesh landmark indices we rely on for measurements.
# Reference: https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
LM = {
    # Eyes (outer/inner corners and pupils approximated by centroid of iris ring)
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_iris": 468,   # Requires refine_landmarks=True
    "right_iris": 473,  # Requires refine_landmarks=True

    # Brows
    "left_brow_inner": 55,
    "right_brow_inner": 285,

    # Nose
    "nose_tip": 1,
    "nose_bridge_top": 168,
    "nose_left": 129,
    "nose_right": 358,

    # Mouth
    "mouth_left": 61,
    "mouth_right": 291,
    "lip_top": 13,
    "lip_bottom": 14,

    # Face outline
    "chin": 152,
    "forehead": 10,
    "jaw_left": 234,
    "jaw_right": 454,
    "cheek_left": 93,
    "cheek_right": 323,
}


@dataclass
class FaceResult:
    """Container holding all landmarks for a detected face."""

    landmarks: np.ndarray  # shape (N, 3) in pixel coordinates (x, y, z)
    image_shape: Tuple[int, int]  # (height, width)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)

    def pt(self, name: str) -> np.ndarray:
        """Return the 3D coordinates of a named landmark."""
        return self.landmarks[LM[name]]

    def pt2(self, name: str) -> np.ndarray:
        """Return the 2D coordinates of a named landmark."""
        return self.landmarks[LM[name]][:2]


class LandmarkDetector:
    """Thin wrapper around MediaPipe FaceMesh with lazy initialization."""

    def __init__(self, refine_landmarks: bool = True):
        self._refine = refine_landmarks
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
        )

    def detect(self, image_rgb: np.ndarray) -> Optional[FaceResult]:
        """Run face mesh on an RGB image and return the first face, if any."""
        if image_rgb is None or image_rgb.size == 0:
            return None

        h, w = image_rgb.shape[:2]
        results = self._mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        coords = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in face.landmark],
            dtype=np.float32,
        )

        xs = coords[:, 0]
        ys = coords[:, 1]
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        return FaceResult(landmarks=coords, image_shape=(h, w), bbox=bbox)

    def close(self) -> None:
        self._mesh.close()
