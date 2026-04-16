"""Ethnicity classifier that predicts the 5 UTKFace classes.

This module provides two back-ends that share a common interface:

1. **CNNClassifier** - loads a trained Keras model (produced by
   ``scripts/train_ethnicity.py``) and predicts ethnicity probabilities from
   a 48x48 grayscale aligned face crop.

2. **HeuristicClassifier** - a deterministic fallback that derives a rough
   probability vector from skin-tone and landmark geometry so the application
   is useful out-of-the-box, before any training has been done.

``load_classifier`` picks the CNN if a model file exists under ``models/``,
otherwise falls back to the heuristic - both return a ``dict`` of
``{macro_name: probability}`` via the ``predict`` method.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # OpenCV is a required dep
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python is required. `pip install opencv-python-headless`."
    ) from exc

from .countries import CLASS_NAMES
from .landmarks import FaceResult


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "ethnicity_cnn.keras",
)


# --- Base class ------------------------------------------------------------


class Classifier:
    """Abstract classifier interface."""

    name: str = "base"

    def predict(self, image_rgb: np.ndarray, face: FaceResult) -> Dict[str, float]:
        raise NotImplementedError


# --- Face-crop utility -----------------------------------------------------


def _align_face_crop(image_rgb: np.ndarray, face: FaceResult, size: int = 48) -> np.ndarray:
    """Extract, align, and resize the face to ``size x size`` grayscale."""
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = face.bbox
    # Add 15% padding so we don't clip ears/forehead.
    pad = int(0.15 * max(bw, bh))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    crop = image_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = image_rgb
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Histogram equalization helps with lighting variation.
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized


# --- CNN back-end ----------------------------------------------------------


class CNNClassifier(Classifier):
    name = "cnn"

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        try:
            from tensorflow import keras  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tensorflow is required to use the CNN backend. "
                "Install with `pip install tensorflow`."
            ) from exc
        self._model = keras.models.load_model(model_path)

    def predict(self, image_rgb: np.ndarray, face: FaceResult) -> Dict[str, float]:
        x = _align_face_crop(image_rgb, face, size=48).astype(np.float32) / 255.0
        x = x.reshape(1, 48, 48, 1)
        probs = self._model.predict(x, verbose=0)[0]
        return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


# --- Heuristic fallback ----------------------------------------------------


def _skin_sample(image_rgb: np.ndarray, face: FaceResult) -> np.ndarray:
    """Sample the mean skin color from the cheek and forehead regions."""
    pts = face.landmarks[:, :2].astype(np.int32)
    h, w = image_rgb.shape[:2]

    # Indices for left cheek, right cheek, forehead regions
    sample_idxs = [50, 205, 425, 280, 10, 151]
    samples = []
    for idx in sample_idxs:
        x, y = pts[idx]
        x = int(np.clip(x, 2, w - 3))
        y = int(np.clip(y, 2, h - 3))
        patch = image_rgb[y - 2:y + 3, x - 2:x + 3]
        if patch.size:
            samples.append(patch.reshape(-1, 3).mean(axis=0))
    if not samples:
        return np.array([128, 110, 100], dtype=np.float32)
    return np.mean(samples, axis=0).astype(np.float32)


@dataclass
class HeuristicClassifier(Classifier):
    """Skin-tone + geometry heuristic.

    Not a substitute for the trained CNN, but returns sensible probability
    vectors so the app is usable immediately. The logic:

    * Luminance (Y from YCrCb) drives the split between light, medium and
      dark skin groups.
    * Redness (Cr) and yellowness (Cb) tilt between East-Asian and Indian.
    * Nose-width / face-width geometric ratio nudges African vs. European.

    Probabilities are softmaxed so they always sum to 1.
    """

    name: str = "heuristic"

    def predict(self, image_rgb: np.ndarray, face: FaceResult) -> Dict[str, float]:
        rgb = _skin_sample(image_rgb, face)
        # Convert to YCrCb for illumination-invariant skin analysis.
        ycrcb = cv2.cvtColor(rgb.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        y, cr, cb = [float(v) for v in ycrcb.reshape(3)]

        # Nose-to-face width ratio - empirically: wider nose relative to face
        # is more common in African features; narrower in East-Asian features.
        nose_w = np.linalg.norm(face.pt2("nose_left") - face.pt2("nose_right"))
        face_w = max(1.0, np.linalg.norm(face.pt2("jaw_left") - face.pt2("jaw_right")))
        nose_ratio = nose_w / face_w  # ~0.18 to ~0.32

        # Eye-corner inner angle - proxy for epicanthic fold region.
        le_out = face.pt2("left_eye_outer")
        le_in = face.pt2("left_eye_inner")
        eye_tilt = (le_in[1] - le_out[1]) / (abs(le_in[0] - le_out[0]) + 1e-6)

        # Scores per class - raw logits we then softmax.
        # Normalized luminance in [0, 1]
        y_norm = y / 255.0

        # White: lighter skin, moderate cr/cb, narrow nose
        s_white = (1.0 - abs(y_norm - 0.68)) * 2.0 + (0.30 - nose_ratio) * 3.0
        # Black: darker skin, wider nose
        s_black = (1.0 - abs(y_norm - 0.35)) * 2.5 + (nose_ratio - 0.22) * 4.0
        # East Asian: mid-tone, slight eye tilt, narrower nose
        s_asian = (1.0 - abs(y_norm - 0.62)) * 2.0 + max(eye_tilt, 0) * 4.0 + (0.25 - nose_ratio) * 2.0
        # Indian: mid-tone with higher cr (warmer), moderate nose
        s_indian = (1.0 - abs(y_norm - 0.52)) * 2.5 + (cr - 138) / 20.0
        # Other: residual catch-all, centered on mid values
        s_other = (1.0 - abs(y_norm - 0.55)) * 1.0

        logits = np.array([s_white, s_black, s_asian, s_indian, s_other], dtype=np.float32)
        # Temperature softmax - moderate sharpness
        logits = logits - logits.max()
        probs = np.exp(logits * 1.5)
        probs = probs / probs.sum()
        return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


# --- Loader ---------------------------------------------------------------


def load_classifier(model_path: Optional[str] = None) -> Classifier:
    """Load the CNN model if available, otherwise the heuristic classifier."""
    path = model_path or DEFAULT_MODEL_PATH
    if os.path.exists(path):
        try:
            return CNNClassifier(path)
        except Exception as exc:  # pragma: no cover
            print(f"[facescanner] Failed to load CNN ({exc}); using heuristic.")
    return HeuristicClassifier()
