"""Beauty-ratio scoring built from geometric facial landmark measurements.

The score is a weighted composite of well-studied objective facial metrics:

1. **Bilateral symmetry** - we mirror landmarks about the vertical facial axis
   and measure the average offset (smaller = more symmetric).
2. **Golden ratio (phi ~ 1.618)** - ratio of face length to face width; ratio
   of mouth width to nose width; ratio of lower-face height to nose length.
3. **Facial thirds** - forehead, midface and lower-face should be roughly
   equal heights.
4. **Eye spacing** - inter-ocular distance should approximately equal the
   width of one eye (the "five-eyes rule").
5. **Jaw balance** - the jaw/cheek width ratio and chin alignment.
6. **Lip fullness** - upper-lip to lower-lip ratio, and mouth-to-nose width.

Each sub-score maps to [0, 100]. The final score is a weighted mean. This is a
geometric / proportion-based heuristic - "beauty" is subjective, but the
golden-ratio family of metrics is widely cited in aesthetics research and
gives a consistent, objective number per face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .landmarks import FaceResult

PHI = 1.6180339887  # Golden ratio


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ratio_score(value: float, target: float, tolerance: float = 0.35) -> float:
    """Score how close a ratio is to a target, mapped to [0, 100].

    A deviation of ``tolerance`` (as a fraction of target) yields a score of 0.
    No deviation yields 100.
    """
    if target == 0:
        return 0.0
    deviation = abs(value - target) / target
    score = 1.0 - min(deviation / tolerance, 1.0)
    return float(max(0.0, min(1.0, score)) * 100.0)


@dataclass
class BeautyBreakdown:
    symmetry: float
    golden_ratio: float
    facial_thirds: float
    eye_spacing: float
    jaw_balance: float
    lip_proportion: float
    overall: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "symmetry": round(self.symmetry, 1),
            "golden_ratio": round(self.golden_ratio, 1),
            "facial_thirds": round(self.facial_thirds, 1),
            "eye_spacing": round(self.eye_spacing, 1),
            "jaw_balance": round(self.jaw_balance, 1),
            "lip_proportion": round(self.lip_proportion, 1),
            "overall": round(self.overall, 1),
        }


# --- Individual metrics -----------------------------------------------------


def _symmetry_score(face: FaceResult) -> float:
    """Compare left/right halves about the facial midline."""
    pts = face.landmarks[:, :2]

    # Midline: average of forehead, nose bridge, nose tip, chin.
    midline_idxs = [10, 168, 1, 152]
    midline_pts = pts[midline_idxs]
    # Fit a vertical axis through these points (use mean x).
    axis_x = float(np.mean(midline_pts[:, 0]))

    # Mirror all points about this axis.
    mirrored = pts.copy()
    mirrored[:, 0] = 2 * axis_x - mirrored[:, 0]

    # Compare each point to its nearest neighbor among the mirrored set of the
    # opposite half. Using a simple nearest neighbor via kd-tree-like broadcast.
    diff = np.linalg.norm(pts[:, None, :] - mirrored[None, :, :], axis=2)
    nn = diff.min(axis=1)

    # Normalize by face width so scale doesn't matter.
    face_w = max(1.0, face.bbox[2])
    mean_err = float(np.mean(nn)) / face_w

    # Empirically, mean_err of ~0.005 is excellent, ~0.04 is poor.
    score = 1.0 - min(mean_err / 0.04, 1.0)
    return float(max(0.0, min(1.0, score)) * 100.0)


def _golden_ratio_score(face: FaceResult) -> float:
    """Average phi-adherence across three canonical golden-ratio measurements."""
    face_len = _dist(face.pt2("forehead"), face.pt2("chin"))
    face_w = _dist(face.pt2("jaw_left"), face.pt2("jaw_right"))
    if face_w == 0:
        return 0.0
    face_ratio = face_len / face_w

    mouth_w = _dist(face.pt2("mouth_left"), face.pt2("mouth_right"))
    nose_w = _dist(face.pt2("nose_left"), face.pt2("nose_right"))
    mouth_nose = mouth_w / nose_w if nose_w else 0.0

    lower_face = _dist(face.pt2("nose_tip"), face.pt2("chin"))
    nose_len = _dist(face.pt2("nose_bridge_top"), face.pt2("nose_tip"))
    lower_nose = lower_face / nose_len if nose_len else 0.0

    scores = [
        _ratio_score(face_ratio, PHI, tolerance=0.25),
        _ratio_score(mouth_nose, PHI, tolerance=0.30),
        _ratio_score(lower_nose, PHI, tolerance=0.30),
    ]
    return float(np.mean(scores))


def _facial_thirds_score(face: FaceResult) -> float:
    """Forehead, midface, lower-face heights should be roughly equal."""
    forehead_y = face.pt2("forehead")[1]
    brow_y = (face.pt2("left_brow_inner")[1] + face.pt2("right_brow_inner")[1]) / 2
    nose_tip_y = face.pt2("nose_tip")[1]
    chin_y = face.pt2("chin")[1]

    third1 = brow_y - forehead_y
    third2 = nose_tip_y - brow_y
    third3 = chin_y - nose_tip_y
    thirds = np.array([third1, third2, third3], dtype=np.float32)

    if thirds.min() <= 0:
        return 0.0

    # Coefficient of variation - lower is better.
    cv = float(thirds.std() / thirds.mean())
    score = 1.0 - min(cv / 0.25, 1.0)
    return float(max(0.0, min(1.0, score)) * 100.0)


def _eye_spacing_score(face: FaceResult) -> float:
    """Inter-ocular distance should be approximately one eye-width."""
    left_eye_w = _dist(face.pt2("left_eye_outer"), face.pt2("left_eye_inner"))
    right_eye_w = _dist(face.pt2("right_eye_inner"), face.pt2("right_eye_outer"))
    inter_eye = _dist(face.pt2("left_eye_inner"), face.pt2("right_eye_inner"))
    avg_eye_w = (left_eye_w + right_eye_w) / 2

    if avg_eye_w == 0:
        return 0.0
    ratio = inter_eye / avg_eye_w
    return _ratio_score(ratio, 1.0, tolerance=0.35)


def _jaw_balance_score(face: FaceResult) -> float:
    """Jaw/cheek width balance plus chin centering on the facial midline."""
    jaw_w = _dist(face.pt2("jaw_left"), face.pt2("jaw_right"))
    cheek_w = _dist(face.pt2("cheek_left"), face.pt2("cheek_right"))
    if cheek_w == 0:
        return 0.0
    jaw_cheek_ratio = jaw_w / cheek_w

    # A slightly narrower jaw than cheekbones is a classic marker of balance.
    shape_score = _ratio_score(jaw_cheek_ratio, 0.90, tolerance=0.25)

    # Chin centering: how close chin.x is to the midline of forehead/nose.
    mid_x = (face.pt2("forehead")[0] + face.pt2("nose_bridge_top")[0] + face.pt2("nose_tip")[0]) / 3
    chin_x = face.pt2("chin")[0]
    face_w = max(1.0, _dist(face.pt2("jaw_left"), face.pt2("jaw_right")))
    offset = abs(chin_x - mid_x) / face_w
    center_score = (1.0 - min(offset / 0.10, 1.0)) * 100.0

    return float((shape_score + center_score) / 2)


def _lip_proportion_score(face: FaceResult) -> float:
    """Upper-lip : lower-lip should be close to 1:1.6 (golden-lip guideline)."""
    mouth_w = _dist(face.pt2("mouth_left"), face.pt2("mouth_right"))
    nose_w = _dist(face.pt2("nose_left"), face.pt2("nose_right"))
    if nose_w == 0:
        return 0.0

    # Mouth width vs nose width - target ~1.5.
    mouth_nose = mouth_w / nose_w
    width_score = _ratio_score(mouth_nose, 1.5, tolerance=0.30)

    # Fullness: gap between lip top and bottom, vs mouth width.
    lip_gap = _dist(face.pt2("lip_top"), face.pt2("lip_bottom"))
    fullness_ratio = lip_gap / mouth_w if mouth_w else 0.0
    fullness_score = _ratio_score(fullness_ratio, 0.25, tolerance=0.5)

    return float((width_score + fullness_score) / 2)


# --- Public API -------------------------------------------------------------


# Weights chosen to emphasize symmetry and overall proportions which drive
# perception of facial attractiveness most strongly in the literature.
_WEIGHTS = {
    "symmetry": 0.30,
    "golden_ratio": 0.20,
    "facial_thirds": 0.15,
    "eye_spacing": 0.15,
    "jaw_balance": 0.10,
    "lip_proportion": 0.10,
}


def analyze_beauty(face: FaceResult) -> BeautyBreakdown:
    """Compute the composite beauty score and sub-scores for a detected face."""
    scores = {
        "symmetry": _symmetry_score(face),
        "golden_ratio": _golden_ratio_score(face),
        "facial_thirds": _facial_thirds_score(face),
        "eye_spacing": _eye_spacing_score(face),
        "jaw_balance": _jaw_balance_score(face),
        "lip_proportion": _lip_proportion_score(face),
    }
    overall = sum(scores[k] * w for k, w in _WEIGHTS.items())
    return BeautyBreakdown(overall=overall, **scores)


# --- Optional: human-readable notes ----------------------------------------


def describe_scores(breakdown: BeautyBreakdown) -> List[str]:
    """Return short, friendly notes describing each metric's result."""
    notes: List[str] = []

    def tier(score: float, label: str, great: str, ok: str, low: str) -> str:
        if score >= 80:
            return f"{label}: {great}"
        if score >= 55:
            return f"{label}: {ok}"
        return f"{label}: {low}"

    notes.append(tier(
        breakdown.symmetry, "Symmetry",
        "highly symmetric features",
        "generally balanced, minor asymmetry",
        "noticeable asymmetry between halves",
    ))
    notes.append(tier(
        breakdown.golden_ratio, "Golden ratio",
        "proportions align with phi",
        "proportions are close to phi",
        "proportions deviate from phi",
    ))
    notes.append(tier(
        breakdown.facial_thirds, "Facial thirds",
        "evenly balanced facial thirds",
        "slight variance across thirds",
        "uneven vertical thirds",
    ))
    notes.append(tier(
        breakdown.eye_spacing, "Eye spacing",
        "classic five-eyes spacing",
        "eye spacing close to ideal",
        "wide or narrow inter-ocular distance",
    ))
    notes.append(tier(
        breakdown.jaw_balance, "Jaw balance",
        "well-centered, tapered jawline",
        "balanced jaw/cheek ratio",
        "jaw offset or squared ratio",
    ))
    notes.append(tier(
        breakdown.lip_proportion, "Lips",
        "lip proportions match classic ratios",
        "lip proportions near ideal",
        "mouth-nose ratio deviates from norms",
    ))
    return notes
