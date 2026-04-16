"""Flask entry point for the facescanner web app.

Run:
    python app.py
    # then open http://localhost:5000

Endpoints
---------
GET  /            - the main single-page UI.
POST /api/analyze - accepts an uploaded JPEG/PNG (multipart ``image`` field or
                    base64 ``image_base64`` JSON field) and returns a JSON
                    blob with beauty scores + country breakdown.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Optional

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from facescanner.beauty import analyze_beauty, describe_scores
from facescanner.countries import macro_breakdown, macro_to_countries, top_countries
from facescanner.ethnicity import load_classifier
from facescanner.landmarks import LandmarkDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("facescanner")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Heavy objects created once at startup.
_detector = LandmarkDetector(refine_landmarks=True)
_classifier = load_classifier()
log.info("Loaded ethnicity classifier: %s", _classifier.name)


# --- Helpers ---------------------------------------------------------------


def _decode_image(req) -> Optional[np.ndarray]:
    """Return an RGB numpy array from either a multipart upload or JSON body."""
    if "image" in req.files:
        raw = req.files["image"].read()
    else:
        body = req.get_json(silent=True) or {}
        data_url = body.get("image_base64", "")
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        if not data_url:
            return None
        try:
            raw = base64.b64decode(data_url)
        except Exception:
            return None

    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None
    return np.array(img)


def _overall_band(score: float) -> str:
    if score >= 85:
        return "Exceptional"
    if score >= 75:
        return "Very high"
    if score >= 65:
        return "High"
    if score >= 55:
        return "Above average"
    if score >= 45:
        return "Average"
    if score >= 35:
        return "Below average"
    return "Low"


# --- Routes ----------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    img = _decode_image(request)
    if img is None:
        return jsonify({
            "ok": False,
            "error": "Could not read the image. Please upload a valid JPEG or PNG."
        }), 400

    face = _detector.detect(img)
    if face is None:
        return jsonify({
            "ok": False,
            "error": "No face detected. Please use a clear, front-facing photo with good lighting."
        }), 422

    # Beauty analysis from landmarks
    beauty = analyze_beauty(face)
    beauty_notes = describe_scores(beauty)

    # Ethnicity / country analysis
    macro_probs = _classifier.predict(img, face)
    macro = macro_breakdown(macro_probs)
    country_full = macro_to_countries(macro_probs)
    countries = top_countries(country_full, n=5)

    # Bounding box so the UI can annotate the uploaded image.
    x, y, bw, bh = face.bbox
    h, w = img.shape[:2]
    bbox_rel = {
        "x": round(x / w, 4),
        "y": round(y / h, 4),
        "w": round(bw / w, 4),
        "h": round(bh / h, 4),
    }

    return jsonify({
        "ok": True,
        "classifier": _classifier.name,
        "beauty": {
            **beauty.to_dict(),
            "band": _overall_band(beauty.overall),
            "notes": beauty_notes,
        },
        "ethnicity": {
            "macro": macro,
            "countries": countries,
        },
        "bbox": bbox_rel,
    })


@app.route("/health")
def health():
    return jsonify({"ok": True, "classifier": _classifier.name})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("DEBUG")))
