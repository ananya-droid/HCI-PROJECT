"""
app_upload.py
--------------
A small web page for dataset collection WITHOUT needing to run a
Python script with a live webcam window. You record a short video of
yourself doing a sign on your phone/laptop (any normal camera app),
then come here, pick the sign from a dropdown, upload the video, and
the app automatically slides a window over it to generate one or more
(30, 63) training samples, saved straight into dataset/<LABEL>/.

Run:
    python app_upload.py
Then open:
    http://127.0.0.1:5000
"""

import os
import time
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash

import config
from video_to_sequences import sequences_from_video_file
from hand_landmarkers import HandLandmarkExtractor

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

app = Flask(__name__)
app.secret_key = "dev-only-secret-change-me"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB per upload

_extractor = None


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = HandLandmarkExtractor()
    return _extractor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def dataset_counts():
    counts = {}
    for label in config.ALL_VOCABULARY:
        label_dir = os.path.join(config.DATASET_DIR, label)
        if os.path.isdir(label_dir):
            counts[label] = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
        else:
            counts[label] = 0
    return counts


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "upload.html",
        labels=config.ALL_VOCABULARY,
        counts=dataset_counts(),
    )


@app.route("/upload", methods=["POST"])
def upload():
    label = request.form.get("label", "").strip().upper()
    person = request.form.get("person", "anonymous").strip() or "anonymous"
    file = request.files.get("video")

    if not label or label not in config.ALL_VOCABULARY:
        flash("Please choose a valid sign label from the dropdown.", "error")
        return redirect(url_for("index"))

    if not file or file.filename == "":
        flash("Please choose a video file to upload.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", "error")
        return redirect(url_for("index"))

    tmp_name = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(config.UPLOADS_TMP_DIR, tmp_name)
    file.save(tmp_path)

    try:
        extractor = get_extractor()
        samples = sequences_from_video_file(tmp_path, extractor=extractor)

        if not samples:
            flash(
                "No usable hand movement was detected in that video "
                "(try better lighting or keep your hand in frame throughout).",
                "error",
            )
            return redirect(url_for("index"))

        out_dir = os.path.join(config.DATASET_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        import numpy as np
        saved = 0
        for i, seq in enumerate(samples):
            fname = f"{person}_{int(time.time())}_{i}.npy"
            np.save(os.path.join(out_dir, fname), seq)
            saved += 1

        flash(f"✅ Saved {saved} sample(s) for '{label}' from your upload. Thank you!", "success")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return redirect(url_for("index"))


if __name__ == "__main__":
    if not os.path.exists(config.HAND_LANDMARKER_TASK):
        print("⚠️  hand_landmarker.task not found — see README.md before uploading videos.")
    app.run(debug=True, host="127.0.0.1", port=5000)
