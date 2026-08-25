"""
hand_landmarkers.py
--------------------
Thin wrapper around the MediaPipe *Tasks* Hand Landmarker API.

IMPORTANT: this project intentionally uses the new Tasks API
(`mediapipe.tasks.python.vision`) and NOT the legacy
`mp.solutions.hands` API, because current MediaPipe versions
(this project targets MediaPipe 1.0.1) no longer expose
`mp.solutions.hands`.

You must download the model file once:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
and place it at the project root as `hand_landmarker.task`
(see config.HAND_LANDMARKER_TASK). README.md has the exact command.
"""

import os
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


class HandLandmarkExtractor:
    """
    Wraps a MediaPipe HandLandmarker in IMAGE running mode
    (good for processing frames one at a time, whether they
    come from a live webcam loop or from frames pulled out of
    an uploaded video file).
    """

    def __init__(self, model_path: str = None, num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5):
        model_path = model_path or config.HAND_LANDMARKER_TASK

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"hand_landmarker.task not found at '{model_path}'.\n"
                "Download it first — see README.md 'Setup' section."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr):
        """
        frame_bgr: a single OpenCV BGR frame (numpy array, HxWx3).

        Returns:
            landmarks_flat: numpy array of shape (63,) if a hand was
                             detected, otherwise None.
            annotated_points: raw list of (x, y) pixel-ish normalized
                             coords, useful for drawing/visualization,
                             or None.
        """
        import cv2  # local import keeps this module importable without cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        # Only the first detected hand is used (num_hands=1 by default).
        hand = result.hand_landmarks[0]

        flat = []
        points = []
        for lm in hand:
            flat.extend([lm.x, lm.y, lm.z])
            points.append((lm.x, lm.y))

        flat = np.array(flat, dtype=np.float32)

        assert flat.shape[0] == config.FEATURES_PER_FRAME, (
            f"Expected {config.FEATURES_PER_FRAME} values, got {flat.shape[0]}"
        )

        return flat, points

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
