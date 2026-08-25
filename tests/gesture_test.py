"""
gesture_test.py — verifies MediaPipe landmark extraction and (if a model
has been trained) that GestureRecognizer loads and predicts without error.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import config


def test_landmark_extraction():
    from hand_landmarkers import HandLandmarkExtractor

    if not os.path.exists(config.HAND_LANDMARKER_TASK):
        print("⚠️  Skipping: hand_landmarker.task not present.")
        return

    extractor = HandLandmarkExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    flat, points = extractor.extract(dummy_frame)
    extractor.close()

    # A blank frame should yield no hand.
    assert flat is None
    print("✅ Landmark extractor handles a hand-less frame correctly.")


def test_model_predicts():
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.LABELS_PATH)):
        print("⚠️  Skipping: model not trained yet (run train_model.py first).")
        return

    from gesture_recognizer import GestureRecognizer

    recognizer = GestureRecognizer()
    dummy_sequence = np.random.rand(config.FEATURES_PER_FRAME).astype(np.float32)

    result = None
    for _ in range(config.SEQUENCE_LENGTH + config.STABLE_FRAMES_REQUIRED):
        result = recognizer.update(dummy_sequence)

    print(f"✅ GestureRecognizer ran without error. Last result: {result}")


if __name__ == "__main__":
    test_landmark_extraction()
    test_model_predicts()
