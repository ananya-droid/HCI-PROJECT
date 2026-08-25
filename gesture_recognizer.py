"""
gesture_recognizer.py
-----------------------
Streaming LSTM gesture recognizer with:

- confidence threshold
- stable-frame debouncing
- robust hand-release detection
- protection against duplicate tokens
- allows intentional repeated signs such as YES YES
"""

import numpy as np
import tensorflow as tf

import config


class GestureRecognizer:
    def __init__(
        self,
        model_path=None,
        labels_path=None,
        confidence_threshold=None,
        stable_frames_required=None,
    ):
        model_path = model_path or config.MODEL_PATH
        labels_path = labels_path or config.LABELS_PATH

        self.model = tf.keras.models.load_model(model_path)
        self.labels = np.load(labels_path, allow_pickle=True)

        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.CONFIDENCE_THRESHOLD
        )

        self.stable_frames_required = (
            stable_frames_required
            if stable_frames_required is not None
            else config.STABLE_FRAMES_REQUIRED
        )

        # Rolling landmark sequence
        self.buffer = []

        # Prediction stability
        self.stable_prediction = None
        self.stable_count = 0

        # Token emission state
        self.awaiting_hand_release = False
        self.last_emitted = None

        # IMPORTANT:
        # MediaPipe can temporarily lose the hand for a frame or two.
        # Don't unlock immediately.
        self.hand_absent_count = 0

        # Number of consecutive missing-hand frames required
        # before we consider the gesture finished.
        self.hand_release_frames_required = 5

    def update(self, landmarks):
        """
        landmarks:
            (63,) array when a hand is detected
            None when no hand is detected

        Returns:
            token string when a new stable gesture is recognized
            None otherwise
        """

        # ==========================================================
        # CASE 1: NO HAND DETECTED
        # ==========================================================
        if landmarks is None:

            self.hand_absent_count += 1

            # Only consider the gesture finished after several
            # consecutive missing-hand frames.
            if self.hand_absent_count >= self.hand_release_frames_required:

                self.buffer.clear()
                self.stable_prediction = None
                self.stable_count = 0
                self.awaiting_hand_release = False
                self.last_emitted = None

            return None

        # ==========================================================
        # CASE 2: HAND DETECTED
        # ==========================================================

        # Hand has returned, so reset the temporary missing counter.
        self.hand_absent_count = 0

        # Add current landmark frame.
        self.buffer.append(landmarks)

        # Keep only the most recent SEQUENCE_LENGTH frames.
        if len(self.buffer) > config.SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # Need a complete sequence before LSTM prediction.
        if len(self.buffer) < config.SEQUENCE_LENGTH:
            return None

        # ==========================================================
        # CASE 3: WAITING FOR GESTURE RELEASE
        # ==========================================================

        if self.awaiting_hand_release:
            """
            A token has already been emitted.

            Example:

                HELP
                ↓
                token emitted
                ↓
                user keeps holding HELP
                ↓
                IGNORE
                ↓
                hand disappears
                ↓
                unlock

            This prevents:

                HELP HELP HELP HELP
            """

            return None

        # ==========================================================
        # CASE 4: RUN LSTM PREDICTION
        # ==========================================================

        sequence = np.expand_dims(
            np.array(self.buffer, dtype=np.float32),
            axis=0
        )

        probs = self.model.predict(sequence, verbose=0)[0]

        best_idx = int(np.argmax(probs))

        confidence = float(probs[best_idx])

        prediction = str(self.labels[best_idx])

        # ==========================================================
        # CASE 5: LOW CONFIDENCE
        # ==========================================================

        if confidence < self.confidence_threshold:

            self.stable_prediction = None
            self.stable_count = 0

            return None

        # ==========================================================
        # CASE 6: STABILITY CHECK
        # ==========================================================

        if prediction == self.stable_prediction:

            self.stable_count += 1

        else:

            self.stable_prediction = prediction
            self.stable_count = 1

        # ==========================================================
        # CASE 7: EMIT TOKEN
        # ==========================================================

        if self.stable_count >= self.stable_frames_required:

            # Reset stability counter.
            self.stable_count = 0

            # Lock recognizer until gesture boundary.
            self.awaiting_hand_release = True

            # Remember what was emitted.
            self.last_emitted = prediction

            return prediction

        return None

    def status_label(self):
        """
        UI status.
        """

        if len(self.buffer) < config.SEQUENCE_LENGTH:
            return "..."

        if self.awaiting_hand_release:
            return f"Recognized: {self.last_emitted}"

        if self.stable_prediction is None:
            return "Uncertain"

        return self.stable_prediction