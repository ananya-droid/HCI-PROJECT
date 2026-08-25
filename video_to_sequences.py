"""
video_to_sequences.py
----------------------
Shared logic for turning ANY video source (a live webcam loop or an
uploaded .mp4/.mov/.avi file) into one or more (30, 63) landmark
sequences, ready to be saved as dataset samples.

Used by:
  - collect_data.py       (live webcam)
  - app_upload.py         (uploaded recordings, via select+upload UI)
"""

import numpy as np
import cv2

import config
from hand_landmarkers import HandLandmarkExtractor


def sequences_from_video_file(video_path, extractor: HandLandmarkExtractor = None,
                               stride: int = 15, min_hand_frames_ratio: float = 0.6):
    """
    Reads a video file and slides a SEQUENCE_LENGTH-frame window over it
    (hop = `stride` frames) to produce multiple training samples from a
    single uploaded recording.

    A window is only kept if a hand was detected in at least
    `min_hand_frames_ratio` of its frames (so a clip that starts/ends
    with the hand out of frame doesn't produce garbage samples).
    Missing frames inside a kept window are filled by repeating the
    nearest valid frame, so every sample has a full (30, 63) shape.

    Returns: list of numpy arrays, each shape (SEQUENCE_LENGTH, 63)
    """
    own_extractor = extractor is None
    if own_extractor:
        extractor = HandLandmarkExtractor()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    all_frames_landmarks = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            flat, _ = extractor.extract(frame)
            all_frames_landmarks.append(flat)  # may be None
    finally:
        cap.release()
        if own_extractor:
            extractor.close()

    n = len(all_frames_landmarks)
    seq_len = config.SEQUENCE_LENGTH
    samples = []

    if n < 5:
        return samples  # too short to be usable

    start = 0
    while start < n:
        end = start + seq_len
        window = all_frames_landmarks[start:min(end, n)]

        hand_frames = [w for w in window if w is not None]
        if len(window) > 0 and (len(hand_frames) / len(window)) >= min_hand_frames_ratio:
            filled = _fill_and_pad(window, seq_len)
            samples.append(filled)

        start += stride

    return samples


def _fill_and_pad(window, seq_len):
    """Fill None frames with nearest valid neighbor, then pad/truncate to seq_len."""
    filled = list(window)

    # forward fill
    last_valid = None
    for i in range(len(filled)):
        if filled[i] is None:
            filled[i] = last_valid
        else:
            last_valid = filled[i]

    # backward fill for any leading Nones
    next_valid = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is None:
            filled[i] = next_valid
        else:
            next_valid = filled[i]

    if any(f is None for f in filled):
        # entire window had no hand at all — shouldn't happen given the ratio
        # check upstream, but guard anyway.
        filled = [np.zeros(config.FEATURES_PER_FRAME, dtype=np.float32) if f is None else f
                  for f in filled]

    arr = np.stack(filled).astype(np.float32)

    if arr.shape[0] < seq_len:
        pad_count = seq_len - arr.shape[0]
        pad = np.repeat(arr[-1:], pad_count, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > seq_len:
        arr = arr[:seq_len]

    return arr
