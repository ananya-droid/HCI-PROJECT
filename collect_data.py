"""
collect_data.py
-----------------
Collect dataset samples for ONE sign at a time, using your webcam live.

Usage:
    python collect_data.py --label HELLO --person person1 --samples 30

Each sample is a (30, 63) numpy array saved to:
    dataset/<LABEL>/<person>_<timestamp>_<n>.npy

Controls while running:
    SPACE  -> start recording one 30-frame sample
    q      -> quit
"""

import argparse
import os
import time

import cv2
import numpy as np

import config
from hand_landmarkers import HandLandmarkExtractor


def collect(label: str, person: str, num_samples: int):
    label = label.upper()
    out_dir = os.path.join(config.DATASET_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Is another app using it?")

    extractor = HandLandmarkExtractor()
    collected = 0

    print(f"[collect_data] Collecting '{label}' for person '{person}'.")
    print("Press SPACE to record a sample, 'q' to quit.")

    try:
        while collected < num_samples:
            ok, frame = cap.read()
            if not ok:
                print("[collect_data] Failed to read frame from webcam.")
                break

            display = frame.copy()
            cv2.putText(display, f"Label: {label}  Sample: {collected}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE = record   q = quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("collect_data.py", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                sequence = _record_one_sequence(cap, extractor, display_window="collect_data.py")
                if sequence is not None:
                    fname = f"{person}_{int(time.time())}_{collected}.npy"
                    np.save(os.path.join(out_dir, fname), sequence)
                    collected += 1
                    print(f"[collect_data] Saved sample {collected}/{num_samples} -> {fname}")
                else:
                    print("[collect_data] Sample discarded (no hand detected consistently).")
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    print(f"[collect_data] Done. Collected {collected} samples for '{label}'.")


def _record_one_sequence(cap, extractor, display_window):
    """Records exactly SEQUENCE_LENGTH frames of landmarks, showing countdown."""
    seq_len = config.SEQUENCE_LENGTH
    frames = []

    for i in range(seq_len):
        ok, frame = cap.read()
        if not ok:
            return None

        flat, points = extractor.extract(frame)

        display = frame.copy()
        if points:
            h, w, _ = display.shape
            for (x, y) in points:
                cv2.circle(display, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
        cv2.putText(display, f"Recording {i+1}/{seq_len}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(display_window, display)
        cv2.waitKey(1)

        frames.append(flat if flat is not None else np.zeros(config.FEATURES_PER_FRAME, dtype=np.float32))

    valid = sum(1 for f in frames if not np.all(f == 0))
    if valid / seq_len < 0.5:
        return None

    return np.stack(frames).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect dataset samples for one sign via webcam.")
    parser.add_argument("--label", required=True, help="Sign label, e.g. HELLO")
    parser.add_argument("--person", default="person1", help="Contributor id, e.g. person2")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples to collect")
    args = parser.parse_args()

    collect(args.label, args.person, args.samples)
