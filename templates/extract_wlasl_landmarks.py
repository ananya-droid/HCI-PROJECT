import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path("WLASL/videos")
OUTPUT_DIR = Path("WLASL/landmarks")

SEQUENCE_LENGTH = 30
FEATURES_PER_HAND = 63
TOTAL_FEATURES = 126


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# EXTRACT LANDMARKS
# ============================================================

def extract_landmarks(results):

    left_hand = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    right_hand = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    if not results.multi_hand_landmarks:
        return np.concatenate([
            left_hand,
            right_hand
        ])

    for i, hand_landmarks in enumerate(
        results.multi_hand_landmarks
    ):

        values = []

        for landmark in hand_landmarks.landmark:

            values.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        values = np.array(
            values,
            dtype=np.float32
        )

        if (
            results.multi_handedness
            and i < len(results.multi_handedness)
        ):

            label = (
                results.multi_handedness[i]
                .classification[0]
                .label
            )

            if label == "Left":
                left_hand = values

            elif label == "Right":
                right_hand = values

    return np.concatenate([
        left_hand,
        right_hand
    ])


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(
            frame_rgb
        )

        landmarks = extract_landmarks(
            results
        )

        frames.append(
            landmarks
        )

    cap.release()

    if not frames:
        return None

    frames = np.array(
        frames,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # FIX TO 30 FRAMES
    # --------------------------------------------------------

    if len(frames) >= SEQUENCE_LENGTH:

        indices = np.linspace(
            0,
            len(frames) - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        frames = frames[
            indices
        ]

    else:

        padding = np.zeros(
            (
                SEQUENCE_LENGTH - len(frames),
                TOTAL_FEATURES
            ),
            dtype=np.float32
        )

        frames = np.vstack([
            frames,
            padding
        ])

    return frames


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("WLASL LANDMARK EXTRACTION")
    print("==============================")
    print()

    if not INPUT_DIR.exists():

        print(
            "ERROR: WLASL/videos folder not found."
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X = []
    y = []
    class_names = []

    class_dirs = sorted([
        folder
        for folder in INPUT_DIR.iterdir()
        if folder.is_dir()
    ])

    print(
        f"Classes found: {len(class_dirs)}"
    )

    print()

    # ========================================================
    # PROCESS EACH CLASS
    # ========================================================

    for class_index, class_dir in enumerate(
        class_dirs
    ):

        sign = class_dir.name

        class_names.append(
            sign
        )

        videos = sorted(
            class_dir.glob("*.mp4")
        )

        print(
            f"[{class_index + 1}/"
            f"{len(class_dirs)}] "
            f"{sign.upper()} "
            f"({len(videos)} videos)"
        )

        successful = 0

        for video_index, video_path in enumerate(
            videos
        ):

            print(
                f"    "
                f"{video_index + 1}/"
                f"{len(videos)} "
                f"{video_path.name}",
                end=" ... "
            )

            try:

                sequence = process_video(
                    video_path
                )

                if sequence is None:

                    print("FAILED")

                    continue

                X.append(
                    sequence
                )

                y.append(
                    class_index
                )

                successful += 1

                print("OK")

            except Exception as e:

                print("FAILED")
                print(
                    f"        {e}"
                )

        print(
            f"    Successful: "
            f"{successful}/{len(videos)}"
        )

        print()

    # ========================================================
    # CLOSE MEDIAPIPE
    # ========================================================

    hands.close()

    # ========================================================
    # CHECK
    # ========================================================

    if not X:

        print(
            "ERROR: No landmarks extracted."
        )

        return

    # ========================================================
    # SAVE ARRAYS
    # ========================================================

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    class_names = np.array(
        class_names
    )

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    np.save(
        OUTPUT_DIR / "class_names.npy",
        class_names
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("==============================")
    print("EXTRACTION COMPLETE")
    print("==============================")
    print()

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print()

    print("CLASS COUNTS")
    print("------------------------------")

    for index, name in enumerate(
        class_names
    ):

        count = np.sum(
            y == index
        )

        print(
            f"{index:2d} "
            f"{name.upper():20} "
            f"{count}"
        )

    print()

    print(
        "Saved to:",
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()