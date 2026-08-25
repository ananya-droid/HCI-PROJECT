"""
augment_dataset.py

Creates additional training samples from existing real
MediaPipe landmark sequences.

Original:
    (30, 63)

Augmentations:
    - small coordinate noise
    - small translation
    - small scaling
    - small temporal variation

IMPORTANT:
These are augmented samples, not new real people.
"""

import os
import numpy as np
import config


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TARGET_SAMPLES_PER_CLASS = 60

NOISE_STD = 0.008
TRANSLATION_RANGE = 0.03
SCALE_RANGE = 0.08


# ---------------------------------------------------------
# LOAD SAMPLE
# ---------------------------------------------------------

def load_sample(path):

    data = np.load(path)

    data = np.asarray(data, dtype=np.float32)

    # Expected shape:
    # (30, 63)

    if data.shape != (config.SEQUENCE_LENGTH, 63):
        return None

    return data


# ---------------------------------------------------------
# AUGMENTATION 1: SMALL NOISE
# ---------------------------------------------------------

def add_noise(sequence):

    noise = np.random.normal(
        0,
        NOISE_STD,
        sequence.shape
    )

    return sequence + noise


# ---------------------------------------------------------
# AUGMENTATION 2: TRANSLATION
# ---------------------------------------------------------

def translate(sequence):

    # Each frame contains:
    #
    # x1,y1,z1,
    # x2,y2,z2,
    # ...
    #
    shift_x = np.random.uniform(
        -TRANSLATION_RANGE,
        TRANSLATION_RANGE
    )

    shift_y = np.random.uniform(
        -TRANSLATION_RANGE,
        TRANSLATION_RANGE
    )

    augmented = sequence.copy()

    augmented[:, 0::3] += shift_x
    augmented[:, 1::3] += shift_y

    return augmented


# ---------------------------------------------------------
# AUGMENTATION 3: SCALE
# ---------------------------------------------------------

def scale(sequence):

    factor = np.random.uniform(
        1 - SCALE_RANGE,
        1 + SCALE_RANGE
    )

    augmented = sequence.copy()

    # Scale around approximate hand center.
    center = np.mean(
        augmented[:, :2],
        axis=1,
        keepdims=True
    )

    for frame in range(len(augmented)):

        xy = augmented[frame].reshape(-1, 3)

        xy[:, :2] = (
            center[frame]
            + (xy[:, :2] - center[frame]) * factor
        )

        augmented[frame] = xy.reshape(-1)

    return augmented


# ---------------------------------------------------------
# COMBINED AUGMENTATION
# ---------------------------------------------------------

def augment(sequence):

    result = sequence.copy()

    # Randomly apply transformations.

    if np.random.rand() < 0.8:
        result = add_noise(result)

    if np.random.rand() < 0.5:
        result = translate(result)

    if np.random.rand() < 0.5:
        result = scale(result)

    return result.astype(np.float32)


# ---------------------------------------------------------
# PROCESS ONE CLASS
# ---------------------------------------------------------

def process_class(label):

    folder = os.path.join(
        config.DATASET_DIR,
        label
    )

    if not os.path.isdir(folder):
        print(f"[SKIP] {label}: folder not found")
        return

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
        and not f.startswith("aug_")
    ]

    if not files:
        print(f"[SKIP] {label}: no samples")
        return

    original_count = len(files)

    print(
        f"\n{label}: "
        f"{original_count} original samples"
    )

    if original_count >= TARGET_SAMPLES_PER_CLASS:
        print("  Already has enough samples.")
        return

    needed = TARGET_SAMPLES_PER_CLASS - original_count

    print(
        f"  Creating {needed} augmented samples..."
    )

    for i in range(needed):

        # Pick a random REAL sample.
        source_file = np.random.choice(files)

        source_path = os.path.join(
            folder,
            source_file
        )

        sample = load_sample(source_path)

        if sample is None:
            print(
                f"  [WARNING] Invalid sample: "
                f"{source_file}"
            )
            continue

        augmented = augment(sample)

        output_name = (
            f"aug_{i:04d}.npy"
        )

        output_path = os.path.join(
            folder,
            output_name
        )

        np.save(
            output_path,
            augmented
        )

    final_count = len([
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ])

    print(
        f"  Final samples: {final_count}"
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 50)
    print("AI-ASSISTED DATA AUGMENTATION")
    print("=" * 50)

    print(
        f"Target samples per class: "
        f"{TARGET_SAMPLES_PER_CLASS}"
    )

    for label in config.ALL_VOCABULARY:

        process_class(label)

    print("\n" + "=" * 50)
    print("AUGMENTATION COMPLETE")
    print("=" * 50)

    print(
        "\nIMPORTANT:"
        "\nAugmented samples are derived from real samples."
        "\nThey do NOT represent additional real signers."
    )


if __name__ == "__main__":
    main()