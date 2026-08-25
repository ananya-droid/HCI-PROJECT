"""
check_dataset.py
------------------
Verifies the dataset before training:
    - number of samples per class
    - shape of every sample (must be (30, 63))
    - missing classes (folders with 0 samples)
    - malformed / unreadable .npy files

Usage:
    python check_dataset.py
"""

import os
import numpy as np

import config


def check_dataset(dataset_dir: str = None, expected_labels=None, expected_shape=None):
    dataset_dir = dataset_dir or config.DATASET_DIR
    expected_labels = expected_labels or config.ALL_VOCABULARY
    expected_shape = expected_shape or config.INPUT_SHAPE

    print("=" * 40)
    print("DATASET CHECK")
    print("=" * 40)

    total_valid = 0
    total_invalid = 0
    missing_classes = []
    invalid_files = []
    per_class_counts = {}

    for label in expected_labels:
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            missing_classes.append(label)
            per_class_counts[label] = 0
            print(f"{label:<20} -> MISSING FOLDER")
            continue

        files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
        valid_count = 0

        for fname in files:
            fpath = os.path.join(label_dir, fname)
            try:
                arr = np.load(fpath)
            except Exception as e:
                invalid_files.append((fpath, f"failed to load: {e}"))
                total_invalid += 1
                continue

            if arr.shape != expected_shape:
                invalid_files.append((fpath, f"bad shape {arr.shape}, expected {expected_shape}"))
                total_invalid += 1
                continue

            if not np.isfinite(arr).all():
                invalid_files.append((fpath, "contains NaN/Inf"))
                total_invalid += 1
                continue

            valid_count += 1

        per_class_counts[label] = valid_count
        total_valid += valid_count

        if valid_count == 0:
            missing_classes.append(label)

        print(f"{label:<20} -> {valid_count} samples")

    print("-" * 40)
    print(f"TOTAL VALID SAMPLES:   {total_valid}")
    print(f"TOTAL INVALID SAMPLES: {total_invalid}")

    if missing_classes:
        print(f"\n⚠️  Classes with 0 valid samples: {', '.join(missing_classes)}")

    if invalid_files:
        print(f"\n⚠️  Malformed files ({len(invalid_files)}):")
        for fpath, reason in invalid_files:
            print(f"   - {fpath}: {reason}")

    if not missing_classes and not invalid_files:
        print("\n✅ Dataset looks good. Ready for train_model.py.")
    else:
        print("\n❌ Fix the issues above before training, or training will "
              "either fail or produce a poor model.")

    print("=" * 40)

    return {
        "per_class_counts": per_class_counts,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "missing_classes": missing_classes,
        "invalid_files": invalid_files,
    }


if __name__ == "__main__":
    check_dataset()
