"""
train_model.py
----------------
Loads dataset/, trains the LSTM gesture-recognition model, evaluates it,
and saves sign_model.keras + label_classes.npy.

Usage:
    python train_model.py
    python train_model.py --epochs 100 --batch-size 16
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import config
from check_dataset import check_dataset


def load_dataset(dataset_dir=None, labels=None):
    dataset_dir = dataset_dir or config.DATASET_DIR
    labels = labels or config.ALL_VOCABULARY

    X, y = [], []
    for label in labels:
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(label_dir, fname))
            if arr.shape != config.INPUT_SHAPE:
                continue
            X.append(arr)
            y.append(label)

    if not X:
        raise RuntimeError("No valid samples found. Run check_dataset.py first.")

    X = np.array(X, dtype=np.float32)
    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in y], dtype=np.int64)

    return X, y_idx, np.array(classes)


def build_model(num_classes: int, input_shape=None):
    input_shape = input_shape or config.INPUT_SHAPE
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(epochs: int, batch_size: int, test_size: float):
    print("[train_model] Verifying dataset first...")
    report = check_dataset()
    if report["missing_classes"] or report["total_valid"] < 30:
        print("\n[train_model] Dataset isn't ready yet (see warnings above). "
              "Continuing anyway, but expect a weak model.\n")

    print("[train_model] Loading dataset...")
    X, y, classes = load_dataset()
    print(f"[train_model] Loaded {len(X)} samples across {len(classes)} classes: {list(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y if len(classes) > 1 else None, random_state=42
    )

    model = build_model(num_classes=len(classes))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )

    print("\n[train_model] Evaluating on held-out test set...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(
    y_test,
    y_pred,
    labels=range(len(classes)),
    target_names=classes,
    zero_division=0
))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save(config.MODEL_PATH)
    np.save(config.LABELS_PATH, classes)

    print(f"\n[train_model] Saved model to {config.MODEL_PATH}")
    print(f"[train_model] Saved label classes to {config.LABELS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the sign-language LSTM model.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    main(args.epochs, args.batch_size, args.test_size)
