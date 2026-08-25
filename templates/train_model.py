import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"

LABELS = [
    "HELLO",
    "YES",
    "NO",
    "PLEASE",
    "THANK_YOU",
    "HELP",
    "I",
]

SEQUENCE_LENGTH = 30
FEATURES = 63

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

EPOCHS = 30
BATCH_SIZE = 8

RANDOM_STATE = 42


# ============================================================
# AUGMENTATION SETTINGS
# ============================================================

NOISE_STD = 0.008
TRANSLATION_RANGE = 0.03
SCALE_RANGE = 0.08


# ============================================================
# LOAD ONLY REAL DATA
# ============================================================

def load_real_dataset():

    X = []
    y = []

    print("\n========================================")
    print("LOADING REAL DATASET")
    print("========================================")

    for label in LABELS:

        folder = os.path.join(
            DATASET_DIR,
            label
        )

        if not os.path.exists(folder):

            print(
                f"{label:15s} -> folder missing"
            )

            continue

        # IMPORTANT:
        # Ignore augmented files.
        files = [
            f for f in os.listdir(folder)
            if f.endswith(".npy")
            and not f.startswith("aug_")
        ]

        valid_count = 0

        for file in files:

            path = os.path.join(
                folder,
                file
            )

            try:

                sequence = np.load(path)

            except Exception as e:

                print(
                    f"Skipping {file}: {e}"
                )

                continue

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURES
            ):

                print(
                    f"Skipping {file}: "
                    f"wrong shape {sequence.shape}"
                )

                continue

            X.append(
                sequence.astype(np.float32)
            )

            y.append(label)

            valid_count += 1

        print(
            f"{label:15s} -> "
            f"{valid_count} REAL samples"
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(y)

    print("----------------------------------------")
    print(
        f"TOTAL REAL SAMPLES: {len(X)}"
    )

    return X, y


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def add_noise(sequence):

    noise = np.random.normal(
        0,
        NOISE_STD,
        sequence.shape
    )

    return sequence + noise


def translate(sequence):

    augmented = sequence.copy()

    shift_x = np.random.uniform(
        -TRANSLATION_RANGE,
        TRANSLATION_RANGE
    )

    shift_y = np.random.uniform(
        -TRANSLATION_RANGE,
        TRANSLATION_RANGE
    )

    augmented[:, 0::3] += shift_x
    augmented[:, 1::3] += shift_y

    return augmented


def scale(sequence):

    augmented = sequence.copy()

    factor = np.random.uniform(
        1 - SCALE_RANGE,
        1 + SCALE_RANGE
    )

    for frame in range(
        len(augmented)
    ):

        points = augmented[
            frame
        ].reshape(-1, 3)

        center = np.mean(
            points[:, :2],
            axis=0
        )

        points[:, :2] = (
            center
            + (
                points[:, :2]
                - center
            ) * factor
        )

        augmented[
            frame
        ] = points.reshape(-1)

    return augmented


def augment_sequence(sequence):

    result = sequence.copy()

    if np.random.rand() < 0.8:

        result = add_noise(result)

    if np.random.rand() < 0.5:

        result = translate(result)

    if np.random.rand() < 0.5:

        result = scale(result)

    return result.astype(
        np.float32
    )


# ============================================================
# AUGMENT TRAINING DATA
# ============================================================

def augment_training_data(
    X_train,
    y_train,
    target_per_class=60
):

    print("\n========================================")
    print("AUGMENTING TRAINING DATA")
    print("========================================")

    X_augmented = list(X_train)
    y_augmented = list(y_train)

    for label in np.unique(y_train):

        indices = np.where(
            y_train == label
        )[0]

        current_count = len(indices)

        print(
            f"{label:15s} -> "
            f"{current_count} real training samples"
        )

        if current_count >= target_per_class:

            continue

        needed = (
            target_per_class
            - current_count
        )

        print(
            f"                  "
            f"+ {needed} augmented samples"
        )

        for _ in range(needed):

            source_index = np.random.choice(
                indices
            )

            source = X_train[
                source_index
            ]

            augmented = augment_sequence(
                source
            )

            X_augmented.append(
                augmented
            )

            y_augmented.append(
                label
            )

    X_augmented = np.array(
        X_augmented,
        dtype=np.float32
    )

    y_augmented = np.array(
        y_augmented
    )

    print("----------------------------------------")
    print(
        "Training samples after augmentation:",
        len(X_augmented)
    )

    return X_augmented, y_augmented


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD REAL DATA
    # --------------------------------------------------------

    X, y = load_real_dataset()

    if len(X) == 0:

        raise RuntimeError(
            "No valid training data found."
        )

    # --------------------------------------------------------
    # 2. ENCODE LABELS
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(
        y
    )

    classes = encoder.classes_

    print("\n========================================")
    print("CLASSES")
    print("========================================")

    print(classes)

    # --------------------------------------------------------
    # 3. FIRST SPLIT
    #
    # REAL DATA ONLY
    #
    # 80% development
    # 20% final test
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(

        X,
        y_encoded,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_encoded
    )

    print("\n========================================")
    print("TEST SPLIT")
    print("========================================")

    print(
        "Development samples:",
        len(X_dev)
    )

    print(
        "Final test samples:",
        len(X_test)
    )

    # --------------------------------------------------------
    # 4. SECOND SPLIT
    #
    # DEVELOPMENT -> TRAIN + VALIDATION
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(

        X_dev,
        y_dev,

        test_size=VALIDATION_SIZE,

        random_state=RANDOM_STATE,

        stratify=y_dev
    )

    print("\n========================================")
    print("TRAIN / VALIDATION SPLIT")
    print("========================================")

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    # --------------------------------------------------------
    # 5. AUGMENT TRAINING ONLY
    # --------------------------------------------------------

    X_train, y_train = augment_training_data(
        X_train,
        y_train,
        target_per_class=60
    )

    # --------------------------------------------------------
    # 6. ONE-HOT ENCODE
    # --------------------------------------------------------

    y_train_cat = to_categorical(
        y_train,
        num_classes=len(classes)
    )

    y_val_cat = to_categorical(
        y_val,
        num_classes=len(classes)
    )

    y_test_cat = to_categorical(
        y_test,
        num_classes=len(classes)
    )

    # --------------------------------------------------------
    # 7. BUILD LSTM
    # --------------------------------------------------------

    print("\n========================================")
    print("BUILDING LSTM")
    print("========================================")

    model = Sequential([

        Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURES
            )
        ),

        Masking(
            mask_value=0.0
        ),

        LSTM(
            128,
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(
            64
        ),

        Dropout(0.3),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.2),

        Dense(
            len(classes),
            activation="softmax"
        )
    ])

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]
    )

    model.summary()

    # --------------------------------------------------------
    # 8. CALLBACKS
    # --------------------------------------------------------

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=7,

        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(

        "best_sign_model.keras",

        monitor="val_loss",

        save_best_only=True
    )

    # --------------------------------------------------------
    # 9. TRAIN
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING")
    print("========================================")

    history = model.fit(

        X_train,

        y_train_cat,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        validation_data=(

            X_val,

            y_val_cat
        ),

        callbacks=[

            early_stopping,

            checkpoint
        ],

        verbose=2
    )

    # --------------------------------------------------------
    # 10. FINAL TEST
    #
    # IMPORTANT:
    # This data was NEVER augmented.
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL TEST EVALUATION")
    print("========================================")

    loss, accuracy = model.evaluate(

        X_test,

        y_test_cat,

        verbose=0
    )

    print(
        f"\nFINAL TEST ACCURACY: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 11. PREDICTIONS
    # --------------------------------------------------------

    probabilities = model.predict(

        X_test,

        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # 12. CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(

            y_test,

            y_pred,

            labels=np.arange(
                len(classes)
            ),

            target_names=classes,

            zero_division=0
        )
    )

    # --------------------------------------------------------
    # 13. CONFUSION MATRIX
    # --------------------------------------------------------

    print("\n========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    print(
        confusion_matrix(

            y_test,

            y_pred,

            labels=np.arange(
                len(classes)
            )
        )
    )

    # --------------------------------------------------------
    # 14. SAVE FINAL MODEL
    # --------------------------------------------------------

    model.save(
        "sign_model.keras"
    )

    np.save(
        "label_classes.npy",
        classes
    )

    print("\n========================================")
    print("MODEL SAVED")
    print("========================================")

    print(
        "sign_model.keras"
    )

    print(
        "label_classes.npy"
    )

    print(
        "best_sign_model.keras"
    )

    print("\nTraining complete.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()