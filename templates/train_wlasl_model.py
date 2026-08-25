import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("WLASL/landmarks")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("==============================")
print("WLASL MODEL TRAINING")
print("==============================")
print()

X = np.load(
    DATA_DIR / "X.npy"
)

y = np.load(
    DATA_DIR / "y.npy"
)

class_names = np.load(
    DATA_DIR / "class_names.npy"
)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", len(class_names))

print()


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
    
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print()


# ============================================================
# MODEL
# ============================================================

model = Sequential([

    Input(
        shape=(
            X.shape[1],
            X.shape[2]
        )
    ),

    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(
            32
        )
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("MODEL")
print("------------------------------")

model.summary()

print()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print()
print("==============================")
print("TRAINING")
print("==============================")
print()


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=50,

    batch_size=16,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# EVALUATE
# ============================================================

print()
print("==============================")
print("EVALUATION")
print("==============================")
print()


loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(
    f"Test Loss: {loss:.4f}"
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print()


print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("==============================")
print("CONFUSION MATRIX")
print("==============================")
print()

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(class_names))
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "wlasl_sign_model.keras"
)

model.save(
    model_path
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

np.save(
    MODEL_DIR / "class_names.npy",
    class_names
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================")
print("TRAINING COMPLETE")
print("==============================")
print()

print(
    "Model saved to:"
)

print(
    model_path
)

print()

print(
    "Classes:"
)

for index, name in enumerate(
    class_names
):

    print(
        f"{index:2d} -> {name}"
    )