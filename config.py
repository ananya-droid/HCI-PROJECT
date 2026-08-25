"""
config.py
---------
Single source of truth for paths, vocabulary and model shape constants.
Every other script imports from here so you only ever change things
in one place.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "label_classes.npy")
HAND_LANDMARKER_TASK = os.path.join(BASE_DIR, "hand_landmarker.task")
UPLOADS_TMP_DIR = os.path.join(BASE_DIR, "uploads_tmp")

# ---------------------------------------------------------------------------
# Sign vocabulary (Phase 1 — start here, don't add everything at once)
# ---------------------------------------------------------------------------
CORE_VOCABULARY = [
    "HELLO", "YES", "NO", "PLEASE", "THANK_YOU",
    "HELP", "I", "YOU", "NEED", "WATER",
    "FOOD", "STOP", "GOOD", "BAD", "BYE",
]

# Phase 2 — interview demo vocabulary. Add these to dataset/ later and
# re-run collect_data.py / the upload app for them, then retrain.
INTERVIEW_VOCABULARY = [
    "STUDENT", "COMPUTER_SCIENCE", "AI", "PYTHON", "MACHINE_LEARNING",
    "EXPERIENCE", "SKILL", "PROJECT", "INTEREST", "WORK",
    "COMPANY", "JOB", "WHY", "BECAUSE", "LEARN", "TEAM", "PROBLEM", "SOLVE",
]

# All labels currently expected to have a dataset/<LABEL>/ folder.
ALL_VOCABULARY = CORE_VOCABULARY  # extend to CORE_VOCABULARY + INTERVIEW_VOCABULARY when ready

# ---------------------------------------------------------------------------
# Sequence / landmark shape
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30      # frames per sample
NUM_LANDMARKS = 21        # MediaPipe hand landmarks
NUM_COORDS = 3            # x, y, z
FEATURES_PER_FRAME = NUM_LANDMARKS * NUM_COORDS  # 63
INPUT_SHAPE = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)

# ---------------------------------------------------------------------------
# Recognition stability
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES_REQUIRED = 8

# ---------------------------------------------------------------------------
# Conversation modes
# ---------------------------------------------------------------------------
MODE_GENERAL = "GENERAL"
MODE_INTERVIEW = "INTERVIEW"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOADS_TMP_DIR, exist_ok=True)
for label in CORE_VOCABULARY:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)
