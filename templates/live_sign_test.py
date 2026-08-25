import cv2
import numpy as np
import tensorflow as tf
import threading
import pyttsx3
import mediapipe as mp

from communication_pipeline import CommunicationPipeline


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/wlasl_sign_model.keras"
LABEL_PATH = "WLASL/landmarks/class_names.npy"

SEQUENCE_LENGTH = 30

FEATURES_PER_HAND = 63
TOTAL_FEATURES = 126

MIN_CONFIDENCE = 0.70

STABLE_FRAMES_REQUIRED = 8

HAND_RELEASE_FRAMES = 8


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print("Classes:", labels)

print(
    "Expected model input:",
    model.input_shape
)


# ============================================================
# COMMUNICATION PIPELINE
# ============================================================

print("Loading communication pipeline...")

pipeline = CommunicationPipeline()


# ============================================================
# TTS
# ============================================================

def speak_sentence(sentence):

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            150
        )

        engine.say(
            sentence
        )

        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print(
            "TTS error:",
            e
        )


# ============================================================
# STATE
# ============================================================

processing = False

generated_sentence = ""

current_mode = "GENERAL"

interviewer_question = ""


# ============================================================
# GENERATE SENTENCE
# ============================================================

def generate_sentence():

    global processing
    global generated_sentence

    try:

        processing = True

        print()
        print("==============================")
        print("COMMUNICATION")
        print("==============================")

        tokens = pipeline.get_tokens()

        print(
            "Tokens:",
            tokens
        )

        if not tokens:

            print(
                "No signs detected."
            )

            return

        print(
            "Generating natural English..."
        )

        sentence = pipeline.finish_message()

        generated_sentence = sentence

        print(
            "Sentence:",
            sentence
        )

        print("==============================")
        print()

        if sentence:

            speak_sentence(
                sentence
            )

    except Exception as e:

        print()
        print("Communication error:")
        print(e)
        print()

    finally:

        processing = False


# ============================================================
# MEDIAPIPE HANDS
# ============================================================

print(
    "Loading MediaPipe Hands..."
)

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Left hand  = 63 values
    # Right hand = 63 values
    #
    # Total      = 126 values
    #
    # This MUST match the training data.
    # --------------------------------------------------------

    left_hand = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )

    right_hand = np.zeros(
        FEATURES_PER_HAND,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # NO HAND
    # --------------------------------------------------------

    if not results.multi_hand_landmarks:

        return np.concatenate([
            left_hand,
            right_hand
        ])


    # --------------------------------------------------------
    # PROCESS DETECTED HANDS
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # HANDEDNESS
        # ----------------------------------------------------

        if (
            results.multi_handedness
            and i < len(
                results.multi_handedness
            )
        ):

            label = (
                results
                .multi_handedness[i]
                .classification[0]
                .label
            )


            if label == "Left":

                left_hand = values


            elif label == "Right":

                right_hand = values


    # --------------------------------------------------------
    # 126 FEATURES
    # --------------------------------------------------------

    combined = np.concatenate([
        left_hand,
        right_hand
    ])


    return combined


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    hands.close()

    exit()


# ============================================================
# START
# ============================================================

print()

print("==============================")
print("CONTEXT-AWARE COMMUNICATION")
print("==============================")

print(
    "Available signs:",
    labels
)

print()

print("Controls:")
print("S = Finish sentence + speak")
print("I = Interview mode")
print("G = General mode")
print("C = Clear")
print("Q = Quit")

print()

print(
    "Gemini sentence generation ENABLED."
)

print()

print(
    "Live feature size: 126"
)

print(
    "Sequence length: 30"
)

print()


# ============================================================
# VARIABLES
# ============================================================

sequence = []

prediction = "NO HAND"

confidence = 0.0

locked_sign = None

hand_release_count = 0

stable_prediction = None

stable_count = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        continue


    # ========================================================
    # MIRROR CAMERA
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    results = hands.process(
        rgb_frame
    )


    # ========================================================
    # HAND FOUND
    # ========================================================

    if results.multi_hand_landmarks:

        hand_release_count = 0


        # ====================================================
        # EXTRACT 126 FEATURES
        # ====================================================

        landmarks = extract_landmarks(
            results
        )


        # Safety check
        if len(landmarks) != TOTAL_FEATURES:

            print(
                "ERROR: Feature size:",
                len(landmarks)
            )

            continue


        sequence.append(
            landmarks
        )


        if len(sequence) > SEQUENCE_LENGTH:

            sequence = sequence[
                -SEQUENCE_LENGTH:
            ]


        # ====================================================
        # DRAW HANDS
        # ====================================================

        h, w, _ = frame.shape


        for hand_landmarks in (
            results.multi_hand_landmarks
        ):

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


        # ====================================================
        # PREDICTION
        # ====================================================

        if len(sequence) == SEQUENCE_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )


            # Expected:
            #
            # (30, 126)
            #

            if input_data.shape != (
                SEQUENCE_LENGTH,
                TOTAL_FEATURES
            ):

                print(
                    "ERROR: Input shape:",
                    input_data.shape
                )

                continue


            input_data = np.expand_dims(
                input_data,
                axis=0
            )


            # Expected:
            #
            # (1, 30, 126)
            #

            probabilities = model.predict(
                input_data,
                verbose=0
            )[0]


            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )


            prediction = str(
                labels[predicted_index]
            )


            confidence = float(
                probabilities[
                    predicted_index
                ]
            )


            # =================================================
            # STABILITY
            # =================================================

            if confidence >= MIN_CONFIDENCE:

                if prediction == stable_prediction:

                    stable_count += 1

                else:

                    stable_prediction = prediction

                    stable_count = 1


                # =================================================
                # ADD SIGN
                # =================================================

                if (
                    stable_count
                    >= STABLE_FRAMES_REQUIRED

                    and prediction
                    != locked_sign

                    and not processing
                ):

                    pipeline.add_sign(
                        prediction
                    )

                    locked_sign = prediction

                    print(
                        "Recognized:",
                        prediction
                    )

                    stable_count = 0


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        sequence = []

        prediction = "NO HAND"

        confidence = 0.0

        stable_prediction = None

        stable_count = 0


        # ====================================================
        # HAND RELEASE
        # ====================================================

        if locked_sign is not None:

            hand_release_count += 1


            if (
                hand_release_count
                >= HAND_RELEASE_FRAMES
            ):

                locked_sign = None

                hand_release_count = 0


    # ========================================================
    # TOKENS
    # ========================================================

    current_tokens = pipeline.get_tokens()

    current_text = " ".join(
        current_tokens
    )


    # ========================================================
    # DISPLAY PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (1100, 270),
        (0, 0, 0),
        -1
    )


    # ========================================================
    # SIGN
    # ========================================================

    cv2.putText(
        frame,
        f"Sign: {prediction}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # FRAMES
    # ========================================================

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FEATURE SIZE
    # ========================================================

    cv2.putText(
        frame,
        "Features: 126",
        (300, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SIGNS
    # ========================================================

    cv2.putText(
        frame,
        f"Signs: {current_text}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # ENGLISH SENTENCE
    # ========================================================

    display_sentence = generated_sentence


    if len(display_sentence) > 90:

        display_sentence = (
            display_sentence[:87]
            + "..."
        )


    cv2.putText(
        frame,
        f"English: {display_sentence}",
        (30, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2
    )


    # ========================================================
    # MODE
    # ========================================================

    cv2.putText(
        frame,
        f"Mode: {current_mode}",
        (30, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # STATUS
    # ========================================================

    if processing:

        status = "Generating..."

    else:

        status = "Ready"


    cv2.putText(
        frame,
        f"Status: {status}",
        (30, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CAMERA WINDOW
    # ========================================================

    cv2.imshow(
        "Context-Aware Communication",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # FINISH SENTENCE
    # ========================================================

    if key == ord("s"):

        if not pipeline.get_tokens():

            print(
                "No signs detected."
            )

            continue


        if processing:

            print(
                "Already generating. Please wait."
            )

            continue


        thread = threading.Thread(
            target=generate_sentence,
            daemon=True
        )

        thread.start()


        # Reset recognition state

        locked_sign = None

        hand_release_count = 0

        stable_prediction = None

        stable_count = 0

        sequence = []


    # ========================================================
    # INTERVIEW MODE
    # ========================================================

    elif key == ord("i"):

        if processing:

            print(
                "Please wait until generation finishes."
            )

            continue


        print()
        print("==============================")
        print("INTERVIEW MODE")
        print("==============================")


        question = input(
            "Enter interviewer question: "
        ).strip()


        if question:

            current_mode = "INTERVIEW"

            interviewer_question = question


            pipeline.set_mode(
                "INTERVIEW"
            )


            pipeline.add_interviewer_message(
                question
            )


            print()
            print(
                "Interviewer:"
            )

            print(
                question
            )

            print()
            print(
                "Now perform your signs."
            )

            print(
                "Press S when finished."
            )

            print()

        else:

            print(
                "No question entered."
            )


    # ========================================================
    # GENERAL MODE
    # ========================================================

    elif key == ord("g"):

        if processing:

            print(
                "Please wait until generation finishes."
            )

            continue


        current_mode = "GENERAL"

        interviewer_question = ""


        pipeline.set_mode(
            "GENERAL"
        )


        print()
        print(
            "Switched to GENERAL mode."
        )

        print()


    # ========================================================
    # CLEAR
    # ========================================================

    elif key == ord("c"):

        if processing:

            print(
                "Cannot clear while generating."
            )

            continue


        pipeline.clear()


        current_mode = "GENERAL"

        interviewer_question = ""


        locked_sign = None

        hand_release_count = 0

        stable_prediction = None

        stable_count = 0

        sequence = []

        prediction = "NO HAND"

        confidence = 0.0

        generated_sentence = ""


        print()
        print(
            "Communication cleared."
        )

        print(
            "Mode: GENERAL"
        )


    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print(
    "Recognition stopped."
)