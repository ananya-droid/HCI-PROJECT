"""
live_sign_test.py
-------------------
The full end-to-end pipeline, live from your webcam:

    Camera -> MediaPipe -> LSTM -> Sign Tokens -> Communication Engine
           -> Conversation Context -> AI/NLP -> Natural English
           -> Intent (metadata) -> Text-to-Speech

Usage:
    python live_sign_test.py                     # general mode
    python live_sign_test.py --mode interview     # job-interview demo

Controls:
    c      -> clear current tokens (start a fresh message)
    ENTER  -> generate sentence from current tokens + speak it
    q      -> quit
"""

import argparse

import cv2

import config
from hand_landmarkers import HandLandmarkExtractor
from gesture_recognizer import GestureRecognizer
from communication_engine import CommunicationEngine
from conversation_context import ConversationContext
from sentence_generator import generate_sentence
from intent_engine import detect_intent
from tts_engine import TTSEngine

INTERVIEW_QUESTIONS = [
    "Tell me about yourself.",
    "What skills do you have?",
    "Why do you want this position?",
]


def run(mode: str, use_ai: bool):
    extractor = HandLandmarkExtractor()
    recognizer = GestureRecognizer()
    comm = CommunicationEngine()
    context = ConversationContext(mode=mode)
    tts = TTSEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    question_idx = 0
    if mode == config.MODE_INTERVIEW and INTERVIEW_QUESTIONS:
        q = INTERVIEW_QUESTIONS[0]
        print(f"\nInterviewer: {q}")
        context.add_message("Interviewer", q)

    print("\nControls: 'c' clear tokens | ENTER generate+speak | 'q' quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[live_sign_test] Failed to read webcam frame.")
                break

            landmarks, points = extractor.extract(frame)
            token = recognizer.update(landmarks)
            if token:
                comm.add_token(token)
                print(f"[live_sign_test] Recognized token: {token}  (tokens so far: {comm.get_tokens()})")

            display = frame.copy()
            if points:
                h, w, _ = display.shape
                for (x, y) in points:
                    cv2.circle(display, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)

            status = recognizer.status_label()
            cv2.putText(display, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Tokens: {' '.join(comm.get_tokens())}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "c=clear  ENTER=speak  q=quit", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("live_sign_test.py", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                comm.clear()
                print("[live_sign_test] Tokens cleared.")
            elif key in (13, 10):  # ENTER
                tokens = comm.get_tokens()
                if not tokens:
                    print("[live_sign_test] No tokens to generate a sentence from.")
                    continue

                sentence, used_ai = generate_sentence(tokens, context=context, use_ai=use_ai)
                intent = detect_intent(tokens)

                print(f"\nRecognized tokens: {' '.join(tokens)}")
                print(f"Generated ({'AI' if used_ai else 'offline'}): {sentence}")
                print(f"Intent (metadata): {intent}\n")

                context.add_message("You", sentence)
                tts.speak(sentence)
                comm.clear()

                if mode == config.MODE_INTERVIEW:
                    question_idx += 1
                    if question_idx < len(INTERVIEW_QUESTIONS):
                        q = INTERVIEW_QUESTIONS[question_idx]
                        print(f"Interviewer: {q}\n")
                        context.add_message("Interviewer", q)
                    else:
                        print("[live_sign_test] Interview demo questions complete.")
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full end-to-end sign-to-speech demo.")
    parser.add_argument("--mode", choices=[config.MODE_GENERAL, config.MODE_INTERVIEW],
                         default=config.MODE_GENERAL)
    parser.add_argument("--no-ai", action="store_true",
                         help="Force offline deterministic sentence generation")
    args = parser.parse_args()

    run(mode=args.mode, use_ai=not args.no_ai)
