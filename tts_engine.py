"""
tts_engine.py
--------------
Small text-to-speech wrapper. Uses pyttsx3 (fully offline). If it's
not installed or no audio device is available, falls back to just
printing the text so the rest of the app keeps working (spec section 33:
fail gracefully, don't crash the whole app because TTS is unavailable).
"""


class TTSEngine:
    def __init__(self):
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
        except Exception as e:
            print(f"[tts_engine] TTS unavailable ({e}). Will print instead of speaking.")

    def speak(self, text: str):
        if not text:
            return
        if self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception as e:
                print(f"[tts_engine] Speak failed ({e}); printing instead.")
        print(f"🔊 {text}")
