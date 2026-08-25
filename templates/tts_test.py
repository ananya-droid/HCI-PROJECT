import pyttsx3

print("Starting TTS...")

tts = pyttsx3.init()

tts.setProperty("rate", 150)

tts.say("Hello. This is a test.")

tts.runAndWait()

print("TTS finished.")