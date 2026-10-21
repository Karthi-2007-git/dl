"""Tiny speech-to-text example: python speech_to_text.py audio.wav."""
import sys
import whisper

file = sys.argv[1] if len(sys.argv) > 1 else "audio.wav"
text = whisper.load_model("base").transcribe(file)["text"]
print(" ".join(text.split()))
