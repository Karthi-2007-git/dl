"""Find a close experiment title, download its code, and save output.py."""
import argparse
import difflib
import re
import sys
from pathlib import Path
from urllib.request import urlopen

EXPERIMENTS = [
("XOR MLP alarm", "xor_mlp.py", "door sensor binary perceptron"),
("ANN digit character recognition", "ann_characters.py", "handwritten mnist letter plate"),
("Autoencoder image anomaly", "autoencoder_xray.py", "xray reconstruct denoise compress"),
("MFCC speech commands", "speech_commands.py", "spoken word digit audio"),
("Speech to text", "speech_to_text.py", "whisper lecture transcript audio"),
("CNN image classification", "image_classification.py", "vehicle traffic sign image"),
("YOLO vehicle detection", "vehicle_detection.py", "track count traffic helmet video"),
("Financial anomaly detection", "financial_anomaly.py", "stock volume transaction unusual"),
("Fraud classification", "fraud_classification.py", "fraud precision recall imbalance"),
("LSTM stock forecast", "stock_lstm.py", "stock price time series prediction"),
("RBM digit features", "rbm_digits.py", "handwritten generate reconstruct"),
("LSTM sentiment", "sentiment_lstm.py", "review tweet complaint positive negative"),
]

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def choose(question):
    query = tokens(question)
    if not query:
        raise ValueError("enter a question")
    best = (0, None)
    for item in EXPERIMENTS:
        vocab = tokens(item[0] + " " + item[2])
        score = sum(max(difflib.SequenceMatcher(None, q, v).ratio()
                        for v in vocab) for q in query) / len(query)
        if score > best[0]:
            best = score, item
    if best[0] < .45:
        raise LookupError("no similar title found")
    return best[1], best[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--repo")
    p.add_argument("--branch", default="main")
    p.add_argument("--base-url")
    p.add_argument("-o", "--output", default="output.py")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print("\n".join(f"{x} -> {y}" for x, y, _ in EXPERIMENTS))
        return
    if not a.repo and not a.base_url:
        p.error("use --repo OWNER/REPO")
    try:
        (title, file, _), score = choose(" ".join(a.question) or sys.stdin.read())
        base = a.base_url or f"https://raw.githubusercontent.com/{a.repo}/{a.branch}"
        Path(a.output).write_bytes(urlopen(f"{base.rstrip('/')}/{file}").read())
    except (LookupError, ValueError, OSError) as e:
        p.error(str(e))
    print(f"matched: {title}\nestimate: {score:.2f}\nsaved: {a.output}")

if __name__ == "__main__":
    main()
