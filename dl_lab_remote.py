"""Find a close experiment title, download its code, and save output.py."""
import argparse
import difflib
import re
import sys
from pathlib import Path
from urllib.request import urlopen

EXPERIMENTS = [
("XOR MLP alarm", "xor_mlp.py", "xor door sensor sensors exactly one different binary perceptron hidden neuron configuration accuracy"),
("ANN digit character recognition", "ann_characters.py", "ann handwritten handwritten digit character mnist letter register plate dropout hidden layer"),
("Autoencoder image anomaly", "autoencoder_xray.py", "autoencoder xray chest image reconstruction denoising compression unusual radiology"),
("MFCC speech commands", "speech_commands.py", "speech spoken spoken command word digit mfcc mel keyword cnn lstm noise"),
("Speech to text", "speech_to_text.py", "speech text audio lecture transcript whisper recording"),
("CNN image classification", "image_classification.py", "cnn vehicle car bus truck motorcycle traffic sign stop speed image"),
("YOLO vehicle detection", "vehicle_detection.py", "yolo detect locate track count vehicle traffic density congestion helmet rider video"),
("Financial anomaly detection", "financial_anomaly.py", "financial stock trading volume transaction unusual anomaly monitoring threshold alert"),
("Fraud classification", "fraud_classification.py", "fraud fraudulent transaction suspicious legitimate precision recall f1 imbalance class weight random forest"),
("LSTM stock forecast", "stock_lstm.py", "stock price time series lstm forecast prediction error unusual changes"),
("RBM digit features", "rbm_digits.py", "rbm handwritten digit reconstruct generate synthetic limited training feature extraction"),
("LSTM sentiment", "sentiment_lstm.py", "sentiment review movie product airline tweet feedback complaint category summarize classroom teaching laboratory positive negative neutral"),
]

STOP = set("a an the and or to of for in on with using wants need needs develop create create a python program based system model".split())

def tokens(text):
    result = [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP]
    aliases = {"fraudulent": "fraud", "suspicious": "transaction",
               "spoken": "speech", "handwritten": "handwritten"}
    return result + [aliases[x] for x in result if x in aliases]

def choose(question):
    query = tokens(question)
    if not query:
        raise ValueError("enter a question")
    best = (0, None)
    for item in EXPERIMENTS:
        vocab = tokens(item[0] + " " + item[2])
        exact = sum(q in vocab for q in query)
        fuzzy = sum(max(difflib.SequenceMatcher(None, q, v).ratio()
                        for v in vocab) for q in query) / len(query)
        score = exact * 2 + fuzzy
        joined = " ".join(query)
        for word in ("autoencoder", "rbm", "lstm", "yolo", "mfcc", "whisper", "xor"):
            if word in joined and word in vocab:
                score += 5
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
