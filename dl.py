import argparse
import difflib
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

VERSION = "2"

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
GROUPS = [
    (1, 5, "xor_mlp.py"), (6, 12, "ann_characters.py"),
    (13, 18, "autoencoder_xray.py"), (19, 24, "speech_commands.py"),
    (25, 25, "speech_to_text.py"), (26, 26, "image_classification.py"),
    (27, 31, "vehicle_detection.py"), (32, 32, "image_classification.py"),
    (33, 34, "financial_anomaly.py"), (35, 35, "stock_lstm.py"),
    (36, 37, "fraud_classification.py"), (38, 38, "financial_anomaly.py"),
    (39, 42, "rbm_digits.py"), (43, 48, "sentiment_lstm.py"),
]

def tokens(text):
    result = [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP]
    aliases = {"fraudulent": "fraud", "suspicious": "transaction",
               "spoken": "speech", "handwritten": "handwritten"}
    return result + [aliases[x] for x in result if x in aliases]

def choices(question):
    query = tokens(question)
    if not query:
        raise ValueError("enter a question")
    scores = []
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
        scores.append((score, item))
    scores.sort(reverse=True, key=lambda x: x[0])
    if scores[0][0] < .45:
        raise LookupError("no similar title found")
    return scores

def choose(question):
    score, item = choices(question)[0]
    return item, score

def similarity(a, b):
    x, y = tokens(a), tokens(b)
    return sum(max(difflib.SequenceMatcher(None, q, v).ratio() for v in y)
               for q in x) / max(1, len(x))

def full_questions(base):
    data = urlopen(f"{base.rstrip('/')}/DL%20LAB%20QUESTIONS.docx").read()
    root = ET.fromstring(zipfile.ZipFile(__import__("io").BytesIO(data)).read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    result = []
    for p in root.findall(".//w:p", ns):
        text = "".join(x.text or "" for x in p.findall(".//w:t", ns)).strip()
        if len(text) > 40 and not text.lower().startswith(("q.no", "question", "co no", "blooms level")):
            if text not in result:
                result.append(text)
    return result

def matching_question(question, questions, file):
    ranges = [range(a, b + 1) for a, b, f in GROUPS if f == file]
    pool = [questions[i - 1] for r in ranges for i in r if i <= len(questions)]
    return max(pool, key=lambda x: similarity(question, x))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="*")
    p.add_argument("--repo", default="Karthi-2007-git/dl")
    p.add_argument("--branch", default="main")
    p.add_argument("--base-url")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print("\n".join(f"{x} -> {y}" for x, y, _ in EXPERIMENTS))
        return
    try:
        question = " ".join(a.question) or sys.stdin.read()
        base = a.base_url or f"https://raw.githubusercontent.com/{a.repo}/{a.branch}"
        questions = full_questions(base)
        for score, (title, file, _) in choices(question):
            original = matching_question(question, questions, file)
            print(f"\nYour question: {question}")
            print(f"Matched question: {original}")
            print(f"Template: {title} ({score:.2f})")
            if input("Use this answer? [y/n] ").lower().startswith("y"):
                source = urlopen(f"{base.rstrip('/')}/{file}").read()
                header = f'QUESTION = {original!r}\nprint("Question:", QUESTION)\n\n'.encode()
                Path("output.py").write_bytes(header + source)
                break
        else:
            p.error("no answer selected")
    except (LookupError, ValueError, OSError) as e:
        p.error(str(e))
    print("saved: output.py")

if __name__ == "__main__":
    main()
