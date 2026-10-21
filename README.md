# Neural Network Practical Templates

These templates map the 48 questions in the supplied list to reusable program families.
They are practical starting points, not claims that one architecture fits every task.

## Setup

Create a virtual environment, then install only what you need:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install numpy scikit-learn matplotlib tensorflow pandas librosa soundfile opencv-python ultralytics nltk
```

Some packages are optional; install them only for the relevant practical. TensorFlow GPU support depends on your OS/CUDA setup.

## Find a template by question text

`dl_lab.py` matches approximate words from the DOCX question to the closest
experiment and prints the complete corresponding Python file. It does not require
third-party packages just to search.

```bash
python dl_lab.py "A smart door unlocks when exactly one sensor is active"
python dl_lab.py "find unusual stock price and volume patterns" --title
python dl_lab.py --list
```

For a GitHub copy, download the launcher and templates, then query locally:

```bash
curl -O https://raw.githubusercontent.com/USER/REPO/main/dl_lab.py
curl -O https://raw.githubusercontent.com/USER/REPO/main/xor_mlp.py
python dl_lab.py "XOR alarm with two sensors"
```

The launcher is intentionally separate from the experiments, so it can be
published with the template files and used from any checkout.

## Fetch from GitHub without the repository

Publish `dl.py` together with the experiment files. On a machine with
only Python and internet access, download just the launcher and fetch a matching
template directly into `output.py`:

```powershell
curl.exe -L https://raw.githubusercontent.com/USER/REPO/main/dl.py -o dl.py
python dl.py "a smart door unlocks when exactly one sensor is active"
python dl.py "detect unusual stock price and trading volume patterns"
python output.py
```

The launcher uses only Python's standard library. It downloads the selected
source file from GitHub, so the machine does not need the repository or the
matcher dependencies. The generated experiment may still need its own runtime
packages and dataset. Use `--list` to inspect searchable titles:

```powershell
python dl.py --repo USER/REPO --list
```

## Template map

| Questions | File / function | Change for the question |
|---|---|---|
| 1–5 | `xor_mlp.py` | Select XOR, compare perceptron/MLP, or compare hidden sizes |
| 6–12 | `ann_characters.py` | Select MNIST digits, EMNIST letters (external dataset), hidden size, or dropout |
| 13–18 | `autoencoder_xray.py` | Basic vs convolutional AE; denoising; reconstruction error for anomaly ranking |
| 19–24 | `speech_commands.py` | MFCC or log-Mel features; dense/MLP baseline; LSTM/CNN model |
| 25 | `speech_to_text.py` | Transcribe audio with a pretrained Whisper model, then clean transcript |
| 26, 32 | `image_classification.py` | Replace dataset folders/classes for vehicles or traffic signs |
| 27–31 | `vehicle_detection.py` | YOLO pretrained detection, video tracking/counting; custom helmet model requires labeled training data |
| 33, 38 | `financial_anomaly.py` | Autoencoder anomaly score and threshold |
| 34, 36, 37 | `fraud_classification.py` | ANN, class weights, Random Forest comparison; use labeled data |
| 35 | `stock_lstm.py` | Sequence forecast and flag large prediction errors |
| 39–42 | `rbm_digits.py` | RBM training, reconstruction, synthetic samples, learned features |
| 43–48 | `sentiment_lstm.py` | Replace dataset/labels for reviews, tweets, feedback, complaints; plot category distribution |

## Dataset expectations

- MNIST is loaded by TensorFlow/Keras.
- EMNIST letters, chest X-rays, speech datasets, traffic images, and financial datasets are not bundled. Supply a suitable, labeled dataset.
- Folder-based image classification expects `data/train/<class_name>/...` and `data/val/<class_name>/...`.
- Tabular examples expect a CSV path and the column names specified in each script.
- Do not use a random train/test split for time-series forecasting; preserve chronology.
- Medical, financial, and road-safety outputs are educational prototypes, not production decision systems.

## Run

Each script has a `main()` demonstration and/or command-line arguments. See the top of each file.
