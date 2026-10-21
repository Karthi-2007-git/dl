"""Tiny speech classifier. Put WAV files in data/speech/<word>/."""
import pathlib
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

X, y = [], []
for file in pathlib.Path("data/speech").rglob("*.wav"):
    audio, rate = librosa.load(file, sr=16000)
    X.append(librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=20).mean(1))
    y.append(file.parent.name)
if not X:
    raise FileNotFoundError("add WAV files under data/speech/<word>/")
labels = LabelEncoder().fit_transform(y)
model = MLPClassifier((32,), max_iter=300, random_state=1).fit(X, labels)
print("training accuracy:", model.score(X, labels))
