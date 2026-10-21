import sys
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

v = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "prices.csv")["Close"].dropna().to_numpy()
n = 10
X = np.array([v[i-n:i] for i in range(n, len(v))])
y = v[n:]
cut = int(len(X) * .8)
model = keras.Sequential([layers.Input((n,)), layers.Dense(16, "relu"), layers.Dense(1)])
model.compile("adam", "mse")
model.fit(X[:cut], y[:cut], epochs=5, verbose=0)
error = abs(y[cut:] - model.predict(X[cut:], verbose=0).ravel())
print("mean test error:", error.mean())
