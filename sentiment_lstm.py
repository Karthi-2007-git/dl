"""Tiny sentiment LSTM: uses the built-in IMDB review dataset."""
from tensorflow import keras
from tensorflow.keras import layers

(x, y), (xt, yt) = keras.datasets.imdb.load_data(num_words=5000)
x = keras.utils.pad_sequences(x, maxlen=120)
xt = keras.utils.pad_sequences(xt, maxlen=120)
model = keras.Sequential([
    layers.Embedding(5000, 16),
    layers.LSTM(16),
    layers.Dense(1, activation="sigmoid"),
])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=2, batch_size=256, verbose=1)
print("test accuracy:", model.evaluate(xt, yt, verbose=0)[1])
