"""Tiny ANN example: handwritten digit recognition."""
from tensorflow import keras
from tensorflow.keras import layers

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x / 255.0, xt / 255.0
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=2, batch_size=256, verbose=1)
print("test accuracy:", model.evaluate(xt, yt, verbose=0)[1])
