"""Tiny CNN example: classify CIFAR-10 images."""
from tensorflow import keras
from tensorflow.keras import layers

(x, y), (xt, yt) = keras.datasets.cifar10.load_data()
model = keras.Sequential([
    layers.Rescaling(1 / 255, input_shape=(32, 32, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=2, batch_size=256, verbose=1)
print("test accuracy:", model.evaluate(xt, yt, verbose=0)[1])
