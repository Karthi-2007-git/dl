import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

(x, _), (xt, _) = keras.datasets.mnist.load_data()
x, xt = x[..., None] / 255.0, xt[..., None] / 255.0
inp = keras.Input((28, 28, 1))
z = layers.Flatten()(inp)
z = layers.Dense(32, activation="relu")(z)
out = layers.Reshape((28, 28, 1))(layers.Dense(784, activation="sigmoid")(z))
model = keras.Model(inp, out)
model.compile("adam", "mse")
model.fit(x, x, epochs=2, batch_size=256, verbose=1)
error = np.mean((xt - model.predict(xt, verbose=0)) ** 2, axis=(1, 2, 3))
print("mean reconstruction error:", error.mean())
