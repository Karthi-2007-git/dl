import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = MLPClassifier((4,), activation="tanh", solver="lbfgs",
                      max_iter=2000, random_state=1).fit(X, y)
print("inputs:\n", X)
print("alarm predictions:", model.predict(X))
print("alarm accuracy:", model.score(X, y))
