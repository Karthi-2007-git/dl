import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

file = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
X = pd.read_csv(file).select_dtypes("number").dropna().to_numpy()
X = StandardScaler().fit_transform(X)
model = MLPRegressor((16,), max_iter=300, random_state=1).fit(X, X)
error = ((X - model.predict(X)) ** 2).mean(1)
limit = np.quantile(error, .99)
print("threshold:", limit)
print("anomalies:", np.flatnonzero(error > limit))
