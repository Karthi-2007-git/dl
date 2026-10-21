from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import mean_squared_error

data = load_digits()
X = (data.data > 8).astype("float32")
rbm = BernoulliRBM(32, n_iter=5, random_state=1).fit(X)
features = rbm.transform(X)
rebuild = rbm.gibbs(X)
print("feature shape:", features.shape)
print("reconstruction error:", mean_squared_error(X, rebuild))
