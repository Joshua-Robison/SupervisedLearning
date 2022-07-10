"""This file defines a Naive Bayesian Classifier."""
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


class NaiveBayes(object):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, smoothing: float = 10e-3) -> None:
        self.gaussians = dict()
        self.priors = dict()
        labels = set(y)
        for c in labels:
            current_x = X[y == c]
            self.priors[c] = len(y[y == c]) / len(y)
            self.gaussians[c] = {
                "mean": current_x.mean(axis=0),
                "var": current_x.var(axis=0) + smoothing,
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, _ = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g["mean"], g["var"]
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])

        return np.argmax(P, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict(X)
        return np.mean(p == y)


if __name__ == "__main__":
    pass
