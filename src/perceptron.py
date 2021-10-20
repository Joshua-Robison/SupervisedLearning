"""
This file defines a Perceptron Classifier.
"""
import numpy as np


class Perceptron(object):
    
    def __init__(self, lr: float=0.01, max_iters: int=1000):
        self.lr = lr
        self.iters = max_iters
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0
        N = len(y)
        costs = []
        for _ in range(self.iters):
            y_hat = self.predict(X)
            errors = np.nonzero(y != y_hat)[0]
            if len(errors) == 0:
                break
            
            i = np.random.choice(errors)
            self.w += self.lr * y[i] * X[i]
            self.b += self.lr * y[i]
            c = len(errors) / N
            costs.append(c)
            
        plt.plot(costs)
        plt.show()
        
    def predict(self, X: np.ndarray):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X: np.ndarray, y: np.ndarray):
        p = self.predict(X)
        
        return np.mean(p == y)


if __name__ == '__main__':
    pass
