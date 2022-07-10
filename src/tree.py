"""This file defines a Decision Tree Classifier."""
import numpy as np


def entropy(y: np.ndarray):
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0

    p1 = s1 / N
    p0 = 1 - p1

    return -p0 * np.log2(p0) - p1 * np.log2(p1)


class TreeNode(object):
    def __init__(self, depth: int = 0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(y) == 1 or len(set(y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = y[0]
        else:
            D = X.shape[1]
            cols = range(D)
            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, y, col)
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split

            if max_ig == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(y.mean())
            else:
                self.col = best_col
                self.split = best_split
                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(y[X[:, best_col] < self.split].mean()),
                        np.round(y[X[:, best_col] >= self.split].mean()),
                    ]
                else:
                    left_idx = X[:, best_col] < best_split
                    X_left = X[left_idx]
                    y_left = y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(X_left, y_left)
                    right_idx = X[:, best_col] >= best_split
                    X_right = X[right_idx]
                    y_right = y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(X_right, y_right)

    def find_split(self, X: np.ndarray, y: np.ndarray, col: int):
        max_ig = 0
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        y_values = X[sort_idx]
        y_values = y[sort_idx]
        best_split = None
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        for i in boundaries:
            split = (x_values[i] + x_values[i + 1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split

        return max_ig, best_split

    def information_gain(self, x: np.ndarray, y: np.ndarray, split: float):
        N = len(y)
        y0 = y[x < split]
        y1 = y[x >= split]
        if len(y0) == 0 or len(y0) == N:
            return 0

        p0 = len(y0) / N
        p1 = 1 - p0

        return entropy(y) - p0 * entropy(y0) - p1 * entropy(y1)

    def predict_one(self, x: np.ndarray):
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction

        return p

    def predict(self, X: np.ndarray):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])

        return P


class DecisionTree(object):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.root.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        p = self.predict(X)
        return np.mean(p == y)


if __name__ == "__main__":
    pass
