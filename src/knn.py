"""
This file defines a K-Nearest Neighbors Classifier.
"""
import numpy as np
from sortedcontainers import SortedList


class KNN(object):
    
    def __init__(self, k: int):
        self.k = k
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        
    def predict(self, X: np.ndarray):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                dist = np.linalg.norm(x - xt)
                if len(sl) < self.k:
                    sl.add((dist, self.y[j]))
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add((dist, self.y[j]))
                        
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
                
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
                    
            y[i] = max_votes_class
            
        return y
    
    def score(self, X: np.ndarray, y: np.ndarray):
        p = self.predict(X)
        
        return np.mean(p == y)


if __name__ == '__main__':
    pass
