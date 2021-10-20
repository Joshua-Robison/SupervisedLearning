"""
This file contains data loading utilities.
"""
import numpy as np
import pandas as pd


def get_data(limit=None):
    print('Reading in and transforming data...')
    df = pd.read_csv('../data/train.csv')
    data = df.to_numpy()
    np.random.shuffle(data)
    X = data[:,1:] / 255.0
    Y = data[:,0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
        
    return X, Y


def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])
    Y = np.array([0] * 100 + [1] * 100)

    return X, Y


def get_donut(n_samples: int=200, r_inner: int=5, r_outer: int=10):
    N = n_samples // 2
    R1 = np.random.randn(N) + r_inner
    theta = 2 * np.pi * np.random.random(N)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    R2 = np.random.randn(N) + r_outer
    theta = 2 * np.pi * np.random.random(N)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*(N) + [1]*(N))
    
    return X, Y


if __name__ == '__main__':
    pass
