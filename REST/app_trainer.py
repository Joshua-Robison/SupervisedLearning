import pickle
import numpy as np

from util import get_data
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    X, Y = get_data()
    Ntrain = len(Y) // 4
    X_train, y_train = X[:Ntrain], Y[:Ntrain]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    X_test, y_test = X[Ntrain:], Y[Ntrain:]
    print(f'Testing accuracy: {model.score(X_test, y_test)}')

    filename = 'mymodel.pkl'
    pickle.dump(model, open(filename, 'wb'))

