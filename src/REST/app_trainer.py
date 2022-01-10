"""This file trains a Random Forest Classifier to use in a RESTful python application."""
import os
import pickle
from src.util import get_data
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    X, Y = get_data()
    Ntrain = len(Y) // 4
    X_train, y_train = X[:Ntrain], Y[:Ntrain]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    X_test, y_test = X[Ntrain:], Y[Ntrain:]
    print(f'Random Forest Accuracy: {model.score(X_test, y_test)}')

    path = os.path.dirname(__file__)
    filename = path + '/mymodel.pkl'
    pickle.dump(model, open(filename, 'wb'))
