"""This file trains a Random Forest Classifier to use in a RESTful python application."""
import sys
import pickle
import pathlib
from sklearn.ensemble import RandomForestClassifier

path = pathlib.Path(__file__).parent.absolute()
sys.path.append(f"{path}/../")
from util import get_data


if __name__ == "__main__":
    X, Y = get_data()
    Ntrain = len(Y) // 4
    X_train, y_train = X[:Ntrain], Y[:Ntrain]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    X_test, y_test = X[Ntrain:], Y[Ntrain:]
    print(f"Random Forest Accuracy: {model.score(X_test, y_test)}")
    file = f"{path}/mymodel.pkl"
    pickle.dump(model, open(file, "wb"))
