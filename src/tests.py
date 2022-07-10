"""This file defines some example tests using the supervised machine learning models."""
import matplotlib.pyplot as plt
from perceptron import Perceptron
from nb import NaiveBayes
from bayes import Bayes
from knn import KNN
from tree import DecisionTree
from util import gen_data, get_data, get_xor, get_donut


def test_perceptron():
    X, y = gen_data(500)
    N_train = 300
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    model = Perceptron()
    model.fit(X_train, y_train)
    print("\n\nPerceptron Classifier...")
    print(f"Training accuracy: {model.score(X_train, y_train)}")
    print(f"Testing accuracy: {model.score(X_test, y_test)}")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
    plt.title("Perceptron Classifier")
    plt.show()


def test_naive_bayes():
    X, y = get_data(1000)
    N_train = 500
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    model = NaiveBayes()
    model.fit(X_train, y_train)
    print("\n\nNaive Bayes Classifier...")
    print(f"Training accuracy: {model.score(X_train, y_train)}")
    print(f"Testing accuracy: {model.score(X_test, y_test)}")


def test_bayes():
    X, y = get_data(1000)
    N_train = 500
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    model = Bayes()
    model.fit(X_train, y_train)
    print("\n\nBayes Classifier...")
    print(f"Training accuracy: {model.score(X_train, y_train)}")
    print(f"Testing accuracy: {model.score(X_test, y_test)}")


def test_knn():
    X, y = get_data(500)
    N_train = 400
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]
    print("\n\nK-Nearest Neighbors Classifier...")
    for k in (1, 2, 3, 4, 5):
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        print(f"\nK-Nearest Neighbor Model: k = {k}")
        print(f"Training accuracy: {knn.score(X_train, y_train)}")
        print(f"Testing accuracy: {knn.score(X_test, y_test)}")

    X, y = get_donut()
    X_train, y_train = X[::2], y[::2]
    X_test, y_test = X[1::2], y[1::2]
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=100,
        marker="o",
        c=y_train,
        alpha=0.5,
        label="Training Data",
        cmap="jet",
    )

    plt.scatter(
        X_test[:, 0], X_test[:, 1], s=100, marker="s", c=y_test, label="Testing Data"
    )

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    plt.title(f"KNN Accuracy: {knn.score(X_test, y_test)}")
    plt.legend()
    plt.show()


def test_decision_tree():
    X, y = get_xor()
    X_train, y_train = X[::2], y[::2]
    X_test, y_test = X[1::2], y[1::2]
    print("\n\nDecision Tree Classifier...")
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=100,
        marker="o",
        c=y_train,
        alpha=0.5,
        label="Training Data",
        cmap="jet",
    )

    plt.scatter(
        X_test[:, 0], X_test[:, 1], s=100, marker="s", c=y_test, label="Testing Data"
    )

    tree = DecisionTree()
    tree.fit(X_train, y_train)
    plt.title(f"Decision Tree Accuracy: {tree.score(X_test, y_test)}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_perceptron()
    test_naive_bayes()
    test_bayes()
    test_knn()
    test_decision_tree()
