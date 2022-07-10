"""This file runs the python machine learning application."""
import sys
import pathlib
import requests
import numpy as np
import matplotlib.pyplot as plt

path = pathlib.Path(__file__).parent.absolute()
sys.path.append(f"{path}/../")
from util import get_data


if __name__ == "__main__":
    X, Y = get_data()
    N = len(Y)
    while True:
        i = np.random.choice(N)
        try:
            response = requests.post(
                "http://localhost:8080/predict", data={"input": X[i]}
            )

            j = response.json()
            plt.imshow(X[i].reshape(28, 28), cmap="gray")
            plt.title("Target: %d, Prediction: %d" % (Y[i], j["prediction"]))
            plt.show()
        except requests.exceptions.ConnectionError:
            print("ERROR")

        response = input("Continue? (Y/N)\n")
        if response in ("n", "N"):
            break
