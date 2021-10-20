import requests
import numpy as np
import matplotlib.pyplot as plt
from util import get_data

X, Y = get_data()
N = len(Y)
while True:
    i = np.random.choice(N)
    r = requests.post('http://localhost:8888/predict', data={'input': X[i]})
    j = r.json()

    print(j)
    print('Target:', Y[i])

    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.show()

    response = input('Continue? (Y/N)\n')
    if response in ('n', 'N'):
        break

