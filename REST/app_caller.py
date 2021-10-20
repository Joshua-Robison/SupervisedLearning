"""
This file runs the python machine learning application.
"""
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, '../src')
from util import get_data


X, Y = get_data()
N = len(Y)
while True:
    i = np.random.choice(N)
    try:
        r = requests.post('http://localhost:8080/predict', data={'input': X[i]})
        print('RESPONSE:')
        print(r.content)
        j = r.json()
        print(j)
        print('Target:', Y[i])

        plt.imshow(X[i].reshape(28,28), cmap='gray')
        plt.title('Target: %d, Prediction: %d' % (Y[i], j['prediction']))
        plt.show()
    except requests.exceptions.ConnectionError:
        print('ERROR')

    response = input('Continue? (Y/N)\n')
    if response in ('n', 'N'):
        break
