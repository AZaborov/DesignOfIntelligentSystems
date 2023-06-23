"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    19.06.2023
"""
import numpy as np


class Layer:
    def __init__(self, in_size, out_size):
        stdv = 1 / np.sqrt(in_size)
        self.weights = np.random.uniform(-stdv, stdv, size=(in_size, out_size))
        self.biases = np.random.uniform(-stdv, stdv, size=out_size)
        self.x = 0
        self.dx = 0

    def feed_forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.biases

    def descent(self, dz):
        self.dx = np.dot(dz, self.weights.T)
        return self.dx
