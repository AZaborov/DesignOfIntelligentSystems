"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    19.06.2023
"""
import numpy as np


class Sigmoid:
    def __init__(self):
        self.out_val = 0

    def feed_forward(self, in_val):
        self.out_val = 1 / (1 + np.exp(-in_val))
        return self.out_val

    def descent(self, grad_output):
        return self.out_val * (1 - self.out_val) * grad_output


class ReLu:
    def __init__(self):
        self.in_val = 0

    def feed_forward(self, in_val):
        self.in_val = in_val
        return np.maximum(0, in_val)

    def descent(self, dz):
        dz[self.in_val < 0] = 0
        return dz
