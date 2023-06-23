"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    06.05.2023
"""
import numpy as np
from keras.datasets import mnist

from utils import binarize, accuracy, show_confusion_matrix


class Perceptron:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = mnist.load_data()
        stdv = 1 / np.sqrt(784)
        self.weights = np.random.uniform(-stdv, stdv, size=(784, 10))
        self.biases = np.random.uniform(-stdv, stdv, size=10)
        self.accuracy = 0

    def learn(self, epoch_count, learning_rate):
        for epoch in range(epoch_count):
            for img in range(60000):
                stimulus = np.vectorize(binarize)(self.train_x[img]).flatten()
                expected_result = np.zeros(10)
                expected_result[self.train_y[img]] = 1

                real = self.stimulate(stimulus)
                error = expected_result - real
                stimulus = np.vstack(stimulus)
                self.weights += learning_rate * stimulus * error

                yield epoch, img, self.weights.copy()

        self.calculate_accuracy()

    def stimulate(self, stimulus):
        sums = np.dot(stimulus, self.weights)
        sums += self.biases
        max_sum_index = list(sums).index(np.max(sums))

        result = np.zeros(10)
        result[max_sum_index] = 1
        return result

    def calculate_accuracy(self):
        expected_history = []
        real_history = []

        for i in range(10000):
            stimulus = np.vectorize(binarize)(self.test_x[i]).flatten()
            expected_result = np.zeros(10)
            expected_result[self.test_y[i]] = 1
            real = self.stimulate(stimulus)

            expected_history.append(expected_result)
            real_history.append(real)

        self.accuracy = accuracy(expected_history, real_history)
        show_confusion_matrix(expected_history, real_history)
