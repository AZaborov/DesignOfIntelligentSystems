"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    06.05.2023
"""
import numpy as np
import pandas as pd
import seaborn as sn
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import binarize, accuracy


class Perceptron:
    def __init__(self):
        (self.train_x, self.train_y), (_, _) = mnist.load_data()
        stdv = 1 / np.sqrt(784)
        self.weights = np.random.uniform(-stdv, stdv, size=(784, 10))
        self.biases = np.random.uniform(-stdv, stdv, size=10)
        self.accuracy = 0

    def learn(self, epoch_count, learning_rate):
        expected_history = []
        real_history = []

        for epoch in range(epoch_count):
            stimulus = np.vectorize(binarize)(self.train_x[epoch % 60000]).flatten()

            expected_result = np.zeros(10)
            expected_result[self.train_y[epoch % 60000]] = 1

            real = self.stimulate(stimulus)
            error = expected_result - real
            stimulus = np.vstack(stimulus)
            self.weights += learning_rate * stimulus * error

            expected_history.append(real)
            real_history.append(expected_result)

            yield epoch, self.weights.copy()

        self.accuracy = accuracy(expected_history, real_history)

        matrix = confusion_matrix([np.argmax(i) for i in real_history], [np.argmax(i) for i in expected_history])
        df_cm = pd.DataFrame(matrix, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='g', cmap="Reds")
        plt.show()

    def stimulate(self, stimulus):
        sums = np.dot(stimulus, self.weights)
        sums += self.biases
        max_sum_index = list(sums).index(np.max(sums))

        result = np.zeros(10)
        result[max_sum_index] = 1
        return result
