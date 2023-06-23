"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    21.06.2023
"""


from ga_model import GAModel
from utils import *


class NNModel:
    def __init__(self):
        self.train_x, self.test_x, self.train_y, self.test_y = load_data()
        self.weights, self.biases = generate_weights()

    def feed_forward(self, x):
        return np.dot(x, self.weights) + self.biases

    def learn(self, data_count, gens, inds, cros, mut):
        ga_model = GAModel(gens, inds, cros, mut)

        for i in range(data_count):
            a = ga_model.process(self.train_x[i].flatten(), self.train_y[i])
            self.weights = a.chromosomes

        self.calculate_accuracy(len(self.test_x))

    def predict(self, image):
        x = image.flatten()
        y_pred = self.feed_forward(x)
        return np.argmax(y_pred)

    def calculate_accuracy(self, data_count):
        expected_history = []
        real_history = []

        for i in range(data_count):
            inputs = self.test_x[i].flatten()
            expected_history.append(self.test_y[i])
            real_history.append(self.feed_forward(inputs))

        show_confusion_matrix(expected_history, real_history)
