"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    19.06.2023
"""
from keras.datasets import mnist

from activation_functions import *
from layer import Layer
from utils import *


class Model:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = mnist.load_data()
        self.train_x = (self.train_x.reshape((self.train_x.shape[0], 28, 28)) / 255.).round()
        self.test_x = (self.test_x.reshape((self.test_x.shape[0], 28, 28)) / 255.).round()
        self.train_y = make_binary(self.train_y)
        self.test_y = make_binary(self.test_y)

        self.layer1_net = None
        self.layer2_net = None
        self.layer3_net = None

        self.layer1_out = ReLu()
        self.layer2_out = ReLu()
        self.layer3_out = Sigmoid()

        self.accuracies = []
        self.expected_history = []
        self.real_history = []

    def learn(self, epoch_count, learning_rate, layer2_count, layer3_count):
        self.layer1_net = Layer(784, layer2_count)
        self.layer2_net = Layer(layer2_count, layer3_count)
        self.layer3_net = Layer(layer3_count, 10)
        self.accuracies = []

        for epoch in range(epoch_count):
            self.calculate_accuracy()

            for i in range(60000):
                inputs = self.train_x[i].flatten()

                expected = self.train_y[i]
                real = self.feed_forward(inputs)

                error = expected - real
                self.descent(error, learning_rate)

        show_confusion_matrix(self.expected_history, self.real_history)
        show_plot(self.accuracies)

    def feed_forward(self, x):
        x = x.reshape((1, -1))

        net1 = self.layer1_net.feed_forward(x)
        out1 = self.layer1_out.feed_forward(net1)
        net2 = self.layer2_net.feed_forward(out1)
        out2 = self.layer2_out.feed_forward(net2)
        net3 = self.layer3_net.feed_forward(out2)
        out3 = self.layer3_out.feed_forward(net3)

        return out3

    def descent(self, error, learning_rate):
        dE_dw1 = self.layer3_out.descent(error)
        dE_dw2 = self.layer3_net.descent(dE_dw1)
        dE_dw2 = self.layer2_out.descent(dE_dw2)
        dE_dw3 = self.layer2_net.descent(dE_dw2)
        dE_dw3 = self.layer1_out.descent(dE_dw3)

        self.layer1_net.weights += dE_dw3 * self.layer1_net.x.T * learning_rate
        self.layer2_net.weights += dE_dw2 * self.layer2_net.x.T * learning_rate
        self.layer3_net.weights += dE_dw1 * self.layer3_net.x.T * learning_rate

    def calculate_accuracy(self):
        expected_history = []
        real_history = []

        for i in range(10000):
            inputs = self.test_x[i].flatten()
            expected_history.append(self.test_y[i])
            real_history.append(self.feed_forward(inputs))

        self.accuracies.append(accuracy(expected_history, real_history))
        self.expected_history = expected_history.copy()
        self.real_history = real_history.copy()
