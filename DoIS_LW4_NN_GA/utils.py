"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    21.06.2023
"""
import numpy as np
import pandas as pd
import seaborn as sn
import qimage2ndarray
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def pixmap2array(pixmap):
    qimg = pixmap.scaled(28, 28).toImage()
    qimg = qimage2ndarray.rgb_view(qimg)
    qimg = qimg[..., 0]

    return qimg.flatten()


def accuracy(expected, real):
    match_count = 0
    for i in range(len(expected)):
        if np.argmax(expected[i]) == np.argmax(real[i]):
            match_count += 1

    return match_count / len(real)


def show_confusion_matrix(expected_history, real_history):
    matrix = confusion_matrix([np.argmax(i) for i in real_history], [np.argmax(i) for i in expected_history])
    df_cm = pd.DataFrame(matrix, index=[i for i in "01"], columns=[i for i in "01"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='g', cmap="Reds")
    plt.show()


def show_plot(accuracies):
    plt.xticks(range(1, len(accuracies) + 1))
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.show()


def load_data():
    from keras.datasets import mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_filter = np.where((train_y == 0) | (train_y == 1))
    test_filter = np.where((test_y == 0) | (test_y == 1))
    train_x, train_y = train_x[train_filter], train_y[train_filter]
    test_x, test_y = test_x[test_filter], test_y[test_filter]

    train_x = (train_x.reshape((train_x.shape[0], 28, 28)) / 255.).round()
    test_x = (test_x.reshape((test_x.shape[0], 28, 28)) / 255.).round()
    train_y = make_binary(train_y)
    test_y = make_binary(test_y)

    return train_x, test_x, train_y, test_y


def make_binary(label_array, count_number=2):
    label_array = np.array(label_array)
    binary = np.zeros((label_array.shape[0], count_number))
    binary[np.arange(label_array.shape[0]), label_array] = 1

    return binary


def generate_weights():
    stdv = 1 / np.sqrt(784)
    weights = np.random.uniform(-stdv, stdv, size=(784, 2))
    biases = np.random.uniform(-stdv, stdv, size=2)

    return weights, biases
