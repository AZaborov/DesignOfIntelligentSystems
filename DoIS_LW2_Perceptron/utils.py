"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    06.05.2023
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


def binarize(num):
    return round(num / 255)


def accuracy(expected, real):
    match_count = 0
    for i in range(len(expected)):
        if np.argmax(expected[i]) == np.argmax(real[i]):
            match_count += 1

    return match_count / len(real)


def show_confusion_matrix(expected_history, real_history):
    matrix = confusion_matrix([np.argmax(i) for i in real_history], [np.argmax(i) for i in expected_history])
    df_cm = pd.DataFrame(matrix, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='g', cmap="Reds")
    plt.show()
