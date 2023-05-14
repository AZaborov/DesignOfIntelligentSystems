"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    06.05.2023
"""
import numpy as np
import qimage2ndarray


def pixmap2array(pixmap):
    qimg = pixmap.scaled(28, 28).toImage()
    qimg = qimage2ndarray.rgb_view(qimg)
    qimg = qimg[..., 0]

    return qimg.flatten()


def binarize(num):
    return round(num / 255)


def accuracy(expected, real):
    expected = np.array(expected)
    real = np.array(real)

    return (expected == real).sum() / real.size
