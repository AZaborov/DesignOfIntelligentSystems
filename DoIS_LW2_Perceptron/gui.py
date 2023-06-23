"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    06.05.2023
"""

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtWidgets import QMainWindow
from matplotlib import pyplot as plt

from perceptron import Perceptron
from utils import pixmap2array


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('gui.ui', self)
        self.show()
        self.perceptron = Perceptron()
        self.epochs = []

        canvas = QPixmap(280, 280)
        canvas.fill(Qt.black)
        self.input_label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.clear_button.clicked.connect(self.clear_button_clicked)
        self.learn_button.clicked.connect(self.learn_button_clicked)
        self.identify_button.clicked.connect(self.identify_button_clicked)
        self.show_weights_before_button.clicked.connect(self.show_weights_before_button_clicked)
        self.show_weights_after_button.clicked.connect(self.show_weights_after_button_clicked)

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        lx = self.input_label.x()
        ly = self.input_label.y()
        painter = QPainter(self.input_label.pixmap())
        painter.setPen(QPen(Qt.white, 20))
        painter.drawLine(self.last_x - lx, self.last_y - ly, e.x() - lx, e.y() - ly)
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def clear_button_clicked(self):
        canvas = QPixmap(280, 280)
        canvas.fill(Qt.black)
        self.input_label.setPixmap(canvas)
        self.status_label.setText("")

    def learn_button_clicked(self):
        epoch_count = self.epoch_count_spin_box.value()
        learning_rate = self.learning_rate_spin_box.value()
        self.epochs = []

        for e, i, w in self.perceptron.learn(epoch_count, learning_rate):
            self.learning_progress_bar.setValue(i)
            self.status_label.setText(f"Идёт обучение (Эпоха {e + 1} из {epoch_count})")
            self.epochs.append(w)

        self.learning_progress_bar.setValue(0)
        self.status_label.setText(f"Обучение завершено, точность: {round(self.perceptron.accuracy, 2)}")

    def identify_button_clicked(self):
        result = self.perceptron.stimulate(pixmap2array(self.input_label.pixmap()))
        self.status_label.setText(f"Я думаю, это {np.argmax(result)}")

    def show_weights_before_button_clicked(self):
        if not self.epochs:
            return
        else:
            self.show_weights(self.epochs[0].T)

    def show_weights_after_button_clicked(self):
        if not self.epochs:
            return
        else:
            self.show_weights(self.epochs[len(self.epochs) - 1].T)

    def show_weights(self, weights):
        fig = plt.figure()
        for i in range(weights.shape[0]):
            fig.add_subplot(2, 5, i + 1).set_title(i)
            fig.tight_layout()
            plt.imshow(weights[i].reshape((28, 28)), cmap='RdYlGn')
            plt.axis('off')

        plt.savefig("epoch_temp.png")
        self.epoch_label.setPixmap(QPixmap(QImage("epoch_temp.png")))
