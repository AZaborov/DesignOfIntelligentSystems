"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    21.06.2023
"""

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMainWindow

from nn_model import NNModel
from utils import pixmap2array


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('gui.ui', self)
        self.show()
        self.nn_model = NNModel()

        canvas = QPixmap(280, 280)
        canvas.fill(Qt.black)
        self.input_label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.clear_button.clicked.connect(self.clear_button_clicked)
        self.learn_button.clicked.connect(self.learn_button_clicked)
        self.identify_button.clicked.connect(self.identify_button_clicked)

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
        self.nn_model.learn(self.data_count_spin_box.value(),
                            self.gens_count_spin_box.value(),
                            self.individs_count_spin_box.value(),
                            self.crossover_prob_spin_box.value(),
                            self.mutation_prob_spin_box.value())

        # self.status_label.setText(f"Обучение завершено, точность: {round(self.nn_model.accuracies[-1], 2)}")
        self.status_label.setText(f"Обучение завершено, точность")

    def identify_button_clicked(self):
        result = self.nn_model.predict(pixmap2array(self.input_label.pixmap()))
        self.status_label.setText(f"Я думаю, это {np.argmax(result)}")
