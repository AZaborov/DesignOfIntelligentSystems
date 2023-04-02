import random

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from genetic_algorithm import *


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.ui = uic.loadUi('gui.ui')
        self.ui.show()

        # Подписки на события
        #self.ui.mutation_probabilty_spin_box.clicked.connect(self.load_image_button_clicked)

    def load_image_button_clicked(self):
       print()
