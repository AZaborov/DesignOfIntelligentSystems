"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    19.06.2023
"""

import sys
import gui
from PyQt5.QtWidgets import QApplication

#  Точка входа приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = gui.Window()
    sys.exit(app.exec_())
