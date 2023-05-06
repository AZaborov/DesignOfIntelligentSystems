"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    02.04.2023
"""

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QHeaderView

import ga_graph
import genetic_algorithm as ga


class Window(QMainWindow):
    """
    Форма для настройки графа и ГА
    """
    def __init__(self):
        """
        Инициализация класса, полей и событий формы
        """
        super(Window, self).__init__()
        self.ui = uic.loadUi('gui.ui')
        self.ui.show()
        self.graph = []
        self.step = []  # Данные для пошагового режима работы алгоритма
        self.cycle = []  # Данные для циклического режима работы алгоритма
        self.current_gen = 0  # Текущее выбранное поколение для пошагового режима работы алгоритма

        # Подписки на события
        self.ui.generate_graph_button.clicked.connect(self.generate_graph)
        self.ui.find_shortest_path_button.clicked.connect(self.find_shortest_path)
        self.ui.prev_table_data_button.clicked.connect(self.prev_table_data)
        self.ui.next_table_data_button.clicked.connect(self.next_table_data)
        self.ui.step_mode_radio_button.clicked.connect(self.step_mode_radio)
        self.ui.cycle_mode_radio_button.clicked.connect(self.cycle_mode_radio)

    def generate_graph(self):
        """
        Сгенерировать граф с данными для ГА
        :return:
        """
        if self.ui.finish_node_spin_box.value() >= self.ui.graph_size_spin_box.value() or \
                self.ui.start_node_spin_box.value() >= self.ui.graph_size_spin_box.value():
            return

        self.graph = ga_graph.generate_graph(self.ui.graph_size_spin_box.value(),
                                             self.ui.graph_fully_connected_check_box.isChecked(),
                                             self.ui.min_val_spin_box.value(),
                                             self.ui.max_val_spin_box.value())

        ga_graph.show_graph(self.graph)

    def find_shortest_path(self):
        """
        Запустить процесс эволюции для решения задачи
        нахождения кратчайшего пути между выбранными вершинами графа
        :return:
        """
        if not self.graph:
            return

        if self.ui.finish_node_spin_box.value() >= self.ui.graph_size_spin_box.value() or \
                self.ui.start_node_spin_box.value() >= self.ui.graph_size_spin_box.value():
            return

        result, self.step, self.cycle = ga.process(self.graph,
                                                   self.ui.generations_count_spin_box.value(),
                                                   self.ui.population_count_spin_box.value(),
                                                   self.ui.start_node_spin_box.value(),
                                                   self.ui.finish_node_spin_box.value(),
                                                   self.ui.mutation_probabilty_spin_box.value() / 100)

        chromosome = result.get_chromosome().copy()
        chromosome.append(self.ui.finish_node_spin_box.value())
        chromosome.insert(0, self.ui.start_node_spin_box.value())
        chromosome = list(dict.fromkeys(filter(lambda a: a != -1, chromosome)).keys())
        self.ui.result_label.setText("{} {}".format(result.get_fitness_value(), chromosome))
        self.ui.table.clear()

        if self.ui.step_mode_radio_button.isChecked():
            self.fill_step_table()
        elif self.ui.cycle_mode_radio_button.isChecked():
            self.fill_cycle_table()

    def fill_step_table(self):
        """
        Заполнить таблицу с данными при пошаговом режиме работы алгоритма
        :return:
        """
        self.ui.table.setColumnCount(4)
        self.ui.table.setRowCount(len(self.step[0][0]))

        header = self.ui.table.horizontalHeader()
        for i in range(4):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self.ui.table.setHorizontalHeaderLabels(("Изначальные хромосомы",
                                                 "После скрещивания",
                                                 "После селекции",
                                                 "После мутации"))

        row_headers = []
        for i in range(len(self.step[0][0])):
            row_headers.append("Особь {}".format(i + 1))
        self.ui.table.setVerticalHeaderLabels(row_headers)

        for col in range(4):
            for row in range(len(self.step[self.current_gen][col])):
                self.ui.table.setItem(row, col,
                                      QTableWidgetItem(str(self.step[self.current_gen][col][row].get_chromosome())))

    def fill_cycle_table(self):
        """
        Заполнить таблицу с данными при циклическом режиме работы алгоритма
        :return:
        """
        self.ui.table.setColumnCount(len(self.cycle))
        self.ui.table.setRowCount(len(self.cycle[0]))

        row_headers = []
        for i in range(len(self.cycle)):
            row_headers.append("Поколение {}".format(i + 1))
        self.ui.table.setHorizontalHeaderLabels(row_headers)

        col_headers = []
        header = self.ui.table.horizontalHeader()
        for i in range(len(self.cycle[0])):
            col_headers.append("Особь {}".format(i + 1))
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.ui.table.setVerticalHeaderLabels(col_headers)

        for col in range(len(self.cycle)):
            for row in range(len(self.cycle[0])):
                self.ui.table.setItem(row, col, QTableWidgetItem("{} -> {}"
                                                                 .format(self.cycle[col][row].get_chromosome(),
                                                                         self.cycle[col][row].get_fitness_value())))

    def prev_table_data(self):
        """
        Посмотреть данные о предыдущем поколении при пошаговом режиме работы алгоритма
        :return:
        """
        if not self.step:
            return

        if self.current_gen > 0:
            self.current_gen -= 1
            self.ui.current_gen_label.setText(str(self.current_gen + 1))
        self.fill_step_table()

    def next_table_data(self):
        """
        Посмотреть данные о следующем поколении при пошаговом режиме работы алгоритма
        :return:
        """
        if not self.step:
            return

        if self.current_gen < len(self.step) - 1:
            self.current_gen += 1
            self.ui.current_gen_label.setText(str(self.current_gen + 1))
        self.fill_step_table()

    def step_mode_radio(self):
        """
        Установить текущим режимом работы алгоритма пошаговый режим
        :return:
        """
        if not self.step:
            return

        self.ui.table.clear()
        self.fill_step_table()

    def cycle_mode_radio(self):
        """
        Установить текущим режимом работы алгоритма циклический режим
        :return:
        """
        if not self.step:
            return

        self.ui.table.clear()
        self.fill_cycle_table()
