import math
import random
import sys

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

import gui
from PyQt5.QtWidgets import QApplication

NODES_COUNT = 6
START_NODE = 0
FINISH_NODE = 5
POPULATION_COUNT = 100
INF = 100
NETWORK = ((0, 3, 1, 3, INF, INF),
         (3, 0, 4, INF, INF, INF),
         (1, 4, 0, INF, 7, 5),
         (3, INF, INF, 0, INF, 2),
         (INF, INF, 7, INF, 0, 4),
         (INF, INF, 5, 2, 4, 0))


def show_graph():
    g = nx.complete_graph(8)

    for i in range(15):
        a = random.randint(1, NODES_COUNT + 1)
        b = random.randint(1, NODES_COUNT + 1)
        if a != b:
            g.add_edge(a, b)

    g.number_of_edges(1, NODES_COUNT)
    nx.draw(g, with_labels=True)
    plt.show()


def fitness(ind):
    fit = 0
    return fit


def crossover(ind1, ind2):
    for gene1 in ind1:
        for gene2 in ind2:
            print(gene1, gene2)

    return ind1, ind2


if __name__ == "__main__":
    population = []
    chromosome = [item for item in range(0, NODES_COUNT)]

    while len(population) < min(POPULATION_COUNT, math.factorial(NODES_COUNT)):
        random.shuffle(chromosome)
        if chromosome not in population:
            population.append(chromosome.copy())

    fitnesses = []
    for individual in population:
        fitnesses.append(fitness(individual))

    print(fitnesses)
    # app = QApplication(sys.argv)
    # win = gui.Window()
    # sys.exit(app.exec_())
