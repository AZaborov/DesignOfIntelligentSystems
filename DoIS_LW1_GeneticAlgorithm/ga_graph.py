"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    02.04.2023
"""

import random
from matplotlib import pyplot as plt
import networkx as nx

INF = 100  # Большое число, обозначающее, что между вершинами нет пути


def generate_graph(nodes_count, is_fully_connected, min_val, max_val):
    """
    Создать граф для ГА
    :param nodes_count: количество вершин
    :param is_fully_connected: является ли граф полносвязным
    :param min_val: минимальное значение ребра
    :param max_val: максимальное значение ребра
    :return: двумерный лист со значениями рёбер
    """
    graph = [[0 for _ in range(nodes_count)] for _ in range(nodes_count)]
    for y in range(nodes_count):
        for x in range(nodes_count):
            val = random.randint(min_val, max_val)
            if not is_fully_connected and random.random() < .4:
                val = INF
            if x != y:
                graph[x][y] = val
                graph[y][x] = val

    return graph


def show_graph(graph):
    """
    Отобразить граф с номерами вершин и значениями рёбер
    :param graph: двумерный лист со значениями рёбер
    :return:
    """
    plt.close()
    G = nx.Graph()
    plt.figure()
    edge_labels = {}
    edges = []

    for y in range(len(graph)):
        for x in range(len(graph)):
            if graph[x][y] != INF and graph[x][y] != 0 and not edges.count([x, y]) > 0:
                edges.append([x, y])
                edge_labels[(x, y)] = str(graph[x][y])

    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_color='red')
    plt.axis('off')
    plt.show()
