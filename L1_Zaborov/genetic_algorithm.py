"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    02.04.2023
"""

import random
import math

NODES_COUNT = 0
START_NODE = 0
FINISH_NODE = 0
INF = 1000000000  # Очень большое число, обозначающее, что между вершинами нет пути


class Individual:
    """
    Особь для ГА
    """
    def __init__(self, chromosome):
        """
        Задать параметры особи: хромосома и значение функции приспособленности
        :param chromosome: хромосома (набор вершин, по которым нужно пройтись, "-1" - пропуск вершины)
        """
        self._chromosome = chromosome
        self._fitness_value = 0

    def get_chromosome(self):
        return self._chromosome

    def set_chromosome(self, chrom):
        self._chromosome = chrom

    def get_fitness_value(self):
        return self._fitness_value

    def set_fitness_value(self, fit):
        self._fitness_value = fit


def generate_population(count):
    """
    Создать популяцию
    :param count: количестов особей в популяции
    :return: созданная популяция
    """
    population = []

    chromosome = [item for item in range(0, NODES_COUNT)]
    chromosome.remove(START_NODE)
    chromosome.remove(FINISH_NODE)

    # По-умолчанию мы сами задаём число популяции, но если математически невозможно
    # достичь такого числа, берём максимльно возможное (факториал количества вершин)
    while len(population) < min(count, math.factorial(NODES_COUNT - 2)):
        random.shuffle(chromosome)
        copy = chromosome.copy()

        # С вероятностью 40% рандомно меняем вершины в путях на "-1"
        # Это нужно для того, чтобы не пришлось проходиться по всем вершинам
        for j in range(NODES_COUNT - 2):
            if random.random() < .4:
                copy[j] = -1

        # Проверяем, чтобы такой уже особи не было в популяции
        if next((False for i in population if i.get_chromosome == copy), True):
            ind = Individual(copy)
            population.append(ind)

    return population


def fitness(population, graph):
    """
    Вычисляем для каждой особи значение функции приспособленности
    :param population: популяция
    :param graph: граф
    :return:
    """
    for ind in population:
        fit = 0
        chromosome = ind.get_chromosome().copy()
        chromosome.append(FINISH_NODE)
        chromosome.insert(0, START_NODE)
        chromosome = list(filter(lambda a: a != -1, chromosome))

        for i in range(len(chromosome) - 1):
            x = chromosome[i]
            y = chromosome[i + 1]

            distance = graph[x][y]
            if distance == INF:  # Пути не существует
                fit = INF
                break
            else:
                fit += distance

        ind.set_fitness_value(fit)


def selection(population):
    """
    Выбираем лучших из популяции
    Особи делятся на группы по 3, побеждает 1: тот, у которого самое большое значение функции приспособленности
    :param population: популяция
    :return: лучшие
    """
    selected = []

    for i in range(0, len(population), 3):
        if i > len(population) - 3:
            break

        candidate1_val = population[i].get_fitness_value()
        candidate2_val = population[i + 1].get_fitness_value()
        candidate3_val = population[i + 2].get_fitness_value()

        if candidate1_val < candidate2_val and candidate1_val < candidate3_val:
            selected.append(population[i])
        elif candidate2_val < candidate1_val and candidate2_val < candidate3_val:
            selected.append(population[i + 1])
        else:
            selected.append(population[i + 2])

    return selected


def crossover(selected):
    """
    Создаём новых особей путём двуточечного скрещивания лучших
    :param selected: родители (лучшие из популяции)
    :return: дети лучших
    """
    children = []

    for i in range(0, len(selected), 2):
        if i > len(selected) - 2:
            break

        parent1 = selected[i].get_chromosome().copy()
        parent2 = selected[i + 1].get_chromosome().copy()

        mid1 = random.randint(1, NODES_COUNT - 4)
        mid2 = random.randint(mid1 + 1, NODES_COUNT - 3)

        # Для восполнения популяции каждые два родителя создают 6 детей
        children.append(Individual(parent1[:mid1] + parent2[mid1:mid2] + parent2[mid2:]))
        children.append(Individual(parent2[:mid1] + parent1[mid1:mid2] + parent1[mid2:]))
        children.append(Individual(parent1[:mid1] + parent1[mid1:mid2] + parent2[mid2:]))
        children.append(Individual(parent2[:mid1] + parent2[mid1:mid2] + parent1[mid2:]))
        children.append(Individual(parent1[:mid1] + parent2[mid1:mid2] + parent1[mid2:]))
        children.append(Individual(parent2[:mid1] + parent1[mid1:mid2] + parent2[mid2:]))

    return children


def mutation(children, probability):
    """
    Мутация
    :param children: новое поколение
    :param probability: вероятность мутации
    :return:
    """
    choice = list(range(0, NODES_COUNT))
    choice.remove(START_NODE)
    choice.remove(FINISH_NODE)
    choice.append(-1)

    for i in range(len(children)):
        for j in range(len(children[i].get_chromosome())):
            if random.random() < probability:
                chrom = children[i].get_chromosome().copy()
                chrom[j] = random.choice(choice)
                children[i] = Individual(chrom)
                break


def process(graph, epoch_count, pop_count, start, finish, mut_prob):
    """
    Процесс эволюции для решения задачи нахождения кратчайшего пути между выбранными вершинами
    :param graph: граф со значениями рёбер
    :param epoch_count: количество эпох/поколений
    :param pop_count: размер популяции
    :param start: начальная вершина
    :param finish: конечная вершина
    :param mut_prob: вероятность мутации
    :return: лучший из последнего поколения, история поколений для пошагового и циклического режимов работ алгоритма
    """
    global NODES_COUNT, START_NODE, FINISH_NODE
    NODES_COUNT = len(graph)
    START_NODE = start
    FINISH_NODE = finish
    steps = []
    cycle = []

    random.seed(99)
    population = generate_population(pop_count)

    for i in range(epoch_count):
        fitness(population, graph)
        selected = selection(population)
        children = crossover(selected)
        before_mutation = children.copy()
        mutation(children, mut_prob)

        steps.append([population.copy(), selected.copy(), before_mutation.copy(), children.copy()])
        cycle.append(children.copy())
        population = children

    fitness(population, graph)
    population.sort(key=lambda x: x.get_fitness_value(), reverse=False)
    return population[0], steps, cycle
