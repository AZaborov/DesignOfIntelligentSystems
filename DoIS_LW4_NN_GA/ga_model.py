"""
    Заборов Артемий Михайлович
    artem.zaborov@yandex.ru
    https://online.mospolytech.ru/course/view.php?id=10055
    21.06.2023
"""

import random
import math
import numpy as np
import tensorflow as tf


class Individual:
    def __init__(self, chromosomes):
        self.chromosomes = chromosomes
        self.fitness_value = 0


class GAModel:
    def __init__(self, gens, inds, cros, mut):
        self.generations_count = gens
        self.individuals_count = inds
        self.crossover_probability = cros
        self.mutation_probability = mut
        self.population = self.generate_population()
        self.initialized = False

    def feed_forward(self, x, weights):
        return np.dot(x, weights)

    def generate_population(self):
        return [self.generate_individual() for _ in range(self.individuals_count)]

    def generate_individual(self):
        return Individual([self.generate_chromosome() for _ in range(784)])

    def generate_chromosome(self):
        return np.random.uniform(-1, 1, size=2)

    def fitness(self, population, x, expected):
        for i in range(len(population)):
            real = self.feed_forward(x, population[i].chromosomes)
            bce = tf.keras.losses.BinaryCrossentropy()
            loss = bce(expected, real).numpy()
            population[i].fitness_value = 1 / (loss + 0.00000001)

    def selection(self, population):
        selected = []

        for i in range(0, len(population), 3):
            if i > len(population) - 3:
                break

            candidate1_val = population[i].fitness_value
            candidate2_val = population[i + 1].fitness_value
            candidate3_val = population[i + 2].fitness_value

            if candidate1_val > candidate2_val and candidate1_val > candidate3_val:
                selected.append(population[i])
            elif candidate2_val > candidate1_val and candidate2_val > candidate3_val:
                selected.append(population[i + 1])
            else:
                selected.append(population[i + 2])

        return selected

    def crossover(self, selected):
        children = []

        for i in range(0, len(selected) - 1):
            if random.random() < self.crossover_probability:
                child1 = []
                child2 = []

                for j in range(784):
                    parent1_genes = selected[i].chromosomes[j]
                    parent2_genes = selected[i + 1].chromosomes[j]

                    if random.choice([True, False]):
                        child1.append([parent1_genes[0], parent2_genes[1]])
                        child2.append([parent2_genes[0], parent1_genes[1]])
                    else:
                        child1.append([parent2_genes[0], parent1_genes[1]])
                        child2.append([parent1_genes[0], parent2_genes[1]])

                children.append(Individual(child1))
                children.append(Individual(child2))
            else:
                children.append(selected[i])
                children.append(selected[i + 1])

        return children

    def mutation(self, children):
        for i in range(len(children)):
            for j in range(len(children[i].chromosomes)):
                if random.random() < self.mutation_probability:
                    children[i].chromosomes[j][random.choice([0, 1])] = np.random.uniform(-1, 1)

        return children

    def process(self, x, expected):
        if not self.initialized:
            self.fitness(self.population, x, expected)
            self.initialized = True

        for i in range(self.generations_count):
            selected = self.selection(self.population)
            children = self.crossover(selected)
            self.mutation(children)
            self.population = children
            self.fitness(self.population, x, expected)

        self.population.sort(key=lambda j: j.fitness_value)
        return self.population[0]
