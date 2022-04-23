#! /usr/bin/env python3
##
# GA primatives
##
import random
import numpy as np


class Individual(object):
    def __init__(self):
        self.genome = []
        self.fitness = None

def example_generator():
    # Example generator
    individual = []
    individual.append(random.choice([100, 1000, 1000, 10000, 100000, 1100, 20000, 3000]))  # step total
    individual.append(random.choice([0.01, 0.05, 0.10, 0.5, 0.2, 0.35, 0.48, 0.082, 0.29]))  # learning rate
    individual.append(random.choice([0.5, 0.73, 0.57, 0.85, 0.62, 0.9, 1, 0.7, 0.55, 0.6]))  # discount factor
    return individual

# generate each population with a set amount of individuals
def generate_population(n, generator):
    population = []
    for _ in range(n):
        population.append(generator())
    return population


def single_crossover(p1, p2):
    point = random.randint(0, len(p1))
    # cross over parent's genomes to ensure offspring has both parent's traits
    child = [*p1[:point], *p2[point:]]
    return child


def next_population(population, select):
    new_population = []
    # cache fitness so we don't need to recalculate for every parent
    cache = dict()

    for _ in range(len(population)):
        p1 = select(population, cache)
        p2 = select(population, cache)

        new_population.append(single_crossover(p1, p2))
        print("new population: ", new_population)

    # add individual traits to a random individual in the population for possible better performance
    random_individual = new_population[random.randint(0, len(new_population)-1)]
    print("random individual: ", random_individual)

    # pick one part of the genome at random to ensure individual and parent traits exits
    random_genome = random.randint(0, len(random_individual))
    if random_genome is 0:
        random_individual[0] += round(random.uniform(-10, 10))

    if random_genome is 1 or random_genome is 2:
        while True:
            # learn rate and discount factor must be between 0 and 1 to work properly
            random_num = round(random.uniform(-1, 1), 2)
            if (random_individual[random_genome] + random_num) > 0 and (random_individual[random_genome] + random_num) < 1:
                random_individual[random_genome] += round(random_num, 2)
                break

    return new_population, cache

