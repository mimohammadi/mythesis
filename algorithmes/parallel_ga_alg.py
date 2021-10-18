import numpy as np
from multiprocessing import Pool
from algorithmes import ga_alg
import time


def fitness_wrapper(solution):
    return fitness_func(solution, 0)


class PooledGA(ga_alg.GA):

    def cal_pop_fitness(self):
        global pool

        pop_fitness = pool.map(fitness_wrapper, self.population)
        print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness