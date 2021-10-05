from random import random
import numpy as np
#from geneticalgorithm import geneticalgorithm as ga
import pygad
from config.constants import SystemModelEnums as se
from config.data_generator import Distributions as dist
import pandas as pd
import xlsxwriter


class GeneticAlg:
    @classmethod
    def genetic_alg(cls, iteration_num, parent_num, fitness,
                    number_of_solutions, num_genes, crossover_probability,
                    mutation_probability):
        # initial_population is built by sol_per_pop and num_genes
        # num_genes = Number of genes in the solution / chromosome
        ga_instance = pygad.GA(num_generations=iteration_num,
                               num_parents_mating=parent_num,
                               fitness_func=fitness,
                               sol_per_pop=number_of_solutions,
                               num_genes=num_genes,
                               gene_type=[[int, int, int, float] for row in range(num_genes)],
                               gene_space=[[range(1, se.K.value), range(1, se.M.value), [0, 1], range(0, se.f__0.value)]
                                           for row in range(num_genes)],
                               parent_selection_type="rws",
                               keep_parents=0,
                               crossover_type="single_point",
                               crossover_probability=crossover_probability,
                               mutation_type="random",
                               mutation_probability=mutation_probability,
                               mutation_by_replacement=False,
                               on_mutation=on_mutation)
        ga_instance.run()
        ga_instance.plot_fitness()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        filename = 'genetic'
        ga_instance.save(filename=filename)
        loaded_ga_instance = pygad.load(filename=filename)
        return loaded_ga_instance.best_solution()


def on_mutation(ga_instance, offspring):
    arr = offspring
    # c = int(len(offspring[0]) / 3)
    for ind_ch, chromosome in enumerate(offspring):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        chrom, a__i_m, y__ = split_chromosome(chromosome)
        arr[ind_ch] = chrom

    return arr


def split_chromosome(solution2):

    df = pd.read_excel('algorithmes/request.xlsx')
    req_set = df.to_numpy()
    print('req_set')
    print(req_set)
    new_req = [[int(req_set[i][j]) if np.isnan(req_set[i][j]) == False else -1 for i in range(np.size(req_set, 0))] for j in range(np.size(req_set, 1))]
    print('new_req')
    print(new_req)
    a__i_m = [[[0 for j in range(len(new_req[col]))] for row in range(se.M.value)] for col in range(se.K.value)]
    y__ = [[0 for j in range(len(new_req[col]))] for col in range(se.K.value)]
    # f__i_m = [[[0 for j in range(len(new_req[col]))] for row in range(se.M.value)] for col in range(se.K.value)]
    c = int(len(solution2) / 2)
    sum_f_m = [0 for row in range(se.M.value)]
    sum_of_req = 0
    sum_a__i_m = [[] for row in range(se.M.value)]
    for i in range(se.K.value):
        # for j in range(c): # just for mues
        fog_i = 0

        for n in range(len(new_req[i])):
            if new_req[i][n] != -1:
                sum_of_req += 1
                print('sum_of_req')
                print(sum_of_req)
                if int(solution2[sum_of_req - 1]) > se.M.value or int(solution2[sum_of_req - 1]) < 1:
                    solution2[sum_of_req - 1] = dist.random_distribution(1, se.M.value)

                if n == 0:
                    fog_i = int(solution2[sum_of_req - 1]) - 1
                else:
                    solution2[sum_of_req - 1] = fog_i + 1  # first fog of requests of mue witch is not fog+1, we give it to all requests of mue
                a__i_m[i][int(solution2[sum_of_req - 1]) - 1][n] = 1  # [mue][fog]
                if int(solution2[c + sum_of_req - 1]) != 0 and int(solution2[c + sum_of_req - 1]) != 1:  # 2 means local or d2d cache
                    solution2[c + sum_of_req - 1] = dist.random_distribution(0, 1)
                    y__[i][n] = int(solution2[c + sum_of_req - 1])
                else:
                    y__[i][n] = int(solution2[c + sum_of_req - 1])

    sum_of_req = 0

    for i in range(se.K.value):

        for n in range(len(new_req[i])):
            if new_req[i][n] != -1:
                sum_of_req += 1
                for m in range(se.M.value):
                    if np.sum(np.array(a__i_m[i][m])) > 0 and i not in sum_a__i_m[m]:
                        sum_a__i_m[m] = sum_a__i_m[m] + [i]
                        print(sum_a__i_m)

    return solution2, np.array(a__i_m), np.array(y__)
