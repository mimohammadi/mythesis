import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pygad
from config.constants import SystemModelEnums as se
import main
from config.data_generator import Distributions as dist


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
    counter = 0
    deleted = 0
    c = int(len(offspring[0]) / 3)
    for chromosome in offspring:
        a_i_m, y, f_i_m = main.split_chromosome(chromosome)
        sum_req = 0
        sum_a_i_m = [[] for row in range(se.M.value)]
        for i in range(se.K.value):
            for n in range(len(y[i])):  # len(y[i]) = num of requests of mue
                sum_req += 1
                if y[i][n] != 0 and y[i][n] != 1:
                    chromosome[c - 1 + sum_req] = dist.random_distribution(0, 1)
                    # arr = np.delete(arr, counter - deleted, 0)
                    # deleted += 1
                    # break
                for m in range(se.M.value):
                    if np.sum(a_i_m[i][m]) > 0:
                        sum_a_i_m[m] = sum_a_i_m[m] + [i]
                    if len(sum_a_i_m[m]) > se.K__max.value:
                        flag = 0
                        index_m = 0
                        f = 0
                        for mm in range(se.M.value):
                            if i in sum_a_i_m[mm] and mm != m:
                                chromosome[sum_req - 1] = mm + 1
                                flag = 1
                                break
                            elif len(sum_a_i_m[mm]) < se.K__max.value and f == 0:
                                index_m = mm + 1
                                f = 1
                        if flag == 0 and index_m > 0:
                            chromosome[sum_req - 1] = index_m

                        else:
                            chromosome[sum_req - 1] = se.M.value + 1 # means that its going to be computed local or d2d and has no fog
                            y = 2 # ???
                        # arr = np.delete(arr, counter - deleted, 0)
                        # deleted += 1
                        # break
        ########
        for m in range(se.M.value):
            sum_f = 0
            sum_a = 0
            for i in range(se.K.value):
                sum_a += a_i_m[i][m]
                for n in range(len(f_i_m[i][m])):
                    if f_i_m[i][m]:
                        if f_i_m[i][m][n] < 0 or f_i_m[i][m][n] > se.f__0.value:
                            # offspring.remove()
                            arr = np.delete(arr, counter - deleted, 0)
                            deleted += 1
                            break
                        else:
                            sum_f += f_i_m[i][m][n]

            if sum_f > se.f__0.value:
                # offspring.remove(chromosome)
                arr = np.delete(arr, counter - deleted, 0)
                deleted += 1
                break
            elif sum_a > se.K__max.value:
                # offspring.remove(chromosome)
                arr = np.delete(arr, counter - deleted, 0)
                deleted += 1
                break
        counter += 1
    return offspring
