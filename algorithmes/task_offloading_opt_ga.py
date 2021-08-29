import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pygad
from config.constants import SystemModelEnums as se


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
                               gene_space=[[range(1, se.K.value), range(1, se.M.value), [0, 1], range(0, se.f__0.value)] for row in range(num_genes)],
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
    for chromosome in offspring:
        a_i_m = [[0 for row in range(se.M.value)] for col in range(se.K.value)]
        y = [0 for col in range(se.K.value)]
        f_i_m = [[0 for row in range(se.M.value)] for col in range(se.K.value)]
        c = len(chromosome)/4
        for i in range(int(c)):
            a_i_m[chromosome[i] - 1][chromosome[i+1] - 1] = 1
            y[chromosome[i] - 1] = chromosome[i+2]
            f_i_m[chromosome[i] - 1][chromosome[i + 1] - 1] = chromosome[i+3]

        for i in range(se.K.value):
            sum_ = 0
            if y[i] != 0 and y[i] != 1:
                offspring.remove(chromosome)
                break
            else:
                for m in range(se.M.value):
                    if f_i_m[i][m] < 0 or f_i_m[i][m] > se.f__0.value:
                        offspring.remove()
                        break
                    else:
                        sum_ += a_i_m[i][m]
                if sum_ > 1:
                    offspring.remove(chromosome)
                    break
        for m in range(se.M.value):
            sum_f = 0
            sum_a = 0
            for i in range(se.K.value):
                sum_f += f_i_m[i][m]
                sum_a += a_i_m[i][m]

            if sum_f > se.f__0.value:
                offspring.remove(chromosome)
                break
            elif sum_a > se.K__max.value:
                offspring.remove(chromosome)
                break

    return offspring

