import numpy as np
# from geneticalgorithm import geneticalgorithm as ga
import pygad
import algorithmes.ga_alg as ga


class GeneticAlg:
    @classmethod
    def genetic_alg(cls, iteration_num, parent_num, fitness, gene_type,
                    number_of_solutions, num_genes, crossover_probability,
                    mutation_probability, gene_space):
        # initial_population is built by sol_per_pop and num_genes
        # num_genes = Number of genes in the solution / chromosome
        # ga_instance = pygad.GA(num_generations=iteration_num,
        #                        num_parents_mating=parent_num,
        #                        fitness_func=fitness,
        #                        sol_per_pop=number_of_solutions,
        #                        num_genes=num_genes,
        #                        gene_type=[float, 4],
        #                        init_range_low=0.0,
        #                        init_range_high=1.0,
        #                        parent_selection_type="rws",
        #                        keep_parents=0,
        #                        crossover_type="single_point",
        #                        crossover_probability=crossover_probability,
        #                        mutation_type="random",
        #                        mutation_probability=mutation_probability,
        #                        mutation_by_replacement=False,
        #                        random_mutation_min_val=0.0,
        #                        random_mutation_max_val=1.0,
        #                        on_mutation=on_mutation)

        ga_instance = ga.GA(num_generations=iteration_num,
                            initial_pop_num=number_of_solutions,
                            num_parents_mating=parent_num,
                            fitness_func=fitness,
                            gene_num=num_genes,
                            gene_type=gene_type,
                            keep_parents=0,
                            crossover_probability=crossover_probability,
                            mutation_probability=mutation_probability,
                            on_constrain=on_mutation,
                            save_best_solutions=True,
                            gene_space=gene_space)

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

    # @classmethod
    # def on_mutation(cls, offspring, ga_instance):
    #     for chromosome in offspring:
    #         sum_ = 0
    #         for q in chromosome:
    #             if q < 0:
    #                 offspring.remove(chromosome)
    #                 # break
    #             sum_ += q
    #         if sum_ > 1:
    #             offspring.remove(chromosome)
    #     return offspring


def on_mutation(ga_instance, offspring):
    print('offspring=')
    print(offspring)
    counter = 0
    deleted = 0
    arr = offspring
    for chromosome in offspring:
        sum_ = 0
        for q in chromosome:
            if q < 0:
                arr = np.delete(arr, counter - deleted, 0)
                deleted += 1
                break
            sum_ += q
        if sum_ > 1:
            arr = np.delete(arr, counter - deleted, 0)
            deleted += 1
            print('arr = ')
            print(arr)

        counter += 1
    print('offspring = ')
    print(arr)
    return arr
