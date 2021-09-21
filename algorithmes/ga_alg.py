from random import random, choice
import pickle
import matplotlib
import matplotlib.pyplot
import numpy as np
import warnings

from config.data_generator import Distributions as dsb


class GA:
    def __init__(self, num_generations, initial_pop_num,
                 num_parents_mating, gene_num,
                 gene_type,
                 gene_space, fitness_func,
                 crossover_probability, mutation_probability,
                 keep_parents=-1, save_best_solutions=False,
                 on_constrain=None):


        self.num_generations = num_generations
        self.initial_pop_num = initial_pop_num
        self.gene_num = gene_num  # list of number of each gene types
        self.total_gene_num = 0
        self.gene_type = gene_type
        self.gene_space = gene_space
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.fitness_func = None
        self.last_generation_fitness = None  # A list holding the fitness values of all solutions in the last generation.
        self.initial_population = self.initial_population_()
        self.population = self.initial_population.copy()
        self.best_solutions_fitness = []  # A list holding the fitness value of the best solution for each generation.
        self.solutions_fitness = []  # Holds the fitness of the solutions in each generation.
        self.solutions = []  # Holds the solutions in each generation.
        self.best_solutions = []  # Holds the best solution in each generation.
        self.keep_parents = keep_parents
        self.num_parents_mating = num_parents_mating
        self.crossover_type = "single_point"
        self.mutation_type = "random"
        self.save_best_solutions = save_best_solutions
        if not (on_constrain is None):
            self.on_constrain = on_constrain
        else:
            self.on_constrain = None
        for i in self.gene_num:
            self.total_gene_num += i
        # Validate keep_parents.
        if (self.keep_parents == -1):  # Keep all parents in the next population.
            self.num_offspring = self.initial_pop_num - self.num_parents_mating
        elif (self.keep_parents == 0):  # Keep no parents in the next population.
            self.num_offspring = self.initial_pop_num
        elif (self.keep_parents > 0):  # Keep the specified number of parents in the next population.
            self.num_offspring = self.initial_pop_num - self.keep_parents

        # Check if the parent_selection_type is a function that accepts 3 paramaters.
        self.select_parents = self.roulette_wheel_selection

        if (self.crossover_type == "single_point"):
            self.crossover = self.single_point_crossover

        if (self.mutation_type == "random"):
            self.mutation = self.random_mutation

        if callable(fitness_func):
            # Check if the fitness function accepts 2 paramaters.
            if (fitness_func.__code__.co_argcount == 2):
                self.fitness_func = fitness_func
            else:
                self.valid_parameters = False
                raise ValueError("The fitness function must accept 2 parameters:\n1) A solution to calculate its fitness value.\n2) The solution's index within the population.\n\nThe passed fitness function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=fitness_func.__code__.co_name, argcount=fitness_func.__code__.co_argcount))
        else:
            self.valid_parameters = False
            raise ValueError("The value assigned to the fitness_func parameter is expected to be of type function but ({fitness_func_type}) found.".format(fitness_func_type=type(fitness_func)))


    def initial_population_(self):
        initial_pop = []
        for j in range(self.initial_pop_num):
            chromosome = []
            for num in range(len(self.gene_num)):
                for i in range(0, self.gene_num[num]):
                    if self.gene_type[num] == int:
                        chromosome.append(dsb.random_distribution(self.gene_space[num][0], self.gene_space[num][1]))
                    elif self.gene_type[num] == float:
                        rand = random() * (self.gene_space[num][1] - self.gene_space[num][0]) + self.gene_space[num][0]
                        chromosome.append(rand)
                    elif self.gene_type[num] == bool:
                        chromosome.append(np.random.randint(2, size=1))
            initial_pop.append(chromosome)
        print('initial_pop = ')
        print(initial_pop)

        return np.array(initial_pop)

    def cal_pop_fitness(self):

        """
        Calculating the fitness values of all solutions in the current population.
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(self.population):
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(fitness)

        pop_fitness = np.array(pop_fitness)

        return pop_fitness

    # def parent_selection_func(self, fitness, num_parents, ga_instance):
    #
    #     fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    #     fitness_sorted.reverse()
    #
    #     parents = np.empty((num_parents, ga_instance.population.shape[1]))
    #
    #     for parent_num in range(num_parents):
    #         parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()
    #
    #     return parents, fitness_sorted[:num_parents]

    def roulette_wheel_selection(self, fitness, num_parents):

        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = np.sum(fitness)
        probs = fitness / fitness_sum
        print('fitness')
        print(fitness)
        probs_start = np.zeros(probs.shape,
                                  dtype=np.float)  # An array holding the start values of the ranges of probabilities.
        probs_end = np.zeros(probs.shape,
                                dtype=np.float)  # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = np.where(probs == np.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if len(self.gene_type) == 1:
            parents = np.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = np.empty((num_parents, self.population.shape[1]), dtype=object)

        parents_indices = []

        for parent_num in range(num_parents):
            rand_prob = np.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    parents_indices.append(idx)
                    break
        return parents, parents_indices

    def single_point_crossover(self, parents, offspring_size):

        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = np.empty(offspring_size, dtype=object)

        for k in range(offspring_size[0]):
            # The point at which crossover takes place between two parents. Usually, it is at the center.
            crossover_point = np.random.randint(low=0, high=parents.shape[1], size=1)[0]


            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k + 1) % parents.shape[0]

            # The new offspring has its first half of its genes from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring has its second half of its genes from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def random_mutation(self, offspring):

        """
        Applies the random mutation which changes the values of a number of genes randomly.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        # if self.mutation_probability is None:
        #     # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
        #     if not (self.gene_space is None):
        #         # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
        #         offspring = self.mutation_by_space(offspring)
        #     # else:
        #     #     offspring = self.mutation_randomly(offspring)
        # else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
        if not (self.gene_space is None):
            # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
            offspring = self.mutation_probs_by_space(offspring)
            # else:
            #     offspring = self.mutation_probs_randomly(offspring)

        return offspring

    def mutation_probs_by_space(self, offspring):

        """
        Applies the random mutation using the mutation values' space and the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated based on the mutation space.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            probs = np.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_probability:
                    if len(self.gene_space) > 1:
                        # Returning the current gene space from the 'gene_space' attribute.
                        if type(self.gene_space[gene_idx]) in [np.ndarray, list]:
                            curr_gene_space = self.gene_space[gene_idx].copy()
                        else:
                            curr_gene_space = self.gene_space[gene_idx]

                        # If the gene space has only a single value, use it as the new gene value.
                        # if type(curr_gene_space) in GA.supported_int_float_types:
                        #     value_from_space = curr_gene_space
                        # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                        # elif curr_gene_space is None:
                        #     rand_val = np.random.uniform(low=self.random_mutation_min_val,
                        #                                     high=self.random_mutation_max_val,
                        #                                     size=1)
                        #     if self.mutation_by_replacement:
                        #         value_from_space = rand_val
                        #     else:
                        #         value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                        # elif type(curr_gene_space) is dict:
                        #     # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                        #     if 'step' in curr_gene_space.keys():
                        #         value_from_space = np.random.choice(np.arange(start=curr_gene_space['low'],
                        #                                                             stop=curr_gene_space['high'],
                        #                                                             step=curr_gene_space['step']),
                        #                                                size=1)
                        #     else:
                        #         value_from_space = np.random.uniform(low=curr_gene_space['low'],
                        #                                                 high=curr_gene_space['high'],
                        #                                                 size=1)
                        # else:
                        # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                        # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                        if len(curr_gene_space) == 1:
                            value_from_space = curr_gene_space[0]
                        # If the gene space has more than 1 value, then select a new one that is different from the current value.
                        else:
                            values_to_select_from = list(
                                set(curr_gene_space) - {offspring[offspring_idx, gene_idx]})
                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        # if type(self.gene_space) is dict:
                        #     if 'step' in self.gene_space.keys():
                        #         value_from_space = np.random.choice(np.arange(start=self.gene_space['low'],
                        #                                                             stop=self.gene_space['high'],
                        #                                                             step=self.gene_space['step']),
                        #                                                size=1)
                        #     else:
                        #         value_from_space = np.random.uniform(low=self.gene_space['low'],
                        #                                                 high=self.gene_space['high'],
                        #                                                 size=1)
                        # else:
                        values_to_select_from = list(
                            set(self.gene_space[0]) - set(np.array([offspring[offspring_idx, gene_idx]]))) #[off...]
                        if len(values_to_select_from) == 0:
                            value_from_space = offspring[offspring_idx, gene_idx]
                        else:
                            value_from_space = choice(values_to_select_from)

                    # Assigning the selected value from the space to the gene.
                    if len(self.gene_type) == 1:
                        if type(self.gene_type[0]) == list:
                            offspring[offspring_idx, gene_idx] = np.round(self.gene_type[0](value_from_space),
                                                                             self.gene_type[1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            offspring[offspring_idx, gene_idx] = np.round(
                                self.gene_type[gene_idx][0](value_from_space),
                                self.gene_type[gene_idx][1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                    # if self.allow_duplicate_genes == False:
                    #     offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(
                    #         solution=offspring[offspring_idx],
                    #         gene_type=self.gene_type,
                    #         num_trials=10)
        return offspring

    def best_solution(self, pop_fitness=None):

        """
        Returns information about the best solution found by the genetic algorithm.
        Accepts the following parameters:
            pop_fitness: An optional parameter holding the fitness values of the solutions in the current population. If None, then the cal_pop_fitness() method is called to calculate the fitness of the population.
        The following are returned:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
            -best_match_idx: Index of the best solution in the current population.
        """

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        if pop_fitness is None:
            pop_fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(pop_fitness == np.max(pop_fitness))[0][0]

        best_solution = self.population[best_match_idx, :].copy()
        best_solution_fitness = pop_fitness[best_match_idx]

        return best_solution, best_solution_fitness, best_match_idx

    def plot_fitness(self,
                         title="PyGAD - Generation vs. Fitness",
                         xlabel="Generation",
                         ylabel="Fitness",
                         linewidth=3,
                         font_size=14,
                         plot_type="plot",
                         color="#3870FF",
                         save_dir=None):

        """
        Creates, shows, and returns a figure that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.
        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#3870FF".
            save_dir: Directory to save the figure.
        Returns the figure.
        """

        if self.generations_completed < 1:
            raise RuntimeError(
                "The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(
                    generations_completed=self.generations_completed))

        #        if self.run_completed == False:
        #            if not self.suppress_warnings: warnings.warn("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        fig = matplotlib.pyplot.figure()
        if plot_type == "plot":
            matplotlib.pyplot.plot(self.best_solutions_fitness, linewidth=linewidth, color=color)
        elif plot_type == "scatter":
            matplotlib.pyplot.scatter(range(self.generations_completed + 1), self.best_solutions_fitness,
                                      linewidth=linewidth, color=color)
        elif plot_type == "bar":
            matplotlib.pyplot.bar(range(self.generations_completed + 1), self.best_solutions_fitness,
                                  linewidth=linewidth, color=color)
        matplotlib.pyplot.title(title, fontsize=font_size)
        matplotlib.pyplot.xlabel(xlabel, fontsize=font_size)
        matplotlib.pyplot.ylabel(ylabel, fontsize=font_size)

        if not save_dir is None:
            matplotlib.pyplot.savefig(fname=save_dir,
                                      bbox_inches='tight')
        matplotlib.pyplot.show()

        return fig

    def run(self):

        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """
        if not (self.on_constrain is None):
            self.population = self.on_constrain(self, self.population)
            # if self.population.size == 0:
            #     self.population = self.initial_population_()

        # # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
        self.last_generation_fitness = self.cal_pop_fitness()

        best_solution, best_solution_fitness, best_match_idx = self.best_solution(
            pop_fitness=self.last_generation_fitness)

        # Appending the best solution in the initial population to the best_solutions list.
        if self.save_best_solutions:
            self.best_solutions.append(best_solution)
        #
        # # Appending the solutions in the initial population to the solutions list.
        # # if self.save_solutions:
        # #     self.solutions.extend(self.population.copy())

        for generation in range(self.num_generations):
            # if not (self.on_fitness is None):
            #     self.on_fitness(self, self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)
            #
            # if self.save_solutions:
            #     self.solutions_fitness.extend(self.last_generation_fitness)

            # # Selecting the best parents in the population for mating.
            # if callable(self.parent_selection_type):
            self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(
                self.last_generation_fitness, self.num_parents_mating)
            # else:
            #     self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(
            #         self.last_generation_fitness, num_parents=self.num_parents_mating)
            # if not (self.on_parents is None):
            #     self.on_parents(self, self.last_generation_parents)

            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
            if self.crossover_type is None:
                if self.num_offspring <= self.keep_parents:
                    self.last_generation_offspring_crossover = self.last_generation_parents[0:self.num_offspring]
                else:
                    self.last_generation_offspring_crossover = np.concatenate((self.last_generation_parents,
                                                                                  self.population[0:(
                                                                                              self.num_offspring -
                                                                                              self.last_generation_parents.shape[
                                                                                                  0])]))
            else:
                # Generating offspring using crossover.
                if callable(self.crossover_type):
                    self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                              (self.num_offspring, self.gene_num),
                                                                              self)
                else:
                    self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                              offspring_size=(
                                                                              self.num_offspring, self.total_gene_num))
                # if not (self.on_crossover is None):
                #     self.on_crossover(self, self.last_generation_offspring_crossover)

            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                self.last_generation_offspring_mutation = self.last_generation_offspring_crossover
            else:
                # Adding some variations to the offspring using mutation.
                if callable(self.mutation_type):
                    self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover,
                                                                            self)
                else:
                    self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover)
                # if not (self.on_mutation is None):
                #     self.on_mutation(self, self.last_generation_offspring_mutation)

            if (self.keep_parents == 0):
                self.population = self.last_generation_offspring_mutation
            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[0:self.last_generation_parents.shape[0], :] = self.last_generation_parents
                self.population[self.last_generation_parents.shape[0]:, :] = self.last_generation_offspring_mutation
            # elif (self.keep_parents > 0):
            #     parents_to_keep, _ = self.steady_state_selection(self.last_generation_fitness,
            #                                                      num_parents=self.keep_parents)
            #     self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
            #     self.population[parents_to_keep.shape[0]:, :] = self.last_generation_offspring_mutation

            self.generations_completed = generation + 1  # The generations_completed attribute holds the number of the last completed generation.

            # here is where we should apply constrains
            if not (self.on_constrain is None):
                self.population = self.on_constrain(self, self.population)
                if self.population.size == 0:
                    self.population = self.initial_population_()

            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(
                pop_fitness=self.last_generation_fitness)

            # # Appending the best solution in the current generation to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)
            #
            # # Appending the solutions in the current generation to the solutions list.
            # if self.save_solutions:
            #     self.solutions.extend(self.population.copy())

            # If the callback_generation attribute is not None, then cal the callback function after the generation.
            # if not (self.on_generation is None):
            #     r = self.on_generation(self)
            #     if type(r) is str and r.lower() == "stop":
            #         # Before aborting the loop, save the fitness value of the best solution.
            #         _, best_solution_fitness, _ = self.best_solution()
            #         self.best_solutions_fitness.append(best_solution_fitness)
            #         break

            # if not self.stop_criteria is None:
            #     for criterion in self.stop_criteria:
            #         if criterion[0] == "reach":
            #             if max(self.last_generation_fitness) >= criterion[1]:
            #                 stop_run = True
            #                 break
            #         elif criterion[0] == "saturate":
            #             criterion[1] = int(criterion[1])
            #             if (self.generations_completed >= criterion[1]):
            #                 if (self.best_solutions_fitness[self.generations_completed - criterion[1]] -
            #                     self.best_solutions_fitness[self.generations_completed - 1]) == 0:
            #                     stop_run = True
            #                     break

            # if stop_run:
            #     break

            # time.sleep(self.delay_after_gen)

        # Save the fitness value of the best solution.
        _, best_solution_fitness, _ = self.best_solution(pop_fitness=self.last_generation_fitness)
        self.best_solutions_fitness.append(best_solution_fitness)

        self.best_solution_generation = \
        np.where(np.array(self.best_solutions_fitness) == np.max(np.array(self.best_solutions_fitness)))[0][
            0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True  # Set to True only after the run() method completes gracefully.

        # if not (self.on_stop is None):
        #     self.on_stop(self, self.last_generation_fitness)

        # Converting the 'best_solutions' list into a NumPy array.
        self.best_solutions = np.array(self.best_solutions)

        # Converting the 'solutions' list into a NumPy array.
        self.solutions = np.array(self.solutions)


    def save(self, filename):

        """
        Saves the genetic algorithm instance:
            -filename: Name of the file to save the instance. No extension is needed.
        """

        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)


def load(filename):

    """
    Reads a saved instance of the genetic algorithm:
        -filename: Name of the file to read the instance. No extension is needed.
    Returns the genetic algorithm instance.
    """

    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Error reading the file {filename}. Please check your inputs.".format(filename=filename))
    except:
        raise "Error loading the file. Please check if the file exists."
    return ga_in