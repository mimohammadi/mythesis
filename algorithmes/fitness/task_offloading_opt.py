from pygad import pygad
import numpy as np
from config.constants import SystemModelEnums
import math
# from algorithmes.gini_coefficient_alg import GiniCoefficientBasedAlg as gini
from models.task import Task


class GiniFunctions:
    @classmethod
    def income_function(cls, task_library, n, distances_of_fog, i, m):
        t_l_i_n = task_library[n].D__n / SystemModelEnums.w.value
        e_l_i_n = SystemModelEnums.kappa.value * task_library[n].D__n * SystemModelEnums.w.value ** 2
        t_u_i_m_n = task_library[n].S__n / cls.transmit_rate(distances_of_fog, i, m)
        e_u_i_m_n = SystemModelEnums.p__u.value * t_u_i_m_n
        t_d_m_i_n = task_library[n].S__n * task_library[n].Theta__n / cls.downlink_rate(distances_of_fog, i, m)
        e_d_m_i_n = SystemModelEnums.p__m.value * t_d_m_i_n
        t_T_i_m_n = t_u_i_m_n + t_d_m_i_n
        e_T_i_m_n = e_u_i_m_n + e_d_m_i_n
        t__i_m_n_f_0 = task_library[n].D__n / SystemModelEnums.f__0.value  #** f_i_m
        e__cpt_i_m_n_f_0 = SystemModelEnums.kappa_server * task_library[n].D__n * (SystemModelEnums.f__0.value ** 2)

        return ((SystemModelEnums.rho_t.value * t_l_i_n)/(t_T_i_m_n + t__i_m_n_f_0)) + ((SystemModelEnums.rho_e.value * e_l_i_n)/(e_T_i_m_n + e__cpt_i_m_n_f_0))

    @classmethod
    def transmit_rate(cls, distances_of_fog, i, m):
        # m = fog
        # i = MUE
        sum_ = 0
        for o in range(SystemModelEnums.M.value):
            if o != m:
                for j in range(SystemModelEnums.K.value):
                    if j != i:
                        sum_ += SystemModelEnums.p__u.value*(distances_of_fog[j][m]**-SystemModelEnums.a.value)* SystemModelEnums.g_u_i_m.value

        return SystemModelEnums.B.value * math.log(1 + ((SystemModelEnums.p__u*(distances_of_fog[i][m]** -SystemModelEnums.a.value) * SystemModelEnums.g_u_i_m.value)
                                                        /(SystemModelEnums.sigma_2.value + sum_)),2)

    @classmethod
    def downlink_rate(cls, distances_of_fog, i, m):
        # m = fog
        # i = MUE
        sum_ = 0
        for o in range(SystemModelEnums.M.value):
            if o != m:
                for j in range(SystemModelEnums.K.value):
                    if j != i:
                        sum_ += SystemModelEnums.p__m.value * (distances_of_fog[j][m] ** -SystemModelEnums.a.value) * SystemModelEnums.g_d_i_m.value

        return SystemModelEnums.B.value * math.log(1 + ((SystemModelEnums.p__m * (
                    distances_of_fog[i][m] ** -SystemModelEnums.a.value) * SystemModelEnums.g_d_i_m.value)
                                                        / (SystemModelEnums.sigma_2.value + sum_)), 2)

    @classmethod
    def eliminating_conflict(cls, fitness, fogs_in_conflict):
        # finding m_star by genetic alg
        ga_instance = pygad.GA(num_generations=100,
                               num_parents_mating=20,  # number of selected parents in each generation
                               fitness_func=fitness,
                               sol_per_pop=20,  # number oh chromosomes
                               num_genes=1,
                               gene_type=int,
                               gene_space=fogs_in_conflict,
                               parent_selection_type="rws",
                               keep_parents=0,
                               crossover_type="single_point",
                               crossover_probability=0.1,
                               mutation_type="random",
                               mutation_probability=0.001,
                               mutation_by_replacement=False)

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        filename = 'genetic'
        ga_instance.save(filename=filename)
        loaded_ga_instance = pygad.load(filename=filename)
        return loaded_ga_instance.best_solution()

    # @classmethod
    # def fitness_of_eliminating_conflict(cls, list_of_conflict_fogs):
    #     # returns m_star
    #     gamma_m = []
    #     for m in list_of_conflict_fogs:
    #         gamma_m.append(gini.task_makes_income_max(m))
    #
    #     m_star_index = np.argmax(gamma_m)
    #     return list_of_conflict_fogs[m_star_index]

    @classmethod
    def fitness_of_eliminating_conflict(cls, list_of_conflict_fogs, conflict_tasks, task_library, distances_of_fogs, mue):
        # returns m_star
        gamma_m = []
        for m in list_of_conflict_fogs:
            gamma = []
            for n in conflict_tasks:
                gamma.append(cls.income_function(task_library, n, distances_of_fogs, mue, m))
            gamma_m.append(max(gamma))

        m_star_index = np.argmax(gamma_m)
        return list_of_conflict_fogs[m_star_index]
