import numpy as np

from config.data_generator import Distributions
from models.mue import MUE
from models.task import Task
from models.fog import Fog
from config.constants import SystemModelEnums as se, SystemModelRanges as sr
from algorithmes.fitness.ga_fitness import GAFitness
import algorithmes.fitness.ga_fitness as af
from algorithmes.genetic_alg import GeneticAlg as ga
from algorithmes.gini_coefficient_alg import GiniCoefficientBasedAlg as gc_alg
from algorithmes.fitness.task_offloading_optimization_fitness import TaskOffloadingOpt as tolo
from algorithmes import task_offloading_opt_ga as toa
from config.data_generator import Distributions as dist


S_n = [Distributions.random_distribution(sr.Min_Task_Size.value, sr.Max_Task_Size.value) for i in range(se.N.value)]
D_n = [Distributions.random_distribution(sr.Min_Task_Cmp.value, sr.Max_Task_Cmp.value) for i in range(se.N.value)]
P_n = [Distributions.zipf_distribution(n, se.N.value, se.betta.value) for n in range(se.N.value)]
task_library = [Task(D_n[i], S_n[i], se.theta.value, P_n[i]) for i in range(se.N.value)]

xx, yy = Distributions.homogenous_poisson_point_process_distribution(sr.x_Min_FNs.value, sr.x_Max_FNs.value,
                                                                     sr.y_Min_FNs.value, sr.y_Max_FNs.value,
                                                                     se.lambda_.value)


set_of_mues = [MUE(xx[i], yy[i]) for i in range(se.K.value)]


distances_of_mues = [[((set_of_mues[i].x_ - set_of_mues[j].x_) ** 2 + (set_of_mues[i].y_ - set_of_mues[j].y_) ** 2)
                      ** (1 / 2) for i in range(se.K.value)] for j in range(se.K.value)]


fog_x = Distributions.uniform_distribution(sr.x_Min_FNs.value, sr.x_Max_FNs.value, se.M.value)
fog_y = Distributions.uniform_distribution(sr.y_Min_FNs.value, sr.y_Max_FNs.value, se.M.value)
set_of_fogs = [Fog(round(fog_x[i], 4), round(fog_y[i], 4), se.f__0.value) for i in range(se.M.value)]
distance_from_fog = [[((set_of_mues[i].x_ - set_of_fogs[m].x_) ** 2 + (set_of_mues[i].y_ - set_of_fogs[m].y_) ** 2)
                      ** (1 / 2) for m in range(se.M.value)] for i in range(se.K.value)]


distance_from_cloud = [((set_of_fogs[m].x_ - 100000)**2 + (set_of_fogs[m].y_ - 100000)**2) ** (1/2) for m in range(se.M.value)]


def fitness(solution0, solution_idx0):
    return GAFitness.ga_fitness(P_n, solution0, task_library, distances_of_mues)


def task_offloading_fitness(solution1, solution_idx1):
    a__i_m, y_i, f__i_m = split_chromosome(solution1)
    return tolo.task_offloading_opt_problem(a__i_m, y_i, set_of_mues, task_library, f__i_m, distance_from_fog, distance_from_cloud)


def split_chromosome(solution2):
    a__i_m = [[[0 for j in range(len(set_of_mues[col].request_set))] for row in range(se.M.value)] for col in range(se.K.value)]
    y__ = [[0 for j in range(len(set_of_mues[col].request_set))] for col in range(se.K.value)]
    f__i_m = [[[0 for j in range(len(set_of_mues[col].request_set))] for row in range(se.M.value)] for col in range(se.K.value)]
    c = int(len(solution2)/3)
    for i in range(se.K.value):
        for j in range(c):
            for n in range(len(set_of_mues[i].request_set)):
                if int(solution2[j]) > se.M.value or int(solution2[j]) < 1:
                    solution2[j] = dist.random_distribution(1, se.M.value)
                a__i_m[i][int(solution2[j]) - 1][n] = 1  # [mue][fog]
                y__[i][n] = int(solution2[c+j])
                if y__[i][n] == 0:  # fog
                    f__i_m[i][int(solution2[j]) - 1][n] = solution2[2*c + j]
                else:
                    f__i_m[i][int(solution2[j]) - 1][n] = 0

    return np.array(a__i_m), np.array(y__), np.array(f__i_m)


if __name__ == '__main__':
    # GA
    t_max = 100

    solution, solution_fitness, solution_idx = ga.genetic_alg(t_max,
                                                              200,
                                                              fitness,
                                                              gene_type=[float],
                                                              gene_space=[[0, 1]],
                                                              number_of_solutions=200,
                                                              num_genes=[se.N.value],
                                                              crossover_probability=0.9,
                                                              mutation_probability=0.01,
                                                              on_constrain=af.on_mutation)

    for n in range(se.N.value):
        task_library[n].q__n = solution[n]
    set_of_mues_of_fogs = [Distributions.random_distribution(1, se.K__max.value) for i in range(se.M.value)]

    request = []
    cacher = []
    number_of_all_requests = 0
    for n in range(len(task_library)):
        request, y = Distributions.homogenous_poisson_point_process_distribution(0, se.K.value, 0, se.K.value, se.lambda_.value)
        number_of_all_requests += len(request)

        for i in range(len(request)):
            set_of_mues[int(request[i])].request_set.append(n)
            print('set_of_mues[int(request[i])].request_set =')
            print(set_of_mues[int(request[i])].request_set)

    print(number_of_all_requests)
    final_result, final_result_fitness, final_result_idx = ga.genetic_alg(
        iteration_num=2, parent_num=2, fitness=task_offloading_fitness, gene_type=[int, int, float],
        number_of_solutions=2, num_genes=[number_of_all_requests, number_of_all_requests, number_of_all_requests],
        crossover_probability=0.9, mutation_probability=0.01,
        gene_space=[[1, se.M.value], [0, 1], [0, se.f__0.value]],
        on_constrain=toa.on_mutation)








    # a_i_m = gc_alg.gini_
    # ue_association_alg(set_of_mues_of_fogs, request_set_of_mues, set_of_mues, distance_from_fog,
    #                                         D_n, set_of_fogs, task_library)


