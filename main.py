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


S_n = [Distributions.random_distribution(sr.Min_Task_Size.value, sr.Max_Task_Size.value) for i in range(se.N.value)]
D_n = [Distributions.random_distribution(sr.Min_Task_Cmp.value, sr.Max_Task_Cmp.value) for i in range(se.N.value)]
P_n = [Distributions.zipf_distribution(n, se.N.value, se.betta.value) for n in range(se.N.value)]
task_library = [Task(D_n[i], S_n[i], se.theta.value, P_n[i]) for i in range(se.N.value)]
# print(task_library)
# MUES
# x = [], y = []
# for i in range(se.K.value):
xx, yy = Distributions.homogenous_poisson_point_process_distribution(sr.x_Min_FNs.value, sr.x_Max_FNs.value,
                                                                     sr.y_Min_FNs.value, sr.y_Max_FNs.value,
                                                                     se.lambda_.value)

# print(xx)
# print(yy)
set_of_mues = [MUE(xx[i], yy[i]) for i in range(se.K.value)]
# print('set_of_mues')
# print(len(set_of_mues))
distances_of_mues = [[((set_of_mues[i].x_ - set_of_mues[j].x_) ** 2 + (set_of_mues[i].y_ - set_of_mues[j].y_) ** 2)
                      ** (1 / 2) for i in range(se.K.value)] for j in range(se.K.value)]

# print('distances_of_mues:')
# print(distances_of_mues)
# print('++++++++')
fog_x = Distributions.uniform_distribution(sr.x_Min_FNs.value, sr.x_Max_FNs.value, se.M.value)
fog_y = Distributions.uniform_distribution(sr.y_Min_FNs.value, sr.y_Max_FNs.value, se.M.value)
set_of_fogs = [Fog(round(fog_x[i], 4), round(fog_y[i], 4), se.f__0.value) for i in range(se.M.value)]
distance_from_fog = [[((set_of_mues[i].x_ - set_of_fogs[m].x_) ** 2 + (set_of_mues[i].y_ - set_of_fogs[m].y_) ** 2)
                      ** (1 / 2) for m in range(se.M.value)] for i in range(se.K.value)]


distance_from_cloud = [((set_of_fogs[m].x_ - 100000)**2 + (set_of_fogs[m].y_ - 100000)**2) ** (1/2) for m in range(se.M.value)]

# print('distances_of_mues:')
# print(distances_of_mues)
# print('******')

# def print_hi(name):
#     print(f'Hi, {name}')


def fitness(solution0, solution_idx0):
    print('sol = ')
    print(solution0)
    return GAFitness.ga_fitness(P_n, solution0, task_library, distances_of_mues)


def task_offloading_fitness(solution1, solution_idx1):
    a__i_m = [[0 for row in range(se.M.value)] for col in range(se.K.value)]
    y = [0 for col in range(se.K.value)]
    f__i_m = [[0 for row in range(se.M.value)] for col in range(se.K.value)]
    c = len(solution1)/4
    for i in range(int(c)):
        a__i_m[solution1[i] - 1][solution1[i+1] - 1] = 1
        y[solution1[i] - 1] = solution1[i+2]
        f__i_m[solution1[i] - 1][solution1[i + 1] - 1] = solution1[i+3]

    return tolo.task_offloading_opt_problem(a__i_m, y, n, task_library, f__i_m, distance_from_fog, distance_from_cloud)


if __name__ == '__main__':
    # print_hi('PyCharm')
    # Tasks
    # S_n = [Distributions.random_distribution(sr.Min_Task_Size.value, sr.Max_Task_Size.value) for i in range(se.N.value)]
    # D_n = [Distributions.random_distribution(sr.Min_Task_Cmp.value, sr.Max_Task_Cmp.value) for i in range(se.N.value)]
    # P_n = [Distributions.zipf_distribution(n, se.N.value, se.betta.value) for n in range(se.N.value)]
    # task_library = [Task(D_n[i], S_n[i], se.theta.value, P_n[i]) for i in range(se.N.value)]
    # print(task_library)
    # # MUES
    # # x = [], y = []
    # # for i in range(se.K.value):
    # xx, yy = Distributions.homogenous_poisson_point_process_distribution(sr.x_Min_FNs.value, sr.x_Max_FNs.value,
    #                                                                      sr.y_Min_FNs.value, sr.y_Max_FNs.value,
    #                                                                      se.lambda_.value)
    # print(xx)
    # print(yy)
    # set_of_mues = [MUE(xx[i], yy[i]) for i in range(se.K.value)]
    # distances_of_mues = [[((set_of_mues[i].x_ - set_of_mues[j].x_) ** 2 + (set_of_mues[i].y_ - set_of_mues[j].y_) ** 2)
    #                       ** (1/2) for i in range(se.K.value)] for j in range(se.K.value)]
    #
    # print('distances_of_mues:')
    # print(distances_of_mues)
    # print('++++++++')
    # fog_x = Distributions.uniform_distribution(sr.x_Min_FNs.value, sr.x_Max_FNs.value, se.M.value)
    # fog_y = Distributions.uniform_distribution(sr.y_Min_FNs.value, sr.y_Max_FNs.value, se.M.value)
    # set_of_fogs = [Fog(fog_x[i], fog_y[i], se.f__0.value) for i in range(se.M.value)]
    # distance_from_fog = [[((set_of_mues[i].x_ - set_of_fogs[m].x_) ** 2 + (set_of_mues[i].y_ - set_of_fogs[m].y_) ** 2)
    #                       ** (1/2) for m in range(se.M.value)] for i in range(se.K.value)]

    # GA
    t_max = 100
    # solution, solution_fitness, solution_idx = ga.genetic_alg(t_max,
    #                                                           100,
    #                                                           fitness,
    #                                                           100,
    #                                                           se.N.value,
    #                                                           0.1,
    #                                                           0.1)
    solution, solution_fitness, solution_idx = ga.genetic_alg(t_max,
                                                              200,
                                                              fitness,
                                                              gene_type=[float],
                                                              gene_space=[[0, 1]],
                                                              number_of_solutions=200,
                                                              num_genes=[se.N.value],
                                                              crossover_probability=0.1,
                                                              mutation_probability=0.01,
                                                              on_constrain=af.on_mutation)
    print('final solution = ')
    print(solution)
    for n in range(se.N.value):
        task_library[n].q__n = solution[n]
    set_of_mues_of_fogs = [Distributions.random_distribution(1, se.K__max.value) for i in range(se.M.value)]
    print('set_of_mues_of_fogs = ')
    print(set_of_mues_of_fogs)
    
    # print('solution:')
    # print(solution)
    request = []
    cacher = []
    for n in task_library:
        request, cacher = Distributions.homogenous_poisson_point_process_distribution(0, se.K.value, 0, se.K.value,
                                                                                      n.q__n * se.lambda_.value)
        print(request)
        print(cacher)
        if len(request) != 0:
            req_nm = 0
            if len(request) > 20:
                req_nm = Distributions.random_distribution(0, 20)
            else:
                req_nm = Distributions.random_distribution(0, len(request))
            for i in range(0, req_nm):
                set_of_mues[int(request[i])].request_set.append(n)

        if len(cacher) != 0:
            if set_of_mues[int(cacher[0])].cached_task is None:
                set_of_mues[int(cacher[0])].cached_task = int(cacher[0])
                cacher.remove(cacher[0])

    final_result, final_result_fitness, final_result_idx = ga.genetic_alg(
        iteration_num=10, parent_num=10, fitness=task_offloading_fitness, gene_type=[[int, int, int, float]],
        number_of_solutions=10, num_genes=[[se.K.value, se.M.value, se.K.value, se.K.value]],
        crossover_probability=0.9, mutation_probability=0.01,
        gene_space=[[[0, se.K.value-1], [0, se.M.value-1], [0, 1], [0, se.K.value-1]]],
        on_constrain=toa.on_mutation)








    # a_i_m = gc_alg.gini_
    # ue_association_alg(set_of_mues_of_fogs, request_set_of_mues, set_of_mues, distance_from_fog,
    #                                         D_n, set_of_fogs, task_library)


