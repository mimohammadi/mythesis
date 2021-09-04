from config.data_generator import Distributions
from models.mue import MUE
from models.task import Task
from models.fog import Fog
from config.constants import SystemModelEnums as se, SystemModelRanges as sr
from algorithmes.fitness.ga_fitness import GAFitness
from algorithmes.genetic_alg import GeneticAlg as ga
from algorithmes.gini_coefficient_alg import GiniCoefficientBasedAlg as gc_alg


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

# print('distances_of_mues:')
# print(distances_of_mues)
# print('******')

# def print_hi(name):
#     print(f'Hi, {name}')


def fitness(solution, solution_idx):
    print('sol = ')
    print(solution)
    return GAFitness.ga_fitness(P_n, solution, task_library, distances_of_mues)


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
                                                              mutation_probability=0.01)
    print('final solution = ')
    print(solution)
    for n in range(se.N.value):
        task_library[n].q__n = solution[n]
    set_of_mues_of_fogs = [Distributions.random_distribution(1, se.K__max.value) for i in range(se.M.value)]
    # print('solution:')
    # print(solution)
    # request_set_of_mues = Distributions.h_ppp
    # a_i_m = gc_alg.gini_
    # ue_association_alg(set_of_mues_of_fogs, request_set_of_mues, set_of_mues, distance_from_fog,
    #                                         D_n, set_of_fogs, task_library)


