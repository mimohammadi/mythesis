from config.constants import SystemModelEnums
from algorithmes.fitness.task_offloading_opt import GiniFunctions as gf
import numpy as np
import math


class GiniCoefficientBasedAlg:
    @classmethod
    def gini_mue_association_alg(cls, set_of_mues_in_each_fn, request_set_of_mues,
                                 set_of_mues, distances_of_fog, list_of_d__n_of_tasks,
                                 set_of_fogs, library_of_tasks):

        global task_library, distances_of_fogs
        task_library = library_of_tasks
        distances_of_fogs = distances_of_fog
        conflict = 1
        fn_sum_gamma = []
        gamma_fn = []
        a_i_m = []
        # sorted_mues_of_fn = []
        for m in range(SystemModelEnums.M.value):
            sum_of_gamma = 0
            gamma_i_m = []
            # sorted_mues = []
            for i in range(len(set_of_mues_in_each_fn[m])):
                gamma_i_m_n = []
                R_i = request_set_of_mues[set_of_mues_in_each_fn[m][i]]
                for n in range(R_i):
                    gamma_i_m_n.append([gf.income_function(R_i, R_i[n], distances_of_fog, set_of_mues_in_each_fn[m][i], m), i, set_of_mues_in_each_fn[m][i], R_i])
                    # sorted_mues.append([gf.income_function(R_i, R_i[n], distances_of_fog, set_of_mues_in_each_fn[m][i], m), i, set_of_mues_in_each_fn[m][i]])

                gamma_of_mue = max([_[0] for _ in gamma_i_m_n])
                gamma_i_m.append(gamma_i_m[gamma_i_m.index(gamma_of_mue)])

                # sorted_mues.sort()
                # sorted_mues.reverse()

                sum_of_gamma += gamma_of_mue
            gamma_i_m.sort()
            fn_sum_gamma[m] = sum_of_gamma
            gamma_fn.append(gamma_i_m)
            # sorted_mues_of_fn.append(sorted_mues)

        gini_m = []
        kappa_star_m = []
        kappa_star_mues_m = []
        # b_i
        for m in range(SystemModelEnums.M.value):
            # sum_of_gamma_j_m = []
            b_i = []
            # for i in np.array(gamma_fn[m]).T[1]:
            for i in range(len(gamma_fn[m])): # i is based on sort
                sum_of_gamma_j = 0
                for j in range(i):
                    sum_of_gamma_j += gamma_fn[m][j][0]

                b_i.append(sum_of_gamma_j/fn_sum_gamma[m])
            sum_b_i = 0
            for i in range(len(set_of_mues_in_each_fn[m]) - 1):
                sum_b_i += b_i[i]

            gini = 1 - ((1/len(set_of_mues_in_each_fn[m]))*(1 + 2 * sum_b_i))
            gini_m.append(gini)

            # gamma as modified wight factor
            gamma_i = min(SystemModelEnums.f__0.value/np.argmax(list_of_d__n_of_tasks), len(set_of_mues_in_each_fn[m]),
                          SystemModelEnums.K__max.value)

            k_star_m = min((1/gini)+gamma_i*(len(set_of_mues_in_each_fn[m]) - math.ceil(1/gini)),
                           len(set_of_mues_in_each_fn[m]))

            kappa_star = []
            kappa_star_mues = []
            for i in range(len(set_of_mues_in_each_fn[m]), len(set_of_mues_in_each_fn[m]) + 1 - k_star_m, -1):
                kappa_star.append(gamma_fn[m][i]) # [2] means mue
                kappa_star_mues.append(gamma_fn[m][i][2])
            kappa_star_m.append(kappa_star)
            kappa_star_mues_m.append(kappa_star_mues)
            a_i_m = cls.update_association_policy(kappa_star_mues_m)

        while conflict:
            for i in set_of_mues:
                global mue
                mue = i
                count_mue_in_fogs = 0
                conflict_fogs = []
                global conflict_tasks
                conflict_tasks = []
                for m in range(len(kappa_star_m)):
                    kappa_stars = np.array(kappa_star_m[m])
                    if list(kappa_stars[:, 2]).count(i) > 0:
                        count_mue_in_fogs += 1
                        conflict_fogs.append(set_of_fogs[m]) # fogs are sorted base index?
                        conflict_tasks.append(kappa_star_m[m][list(kappa_stars[:, 2]).index(i)][3])

                if count_mue_in_fogs > 1:
                    conflict = 1
                    # gf.eliminating_conflict(gf.fitness_of_eliminating_conflict())
                    m_star = gf.fitness_of_eliminating_conflict(conflict_fogs)
                    a_i_m = cls.update_association_policy(kappa_star_mues_m, m_star, i, a_i_m)
                    #####
                    conflict_fogs.remove(m_star)
                    an_fn_lost_mue = conflict_fogs[0]
                    disassociated_mues = []
                    for j in range(len(gamma_fn[an_fn_lost_mue])):
                        if kappa_star_mues_m[an_fn_lost_mue].count(gamma_fn[an_fn_lost_mue][j][2]) == 0:
                            disassociated_mues.append([gamma_fn[an_fn_lost_mue][j][2], gamma_fn[an_fn_lost_mue][j][0]])

                    #####
                    # disassociated_mues = []
                    # for j in a_i_m:
                    #     if j.count(1) == 0:
                    #         for m in range(len(gamma_fn)):
                    #             if m != m_star:
                    #                 fn_mues = np.array(gamma_fn[m])
                    #                 if fn_mues[:, 2].count(j) > 0:
                    #                     idx1 = list(fn_mues[:, 0]).index(j)
                    #                     array = np.array(disassociated_mues)
                    #                     if array[:, 0].count(j) > 0:
                    #                         idx = list(array[:, 0]).index(j)
                    #                         if disassociated_mues[idx][1] < fn_mues[idx1, 0]:
                    #                             disassociated_mues[idx][1] = fn_mues[idx1, 0]
                    #                             disassociated_mues[idx][2] = m
                    #                     else:
                    #                         disassociated_mues.append([j, fn_mues[idx1, 0], m])
                    #####

                    max_income_mue = max(np.amax(np.array(disassociated_mues[:, 1])))
                    index = list(np.array(disassociated_mues[:, 1])).index(max_income_mue)
                    selected_mue = disassociated_mues[index][0]
                    #####
                    # a_i_m = cls.update_association_policy(a_i_m=a_i_m, reselected_mue=selected_mue,
                    #                                       reselected_fog=disassociated_mues[index][2])
                    #####
                    a_i_m = cls.update_association_policy(a_i_m=a_i_m, reselected_mue=selected_mue,
                                                          reselected_fog=an_fn_lost_mue)
                    kappa_star_mues_m[an_fn_lost_mue].append(selected_mue)
                    idx_mue = list(np.array(gamma_fn[an_fn_lost_mue][:, 2])).index(selected_mue)
                    kappa_star_m[an_fn_lost_mue].append(gamma_fn[an_fn_lost_mue][idx_mue])

                else:
                    conflict = 0

        return a_i_m

    # @classmethod
    # def task_makes_income_max(cls, m):
    #     gamma = []
    #     for n in conflict_tasks:
    #         gamma.append(gf.income_function(task_library, n, distances_of_fogs, mue, m))
    #     return max(gamma)

    @classmethod
    def update_association_policy(cls, kappa_m=None, m_star=None, mue_m_star=None, a_i_m=None, reselected_mue=None,
                                  reselected_fog=None):
        if m_star is not None and mue_m_star is not None and a_i_m is not None:
            a_i_m[mue_m_star][m_star] = 1
            for m in range(len(a_i_m[mue_m_star])):
                a_i_m[mue_m_star][m] = 0
        elif reselected_mue is not None and reselected_fog is not None:
            a_i_m[reselected_mue][reselected_fog] = 1
        else:
            a_i_m = [[0 for col in range(SystemModelEnums.M.value)] for row in range(SystemModelEnums.K.value)]
            for m_ in range(len(kappa_m)):
                if m_star is None or m_ != m_star:
                    for i_ in range(SystemModelEnums.K.value): # mues = range 1 ... n
                        if len(kappa_m[m_]) <= i_ + 1:
                            if kappa_m[m_][i_] == i_ + 1:
                                a_i_m[i_][m_] = 1
                            else:
                                a_i_m[kappa_m[m_][i_]][m_] = 1
        return a_i_m


