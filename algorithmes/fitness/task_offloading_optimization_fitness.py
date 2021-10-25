from config.constants import SystemModelEnums as se
from algorithmes.fitness.ga_fitness import GAFitness
import math
import numpy as np
from config.data_generator import Distributions as dist
import pandas as pd


class TaskOffloadingOpt:
    @classmethod
    def task_offloading_opt_problem(cls, a__i_m, y_m_i, set_of_mues, task_library, distance_from_fog, distance_from_cloud):
        sum_ = 0
        fog_tasks = [[] for m in range(se.M.value)]
        sum_req = 0
        output_tasks = [[] for m in range(se.M.value)]
        # print('a__i_m')
        # print(a__i_m)
        df = pd.read_excel('algorithmes/request.xlsx')
        req_set = df.to_numpy()
        new_req = [[int(req_set[i][j]) for i in range(np.size(req_set, 0)) if np.isnan(req_set[i][j]) == False] for j in
                   range(np.size(req_set, 1))]
        # print(new_req)
        for i in range(se.K.value):
            if new_req[i] != []:
                for index_n, n in enumerate(new_req[i]):
                    if n != -1:
                        sum_req += 1
                        for m in range(se.M.value):
                            if a__i_m[i][m][index_n] == 1:
                                sum_f_i_m = 0
                                enter_time = 0
                                latency = 0
                                f_i_m = 0
                                # if len(fog_tasks[m]) != 0:
                                #     sum_f_i_m = np.sum([fog_tasks[m][col][2] for col in range(len(fog_tasks[m]))])
                                if len(fog_tasks[m]) < se.N__max.value:
                                    f_i_m = se.f__0.value / se.N__max.value

                                if len(fog_tasks[m]) == se.N__max.value:
                                    ll = [fog_tasks[m][col][4] for col in range(len(fog_tasks[m]))]
                                    min_late = np.min(ll)
                                    min_late_indx = ll.index(min_late)
                                    output_tasks[m].append(fog_tasks[m][min_late_indx])
                                    # f_i_m = fog_tasks[m][min_late_indx][2]
                                    f_i_m = se.f__0.value / se.N__max.value
                                    latency = min_late
                                    enter_time = min_late
                                    fog_tasks[m].remove(fog_tasks[m][min_late_indx])

                                time_of_comp = task_library[n].D__n / f_i_m
                                u_fog_i, u_cloud_i = cls.utility_for_fog_and_cloud_offloading(i, m, n, y_m_i[i][index_n], task_library, f_i_m, distance_from_fog, distance_from_cloud)
                                # print('u_fog_i')
                                # print(u_fog_i)
                                # print('u_cloud_i')
                                # print(u_cloud_i)
                                fog_tasks[m].append([index_n, n, f_i_m, time_of_comp, latency + u_fog_i, enter_time])
                                sum_ += (((1 - y_m_i[i][index_n]) * (latency + u_fog_i)) + (y_m_i[i][index_n] * u_cloud_i))
                                # print('sum_')
                                # print(sum_)
                            # return 1 / sum_
        print('sum_ = ' + str(sum_))
        if sum_ != 0:
            return 1 / sum_
        return -10000000

    @classmethod
    def utility_for_fog_and_cloud_offloading(cls, i, m, n, y_i_m, task_library, f__i_m, distance_from_fog, distance_from_cloud):
        # print(n)
        u_fog_i = 0
        u_cloud_i = 0
        t_l_i_n = task_library[n].D__n / se.w.value
        # print('t_l_i_n')
        # print(t_l_i_n)
        e_l_i_n = se.kappa.value * task_library[n].D__n * (se.w.value ** 2)
        # print('e_l_i_n')
        # print(e_l_i_n)
        t_r = cls.transmit_rate_of_requester_associated_with_fog(m, i, distance_from_fog)
        if t_r != 0:
            t_u_i_m_n = task_library[n].S__n / t_r
        else:
            t_u_i_m_n = 0
        e_u_i_m_n = se.p__u.value * t_u_i_m_n
        t_d = cls.downlink_rate(m, i, distance_from_fog)
        if t_d != 0:
            t_d_m_i_n = (task_library[n].theta__n * task_library[n].S__n) / t_d
        else:
            t_d_m_i_n = 0
        e_d_m_i_n = se.p__m.value * t_d_m_i_n
        #fog
        if y_i_m == 0 and f__i_m > 0:
            t_f_i_m_n = task_library[n].D__n / f__i_m
            e_cpt_i_m_n = se.kappa_server.value * task_library[n].D__n * (f__i_m ** 2)

            t_fog_i_m_n = t_u_i_m_n + t_d_m_i_n + t_f_i_m_n
            e_fog_i_m_n = e_u_i_m_n + e_cpt_i_m_n + e_d_m_i_n
            u_fog_i = se.rho_t.value * (t_l_i_n - t_fog_i_m_n) + se.rho_e.value * (e_l_i_n - e_fog_i_m_n)
        elif y_i_m == 1:
            t_c_i_n = task_library[n].D__n / se.f__0.value
            e_cpt_i_n = se.kappa_server.value * task_library[n].D__n * (se.f__0.value ** 2)
            t_u = cls.cloud_transmit_rate(m, distance_from_cloud)
            if t_u != 0:
                t_u_m_o = task_library[n].S__n / t_u
            else:
                t_u_m_o = 0
            t_cl = cls.cloud_downlink_rate(m, distance_from_cloud)
            if t_cl != 0:
                t_d_o_m = (task_library[n].theta__n * task_library[n].S__n) / t_cl
            else:
                t_d_o_m = 0
            t_c = t_u_m_o + t_d_o_m
            t_cloud_i_m_n = t_u_i_m_n + t_d_m_i_n + t_c_i_n + t_c
            e_cloud_i_m_n = e_u_i_m_n + e_d_m_i_n + e_cpt_i_n
            u_cloud_i = se.rho_t.value * (t_l_i_n - t_cloud_i_m_n) + se.rho_e.value * (e_l_i_n - e_cloud_i_m_n)
        return u_fog_i, u_cloud_i

    # @classmethod
    # def utility_of_cloud_offloading(cls, i, m, n, task_library, distance_from_fog):
    #     t_c_i_n = task_library[n].D__n / se.f__0.value
    #     e_cpt_i_n = se.kappa_server.value * task_library[n].D__n * (se.f__0.value ** 2)
    #     t_u_i_m_n = task_library[n].s__n / cls.transmit_rate_of_requester_associated_with_fog(m, i, distance_from_fog)
    #     e_u_i_m_n = se.p__u.value * t_u_i_m_n
    #     t_d_m_i_n = (task_library[n].theta__n * task_library[n].s__n) / cls.downlink_rate(m, i, distance_from_fog)
    #     e_d_m_i_n = se.p__m.value * t_d_m_i_n

    @classmethod
    def cloud_downlink_rate(cls, m, distance_from_cloud):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                sum_ += se.p__m.value * (distance_from_cloud[mm] ** (- se.a.value)) * se.g_d_i_m.value

        # print('ii = ' + str(1 + ((se.p__m.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_d_i_m.value) / (
        #             se.sigma_2.value + sum_))))
        return 1.25 * 10 ** -7 * (10 ** (se.B.value * math.log(
            1 + ((se.p__m.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_d_i_m.value) / (
                    se.sigma_2.value + sum_)), 2) / 10))

    @classmethod
    def cloud_transmit_rate(cls, m, distance_from_cloud):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                sum_ += se.p__u.value * (distance_from_cloud[mm] ** (- se.a.value)) * se.g_u_i_m.value

        # print('ii = ' + str(1 + ((se.p__u.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_u_i_m.value) / (
        #                 se.sigma_2.value + sum_))))
        return 1.25 * 10 ** -7 * (10 ** (se.B.value * math.log(1 + ((se.p__u.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_u_i_m.value) / (
                        se.sigma_2.value + sum_)), 2) / 10))

    @classmethod
    def transmit_rate_of_requester_associated_with_fog(cls, m, i, distance_from_fog):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                for ii in range(se.K.value):
                    if ii != i:
                        sum_ += se.p__u.value * (distance_from_fog[ii][mm] ** (- se.a.value)) * se.g_u_i_m.value

        # print('ii = ' + str(1 + ((se.p__u.value * (distance_from_fog[i][m] ** (- se.a.value)) * se.g_u_i_m.value)/(se.sigma_2.value + sum_))))
        return 1.25 * 10 ** -7 * (10 ** (se.B.value * math.log(1 + ((se.p__u.value * (distance_from_fog[i][m] ** (- se.a.value)) * se.g_u_i_m.value)/(se.sigma_2.value + sum_)), 2) / 10))

    @classmethod
    def downlink_rate(cls, m, i, distance_from_fog):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                for ii in range(se.K.value):
                    if ii != i:
                        sum_ += se.p__m.value * (distance_from_fog[ii][mm] ** (- se.a.value)) * se.g_d_i_m.value

        print('ll = ' + str(1 + ((se.p__m.value * (distance_from_fog[i][m] ** (- se.a.value)) * se.g_d_i_m.value)/(se.sigma_2.value + sum_))))
        return 1.25 * 10 ** -7 * (10 ** (se.B.value * math.log(1 + ((se.p__m.value * (distance_from_fog[i][m] ** (- se.a.value)) * se.g_d_i_m.value)/(se.sigma_2.value + sum_)), 2) / 10))
