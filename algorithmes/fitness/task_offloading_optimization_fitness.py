from config.constants import SystemModelEnums as se
from algorithmes.fitness.ga_fitness import GAFitness
import math


class TaskOffloadingOpt:
    @classmethod
    def task_offloading_opt_problem(cls, a__i_m, y_m_i, set_of_mues, task_library, f__i_m, distance_from_fog, distance_from_cloud):
        sum_ = 0
        for i in range(se.K.value):
            for m in range(se.M.value):
                for index_n, n in enumerate(set_of_mues[i].request_set):
                    print(set_of_mues[i].request_set)
                    print(index_n)
                    if y_m_i[i] != [] and f__i_m[i][m] != [] and a__i_m[i][m] != []:
                        print('y_[i]=')
                        print(len(y_m_i[i]))
                        print(len(f__i_m[i][m]))
                        u_fog_i, u_cloud_i = cls.utility_for_fog_and_cloud_offloading(i, m, n, y_m_i[i][index_n], task_library, f__i_m[i][m][index_n], distance_from_fog, distance_from_cloud)
                        sum_ += a__i_m[i][m] * (y_m_i[i][index_n] * u_fog_i + (1 - y_m_i[i][index_n]) * u_cloud_i)

        return - sum_

    @classmethod
    def utility_for_fog_and_cloud_offloading(cls, i, m, n, y_i_m, task_library, f__i_m, distance_from_fog, distance_from_cloud):
        # print(n)
        u_fog_i = 0
        u_cloud_i = 0
        t_l_i_n = task_library[n].D__n / se.w.value
        e_l_i_n = se.kappa.value * task_library[n].D__n * (se.w.value ** 2)
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
        if y_i_m == 0:
            t_f_i_m_n = task_library[n].D__n / f__i_m
            e_cpt_i_m_n = se.kappa_server.value * task_library[n].D__n * (f__i_m ** 2)


            t_fog_i_m_n = t_u_i_m_n + t_d_m_i_n + t_f_i_m_n
            e_fog_i_m_n = e_u_i_m_n + e_cpt_i_m_n + e_d_m_i_n
            u_fog_i = se.rho_t.value * (t_l_i_n - t_fog_i_m_n) + se.rho_e.value * (e_l_i_n - e_fog_i_m_n)
        else:
            t_c_i_n = task_library[n].D__n / se.f__0.value
            e_cpt_i_n = se.kappa_server.value * task_library[n].D__n * (se.f__0.value ** 2)
            t_u = cls.cloud_transmit_rate(m, distance_from_cloud)
            if t_u != 0:
                t_u_m_o = task_library[n].S__n / t_u
            else:
                t_u_m_o = 0
            t_cl = cls.cloud_downlink_rate(m, distance_from_cloud)
            if t_cl != 0:
                t_d_o_m = (task_library[n].theta__n * task_library[n].s__n) / t_cl
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

        return se.B.value * math.log(
            1 + ((se.p__m.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_d_i_m.value) / (
                    se.sigma_2.value + sum_)), 2)

    @classmethod
    def cloud_transmit_rate(cls, m, distance_from_cloud):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                sum_ += se.p__u.value * (distance_from_cloud[mm] ** (- se.a.value)) * se.g_u_i_m.value

        return se.B.value * math.log(1 + ((se.p__u.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_u_i_m.value) / (
                        se.sigma_2.value + sum_)), 2)

    @classmethod
    def transmit_rate_of_requester_associated_with_fog(cls, m, i, distance_from_fog):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                for ii in range(se.K.value):
                    if ii != i:
                        sum_ += se.p__u.value * (distance_from_fog[ii][mm] ** (- se.a.value)) * se.g_u_i_m.value

        return se.B.value * math.log(1 + ((se.p__u.value * (distance_from_fog[i][m] ** (- se.a.value)) * se.g_u_i_m.value)/(se.sigma_2.value + sum_)), 2)

    @classmethod
    def downlink_rate(cls, m, i, distance_from_fog):
        sum_ = 0
        for mm in range(se.M.value):
            if mm != m:
                for ii in range(se.K.value):
                    if ii != i:
                        sum_ += se.p__m.value * (distance_from_fog[ii][mm] ** (- se.a.value)) * se.g_d_i_m.value

        return se.B.value * math.log(1 + ((se.p__m.value * (distance_from_fog[i][m] ** (- se.a.value)) * se.g_d_i_m.value)/(se.sigma_2.value + sum_)), 2)
