from config.constants import SystemModelEnums as se
from algorithmes.fitness.ga_fitness import GAFitness
import math


class TaskOffloadingOpt:
    @classmethod
    def task_offloading_opt_problem(cls, a__i_m, y_m_i, n, task_library, f__i_m, distance_from_fog, distance_from_cloud):
        sum_ = 0
        for i in range(se.K.value):
            for m in range(se.M.value):
                u_fog_i, u_cloud_i = cls.utility_for_fog_and_cloud_offloading(i, m, n, task_library, f__i_m, distance_from_fog, distance_from_cloud)
                sum_ += a__i_m[i][m] * (y_m_i[i] * u_fog_i + (1 - y_m_i[i]) * u_cloud_i)

        return - sum_

    @classmethod
    def utility_for_fog_and_cloud_offloading(cls, i, m, n, task_library, f__i_m, distance_from_fog, distance_from_cloud):
        t_l_i_n = task_library[n].D__n / se.w.value
        e_l_i_n = se.kappa.value * task_library[n].D__n * (se.w.value ** 2)
        t_f_i_m_n = task_library[n].D__n / f__i_m[i][m]
        e_cpt_i_m_n = se.kappa_server * task_library[n].D__n * (f__i_m[i][m] ** 2)
        t_u_i_m_n = task_library[n].s__n / cls.transmit_rate_of_requester_associated_with_fog(m, i, distance_from_fog)
        e_u_i_m_n = se.p__u.value * t_u_i_m_n
        t_d_m_i_n = (task_library[n].theta__n * task_library[n].s__n) / cls.downlink_rate(m, i, distance_from_fog)
        e_d_m_i_n = se.p__m.value * t_d_m_i_n
        t_fog_i_m_n = t_u_i_m_n + t_d_m_i_n + t_f_i_m_n
        e_fog_i_m_n = e_u_i_m_n + e_cpt_i_m_n + e_d_m_i_n
        u_fog_i = se.rho_t.value * (t_l_i_n - t_fog_i_m_n) + se.rho_e.value * (e_l_i_n - e_fog_i_m_n)
        t_c_i_n = task_library[n].D__n / se.f__0.value
        e_cpt_i_n = se.kappa_server.value * task_library[n].D__n * (se.f__0.value ** 2)
        t_u_m_o = task_library[n].S__n / cls.cloud_transmit_rate(m, distance_from_cloud)
        t_d_o_m = (task_library[n].theta__n * task_library[n].s__n) / cls.cloud_downlink_rate(m, distance_from_cloud)
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
                sum_ += se.p__m.value * (distance_from_cloud[mm] ** (- se.a.value)) * se.g_m_i_m.value

        return se.B.value * math.log(
            1 + ((se.p__m.value * (distance_from_cloud[m] ** (- se.a.value)) * se.g_m_i_m.value) / (
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
