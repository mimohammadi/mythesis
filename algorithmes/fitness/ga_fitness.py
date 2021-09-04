# average utility of task caching bring by d2d sharing
from config.data_generator import Distributions
from config.constants import SystemModelEnums as se, SystemModelRanges as sr
import math
from scipy import integrate as integrate


class GAFitness:
    @classmethod
    def ga_fitness(cls, popularity_of_tasks, caching_probability, task_library, distance_of_mues):
        # print(popularity_of_tasks)
        # print(caching_probability)
        # print(task_library)
        # print(distance_of_mues)
        return - (cls.avg_utility_of_local_task_caching(popularity_of_tasks, caching_probability, task_library) +
                     cls.avg_utility_of_d2d_task_caching(popularity_of_tasks, caching_probability, task_library,
                                                         distance_of_mues))
        # return

    # @classmethod
    # def popularity_of_tasks(cls, number_of_tasks, betta):
    #     return [Distributions.zipf_distribution(n, number_of_tasks, betta) for n in number_of_tasks]

    @classmethod
    def avg_utility_of_local_task_caching(cls, popularity_of_tasks, caching_probability, task_library):
        # p = cls.popularity_of_tasks(SystemModelEnums.N.value, SystemModelEnums.betta.value)
        # q =
        sum_ = 0
        for i in range(se.K.value):
            for n in range(se.N.value):
                t_l_i_n = task_library[n].D__n / se.w.value
                e_l_i_n = se.kappa.value * task_library[n].D__n * se.w.value ** 2
                sum_ += popularity_of_tasks[n] * caching_probability[n] * ((se.rho_t.value * t_l_i_n)
                                                                      + (se.rho_e.value * e_l_i_n))

        # print('avg_utility_of_local_task_caching=')
        # print(round(sum_ / se.K.value, 4))
        return round(sum_ / se.K.value, 4)

    @classmethod
    def avg_utility_of_d2d_task_caching(cls, popularity_of_tasks, caching_probability, task_library, distance_of_mues):
        ### ???
        sum_ = 0
        r = (se.p__u.value / se.D2D_establish_threshold.value) ** (1 / se.a.value)
        for i in range(se.K.value):
            # print('distance_of_mues[i]:')
            # print(distance_of_mues[i])
            x_1 = min(distance_of_mues[i])
            if x_1 == 0:
                distance_of_mues[i].remove(x_1)
                x_n = min(distance_of_mues[i])
                j = distance_of_mues[i].index(x_n)
            else:
                x_n = x_1
                j = distance_of_mues[i].index(x_n)
            for n in range(se.N.value):
                    t_l_i_n = task_library[n].D__n / se.w.value
                    e_l_i_n = se.kappa.value * task_library[n].D__n * se.w.value ** 2
                    e_l_i_n = se.kappa.value * task_library[n].D__n * se.w.value ** 2

                    t_D_i_n = 0
                    e_D_i_n = 0
                    # t_D_i_n = (task_library[n].theta__n * task_library[n].S__n) / cls.avg_transmit_rate(r, distance_of_mues,
                    #                                                                                     i, j,
                    #                                                                                     caching_probability[
                    #                                                                                         n])[0]
                    # e_D_i_n = se.p__u.value * t_D_i_n
                    sum_ += popularity_of_tasks[n] * (1 - caching_probability[n]) * (
                            (se.rho_t.value * (t_l_i_n - t_D_i_n)) + (se.rho_e.value * (e_l_i_n - e_D_i_n)))
        return sum_ / se.K.value

    @classmethod
    def transmit_rate(cls, distance_of_mues, i, j, x_n):  ### channel gail ???
        sum_ = 0
        for z in range(se.K.value):  # len distribution mues transmitting n
            if z != i and z != j:
                for k in range(se.K.value):
                    if k != i and k != j and k != z:
                        if distance_of_mues[z][k] != 0:
                            sum_ += (distance_of_mues[z][k] ** (- se.a.value)) * se.g_d2d_i_j.value
                            print('transmit_rate')
                            print(sum_)
                    # else:
                    #     break
        # print('face: ')
        # print((se.p__u.value * (x_n ** (- se.a.value)) * se.g_d2d_i_j.value))
        # print('body: ')
        # print((se.sigma_2.value + sum_))
        s = ((se.p__u.value * (x_n ** (- se.a.value)) * se.g_d2d_i_j.value) / (se.sigma_2.value + sum_))
        # print(1.0+s)
        t = se.B.value * math.log(1.0 + s, 2)
        # print('t=')
        # print(t)
        return t

    @classmethod
    def association_distance(cls, x_n, q_n):
        # print('qn=')
        # print(q_n)
        # print(2 * math.pi * x_n * se.lambda_.value * q_n * math.exp((- se.lambda_.value) * q_n * math.pi * (x_n ** 2)))
        return 2 * math.pi * x_n * se.lambda_.value * q_n * math.exp((- se.lambda_.value) * q_n * math.pi * (x_n ** 2))

    # @classmethod
    # def f(cls, q_n, distance_of_mues, i, j, x_n):
    #     return cls.transmit_rate(distance_of_mues, i, j, x_n) * cls.association_distance(x_n, q_n)

    @classmethod
    def avg_transmit_rate(cls, r, distance_of_mues, i, j, q_n):
        y = lambda x_n: GAFitness.transmit_rate(distance_of_mues, i, j, x_n) * GAFitness.association_distance(x_n, q_n) #f(q_n, distance_of_mues, i, j, x_n)
        # print(y)
        # rr = cls.f(q_n, distance_of_mues, i, j, x_n)
        # integrate.simpson(rr, x_n)
        res = integrate.quad(y, 0, r)
        # print(r)
        # print('res = ')
        # print(res)
        # print('-----')
        return res


def f(q_n, distance_of_mues, i, j, x_n):
    return GAFitness.transmit_rate(distance_of_mues, i, j, x_n) * GAFitness.association_distance(x_n, q_n)
    #return GAFitness.association_distance(x_n, q_n)
