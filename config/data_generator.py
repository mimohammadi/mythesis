from random import normalvariate
import numpy as np
from numpy.random import randint, uniform, poisson, random_integers


class Distributions:
    @classmethod
    def uniform_distribution(cls, _low, _high, _size):
        return uniform(_low, _high, _size)

    @classmethod
    def random_distribution(cls, _low, _high):
        return random_integers(_low, _high)

    @classmethod
    def normal_distribution(cls, _mean, _sigma):
        return round(normalvariate(_mean, _sigma), 4)

    @classmethod
    def homogenous_poisson_point_process_distribution(cls, _x_min, _x_max, _y_min,
                                                      _y_max, _lambda):
        x_delta = _x_max - _x_min
        y_delta = _y_max - _y_min  # rectangle dimensions
        area_total = x_delta * y_delta

        # Simulate a Poisson point process
        numb_points = poisson(_lambda * area_total)  # Poisson number of points
        xx = x_delta * uniform(10**(-20), 1, numb_points) + _x_min  # x coordinates of Poisson points
        yy = y_delta * uniform(10**(-20), 1, numb_points) + _y_min  # y coordinates of Poisson points
        xxx = [round(num, 4) for num in xx]
        yyy = [round(num, 4) for num in yy]
        if not xxx:
            cls.homogenous_poisson_point_process_distribution(_x_min, _x_max, _y_min,
                                                              _y_max, _lambda)
        # if len(xxx) == 1:
        #     cls.homogenous_poisson_point_process_distribution(_x_min, _x_max, _y_min,
        #                                                       _y_max, _lambda)
        return xxx, yyy

    @classmethod
    def h_ppp(cls, lambda_, size_, row_=None, col_=None):
        return np.random.poisson(lambda_, (row_, col_, size_))

    @classmethod
    def zipf_distribution(cls, _n, _number_of_tasks, _betta):
        _sum = 0
        for m in range(_number_of_tasks):
            _sum += 1/((m + 1) ** _betta)

        return round((1/((_n + 1) ** _betta))/_sum, 4)
