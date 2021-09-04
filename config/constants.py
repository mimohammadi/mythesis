from enum import Enum
from collections import namedtuple
import math

name_tuple = namedtuple("tuple", ['value', 'description'])


class BaseEnum(Enum):
    @classmethod
    def get_value_list(cls):
        return list(map(lambda x: x.value, cls))

    @classmethod
    def get_value_dict(cls):
        return dict(zip(list(a.value for a in cls), list(a.description for a in cls)))

    @property
    def description(self):
        return self._value_.description

    @property
    def value(self):
        return self._value_.value


class SystemModelEnums(BaseEnum):
    M = name_tuple(2, "Number of FNs")
    K = name_tuple(5, "Number of MUEs")
    K__max = name_tuple(2, "maximum number of accessible MUEs in each FN")
    N = name_tuple(5, "Number of Tasks")
    B = name_tuple(1, "MHz Wireless Bandwidth")
    a = name_tuple(3, "Path Lose Factor")
    sigma_2 = name_tuple(10**(-10), "mw -100 dmb Background Noise")
    w = name_tuple(900, "MHz CPU Clock speed of MUEs")
    f__0 = name_tuple(4000, "MHz CPU Clock speed of Fog Server and Cloud Server")
    p__u = name_tuple(0.1, "mw Transmit Power of MUEs")
    p__m = name_tuple(0.2, "mw Transmit Power of FNs")
    rho_e = name_tuple(0.016, "/J Revenue Coefficient Per Unit of Energy")
    rho_t = name_tuple(0.5, "/s Revenue Coefficient Per Unit of Saving Delay")
    betta = name_tuple(0.8, "The Popularity Parameter")
    theta = name_tuple(0, "ratio of data size after to data size before computation of task")
    lambda_ = name_tuple(0.1, "MUE distribution density")
    kappa = name_tuple(10**(-25), "energy effective switched capacitance of MUE")
    kappa_server = name_tuple(10**(-25), "energy effective switched capacitance of Fog server")
    g_d2d_i_j = name_tuple(math.exp(1), "small-scale fading coefficient")
    D2D_establish_threshold = name_tuple(10**(-25), "0? dBm D2d establishment threshold") #???
    g_u_i_m = name_tuple(10**(-25), "? channel gain of uplink")  #???
    g_d_i_m = name_tuple(10**(-25), "? channel gain of downlink")  #???


class SystemModelRanges(BaseEnum):
    x_Min_FNs = name_tuple(0, "minimum of x of location of FNs")
    x_Max_FNs = name_tuple(600, "maximum of x of location of FNs")
    y_Min_FNs = name_tuple(0, "minimum of y of location of FNs")
    y_Max_FNs = name_tuple(600, "maximum of y of location of FNs")
    Min_Task_Size = name_tuple(10, "MB minimum data size of tasks")
    Max_Task_Size = name_tuple(30, "MB maximum data size of tasks")
    Min_Task_Cmp = name_tuple(1000, "MHz minimum required computational resource of tasks")
    Max_Task_Cmp = name_tuple(6000, "MHz maximum required computational resource of tasks")
