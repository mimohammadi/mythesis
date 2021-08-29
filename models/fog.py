from config.data_generator import Distributions
from config.constants import SystemModelEnums, SystemModelRanges


class Fog:
    def __init__(self, x_, y_, f_0):
        self.x_ = x_
        self.y_ = y_
        self.f_0 = f_0

    # def create_fog(self):
    #     return {"x": self.x_, "y": self.y_}


# class ListOfFogs(Fog):
#
#     def __init__(self, x_, y_, number_of_fogs):
#         super().__init__(x_, y_)
#         self.number_of_fogs = number_of_fogs
#
#     def create_location_of_fogs(self):
#         return [Distributions.uniform_distribution
#                 (SystemModelRanges.x_Max_FNs.value,
#                  SystemModelRanges.y_Max_FNs.value,
#                  SystemModelEnums.M.value)
#                 for i in range(self.number_of_fogs)]
#
#     def create_list_of_fogs(self):
#         fogs = []
#         xx, yy = self.create_location_of_fogs()
#         for i in range(self.number_of_fogs):
#             fogs.append(Fog.create_fog(Fog(xx[i], yy[i])))
#
#         return fogs
