from config.data_generator import Distributions
from config.constants import SystemModelEnums, SystemModelRanges


class Task:
    def __init__(self, d__n, s__n, theta__n, p__n, q__n=None):
        self.D__n = d__n
        self.S__n = s__n
        self.theta__n = theta__n
        self.q__n = q__n
        self.p__n = p__n

    # def create_task(self):
    #     return {"D__n": self.d__n, "S__n": self.s__n, "Theta__n": self.theta__n,
    #             "q__n": self.q__n, "p__n": self.p__n}


# class TaskLibrary(Task):
#
#     def __init__(self, d__n, s__n, theta__n, number_of_tasks):
#         super().__init__(d__n, s__n, theta__n)
#         self.number_of_tasks = number_of_tasks
#
#     def create_data_size_of_tasks(self):
#         return [Distributions.random_distribution(SystemModelRanges.Min_Task_Size, SystemModelRanges.Max_Task_Size) for
#                 i in range(self.number_of_tasks)]
#
#     def create_cmp_of_tasks(self):
#         return [Distributions.random_distribution(SystemModelRanges.Min_Task_Cmp, SystemModelRanges.Max_Task_Cmp) for
#                 i in range(self.number_of_tasks)]
#
#     def create_task_library(self):
#         tasks = []
#         s__i = self.create_data_size_of_tasks()
#         d__i = self.create_cmp_of_tasks()
#         for i in range(self.number_of_tasks):
#             tasks.append(Task.create_task(Task(d__i[i], s__i[i], SystemModelEnums.theta)))
#
#         return tasks
