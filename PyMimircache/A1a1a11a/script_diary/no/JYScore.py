# coding=utf-8


import math


class JYScore:
    def __init__(self, distr_list, type=None):
        self.N = len(distr_list)
        self.distr_const_list = [0] * self.N
        self.distr_const_list[0] = distr_list[0]


        for i in range(1, len(distr_list)):
            self.distr_const_list[i] = distr_list[i] + self.distr_const_list[i-1]


        if type == "count":
            self.replace_one_entry = self.replace_one_entry_count
            s = sum(self.distr_const_list)
            for i in range(len(distr_list)):
                self.distr_const_list[i] /= s

        else:
            s = sum(self.distr_const_list)
            for i in range(len(distr_list)):
                self.distr_const_list[i] /= s

            self.replace_one_entry = self.replace_one_entry_test1
            # self.replace_one_entry = self.replace_one_entry_old
            # self.replace_one_entry = self.replace_one_entry_default

        # print(self.distr_const_list[:200])
        # print("{} {}".format(len(self.distr_const_list), sum(self.distr_const_list)))
        print(self.distr_const_list[:20])


        self.distr_change_list = self.distr_const_list[:]
        self.symmetric = False
        self.info = 0




    def replace_one_entry_default(self, old, new):
        # old_KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, False)
        new_info = 0
        # new_info = self.info
    
    
        # if self.distr_const_list[old] > 1e-8 and self.distr_change_list[old] > 1e-8:
        #     new_info = self.distr_const_list[old] * math.log(
        #         self.distr_const_list[old] / self.distr_change_list[old]) * (old - new)
    
        # this is far different from cdf KL,
        # 1. the sum of self.distr_change_list is not normalized
        # 2. 1 / self.N is not normalized either, in normalized form self.N should be the sumed cdf rd count
    
        if old < new:
            # [old, new) reduce one
            for i in range(old, new):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    new_info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] - 1 / self.N > 1e-8:
                    new_info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] - 1 / self.N))
    
                if self.symmetric:
                    if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                        new_info -= self.distr_change_list[i] * math.log(
                            self.distr_change_list[i] / self.distr_const_list[i])
                    if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] - 1 / self.N > 1e-8:
                        new_info += self.distr_change_list[i] * math.log(
                            (self.distr_change_list[i] - 1 / self.N) / self.distr_const_list[i])
    
                self.distr_change_list[i] -= 1 / self.N
        else:
            # [new, old) add one
            for i in range(new, old):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    new_info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] + 1 / self.N > 1e-8:
                    new_info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] + 1 / self.N))
    
                if self.symmetric:
                    if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                        new_info -= self.distr_change_list[i] * math.log(
                            self.distr_change_list[i] / self.distr_const_list[i])
                    if self.distr_change_list[i] + 1 / self.N > 1e-8 and self.distr_const_list[i] > 1e-8:
                        new_info += self.distr_change_list[i] * math.log(
                            (self.distr_change_list[i] + 1 / self.N) / self.distr_const_list[i])
    
                self.distr_change_list[i] += 1 / self.N

        self.info = new_info
        return new_info

    def replace_one_entry_old(self, old, new):
        # this is far different from cdf KL,
        # 1. the sum of self.distr_change_list is not normalized
        # 2. 1 / self.N is not normalized either, in normalized form self.N should be the sumed cdf rd count

        if old < new:
            # [old, new) reduce one
            for i in range(old, new):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    self.info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] - 1 / self.N > 1e-8:
                    self.info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] - 1 / self.N))

                self.distr_change_list[i] -= 1 / self.N
        else:
            # [new, old) add one
            for i in range(new, old):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    self.info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] + 1 / self.N > 1e-8:
                    self.info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] + 1 / self.N))
                self.distr_change_list[i] += 1 / self.N

        return self.info

    def replace_one_entry_test1(self, old, new):
        # old_KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, False)
        new_info = 0
        new_info = self.info

        # if self.distr_const_list[old] > 1e-8 and self.distr_change_list[old] > 1e-8:
        #     new_info = self.distr_const_list[old] * math.log(
        #         self.distr_const_list[old] / self.distr_change_list[old]) * (old - new)

        # this is far different from cdf KL,
        # 1. the sum of self.distr_change_list is not normalized
        # 2. 1 / self.N is not normalized either, in normalized form self.N should be the sumed cdf rd count
        change1 = 1/100
        # change2 = 1/self.N
        change2 = 1/1000

        # new_info = new - old
        if old < new:
            for i in range(old, new):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    new_info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] - change1 > 1e-8:
                    new_info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] - change1))

                self.distr_change_list[i] -= change2

        if new < old:
            for i in range(new, old):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    new_info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] + change1 > 1e-8:
                    new_info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] + change1))

                self.distr_change_list[i] += change2

        self.info = new_info
        return new_info


    def replace_one_entry_count(self, old, new):
        # old_KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, False)
        new_info = self.info

        # this is far different from cdf KL,
        # 1. the sum of self.distr_change_list is not normalized
        # 2. 1 / self.N is not normalized either, in normalized form self.N should be the sumed cdf rd count
        change = 1/self.N

        if old < new:
            for i in range(old, new):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    new_info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] - change > 1e-8:
                    new_info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] - change ))

                self.distr_change_list[i] -= change
                # self.distr_change_list[i] -= 1/100

        if new < old:
            for i in range(new, old):
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] > 1e-8:
                    new_info -= self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / self.distr_change_list[i])
                if self.distr_const_list[i] > 1e-8 and self.distr_change_list[i] + change > 1e-8:
                    new_info += self.distr_const_list[i] * math.log(
                        self.distr_const_list[i] / (self.distr_change_list[i] + change ))

                self.distr_change_list[i] += change
                # self.distr_change_list[i] += 1/100

        self.info = new_info
        return new_info
