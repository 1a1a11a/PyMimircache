# coding=utf-8
"""
this module is work on 09/01
about reuse distance similarity measurement with cloudphysics data

initial data choice: w92 w106 w78




"""


from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.bin.conf import *
from concurrent.futures import ProcessPoolExecutor, as_completed


import os
import sys
import math
import time
from collections import deque, defaultdict
import matplotlib

from PyMimircache.profiler.twoDPlots import draw2d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.abspath(__file__))
# from JYScore import JYScore
from PyMimircache.A1a1a11a.script_diary.JYScore import JYScore


NO_REPEAT_PLOTTING = False
DISTRI_INC_FOLD = 1


class distribution:
    def __init__(self, N, l, cdf):
        self.distr_list = [0] * N
        self.rd_count_list = [0] * N
        self.N = N
        self.cdf = cdf
        for i in l:
            if i > N-2:
                self.distr_list[-1] += 1 / N
                self.rd_count_list[-1] += 1
            else:
                self.distr_list[i] += 1 / N
                self.rd_count_list[i] += 1

        if abs(sum(self.distr_list) -1) > 0.00001:
            raise RuntimeError("distribution sum {} not equal 1".format(sum(self.distr_list)))

        if self.cdf:
            for i in range(1, len(self.distr_list)):
                self.distr_list[i] += self.distr_list[i - 1]
            dist_list_sum = sum(self.distr_list)
            for i in range(len(self.distr_list)):
                self.distr_list[i] = self.distr_list[i] / dist_list_sum

    def get_rd_count_list(self):
        return self.rd_count_list

    def get_distr_list(self):
        return self.distr_list

    def get_N(self):
        return self.N

    def replace_one_entry(self, old, new):
        if self.cdf:
            raise RuntimeError("cdf not supported")

        if old > self.N-2:
            old = -1
        if new > self.N-2:
            new = -1
        self.distr_list[old] -= 1 / self.N
        self.distr_list[new] += 1 / self.N
        if self.distr_list[old] < 0:
            raise RuntimeError("probability {} < 0".format(self.distr_list[old]))



    def cal_KL_Divergence(self, dist2, symmetric=False):
        """
        calculate KL between
        :param dist2:
        :param symmetric:
        :return:
        """
        if self.N != dist2.get_N():
            raise RuntimeError("two distribution have different elements {} {}".format(self.N, dist2.get_N()))

        dist2_list = dist2.get_distr_list()
        KL = 0
        for i in range(len(self.distr_list)):
            if self.distr_list[i] !=0 and dist2_list[i] != 0:
                KL += self.distr_list[i] * math.log(self.distr_list[i] / dist2_list[i])
                if symmetric:
                    KL += dist2_list[i] * math.log(dist2_list[i] / self.distr_list[i])
        return KL




class KLDivergenceCalculator:
    def __init__(self, N, rd_list, symmetric, cdf, distr_list_size=-1):
        if distr_list_size == -1:
            distr_list_size = N

        self.rd_count_list = [0] * N
        for rd in rd_list:
            if rd > N-1:
                rd = N - 1
            self.rd_count_list[rd] += 1             # only used with cdf

        if cdf:
            for i in range(1, len(self.rd_count_list)):
                self.rd_count_list[i] += self.rd_count_list[i-1]

        self.distr_const_list = distribution(N, rd_list, cdf).get_distr_list()

        self.distr_const_list.extend([0] * (distr_list_size))

        self.distr_change_list = self.distr_const_list[:]              # only used without cdf

        self.N = N
        self.symmetric = symmetric
        self.cdf = cdf
        # self.KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, symmetric)
        self.KL = 0

    @staticmethod
    def cal_KL_Divergence_with_rd_list(rd1, rd2, cdf, symmetrc):
        """
        calculate KL using two lists, each list is a reuse distance list
        :param rd1:
        :param rd2:
        :param cdf:
        :param symmetrc:
        :return:
        """
        assert len(rd1) == len(rd2), "two lists have difference size {}: {}".format(len(rd1), len(rd2))
        return distribution(len(rd1), rd1, cdf).cal_KL_Divergence(distribution(len(rd2), rd2, cdf), symmetrc)


    @staticmethod
    def cal_KL_Divergence(l1, l2, symmetric):
        """
        calculate KL between two list, each one is a distribution list
        :param l1:
        :param l2:
        :param symmetric:
        :return:
        """
        KL = 0
        for i in range(len(l1)):
            if l1[i] > 1e-8 and l2[i] > 1e-8:
                # if any of them equals 0, then the xlogx = 0, so no need for calculation
                KL += l1[i] * math.log(l1[i] / l2[i])
                if symmetric:
                    KL += l2[i] * math.log(l2[i] / l1[i])
        return KL


    def replace_one_entry(self, old, new):

        # old_KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, False)
        new_info = 0

        if not self.cdf:
            if self.distr_const_list[old] > 1e-8 and self.distr_change_list[old] > 1e-8:
                self.KL -= self.distr_const_list[old] * math.log(self.distr_const_list[old] / self.distr_change_list[old])
            if self.distr_const_list[old] > 1e-8 and self.distr_change_list[old] - 1/self.N > 1e-8:
                self.KL += self.distr_const_list[old] * math.log(self.distr_const_list[old] / (self.distr_change_list[old] - 1 / self.N))
            if self.distr_const_list[new] > 1e-8 and self.distr_change_list[new] > 1e-8:
                self.KL -= self.distr_const_list[new] * math.log(self.distr_const_list[new] / self.distr_change_list[new])
            if self.distr_const_list[new] > 1e-8 and self.distr_change_list[new] + 1/self.N > 1e-8:
                self.KL += self.distr_const_list[new] * math.log(self.distr_const_list[new] / (self.distr_change_list[new] + 1 / self.N))

            if self.symmetric:
                if self.distr_const_list[old] > 1e-8 and self.distr_change_list[old] > 1e-8:
                    self.KL -= self.distr_change_list[old] * math.log(self.distr_change_list[old] / self.distr_const_list[old])
                if self.distr_const_list[old] > 1e-8 and self.distr_change_list[old] - 1 / self.N > 1e-8:
                    self.KL += self.distr_change_list[old] * math.log((self.distr_change_list[old] - 1 / self.N) / self.distr_const_list[old])
                if self.distr_const_list[new] > 1e-8 and self.distr_change_list[new] > 1e-8:
                    self.KL -= self.distr_change_list[new] * math.log(self.distr_change_list[new] / self.distr_const_list[new])
                if self.distr_const_list[new] > 1e-8 and self.distr_change_list[new] + 1 / self.N > 1e-8:
                    self.KL += self.distr_change_list[new] * math.log((self.distr_change_list[new] + 1 / self.N) / self.distr_const_list[new])

            self.distr_change_list[old] -= 1 / self.N
            self.distr_change_list[new] += 1 / self.N
            # print(self.distr_change_list)
            # if change > 0.001:
            # print("{:.6f}({:6d}) {:.6f}({:6d}) \t\t\t {:.6f} {:.6f} {:.6f} {:.6f}".format(self.distr_change_list[old], old,
            #                                                                               self.distr_change_list[new], new,
            #                                                         change1, change2, change3, change4))
            # new_KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, self.symmetric)
            # if abs(self.KL - new_KL) > 0.00001:
            #     print("{}\n{}, {}".format(self.distr_const_list, self.distr_change_list, self.N))
            #     raise RuntimeError("{} != {}".format(self.KL, new_KL))

        else:
            # because we need to recalculate probability for each bin anyway, so we use easy way for calculating KL

            if old < new:
                # [old, new) reduce one
                for i in range(old, new):
                    self.rd_count_list[i] -= 1
                    # self.distr_change_list[i] -= 1 / self.N

            else:
                # [new, old) add one
                for i in range(new, old):
                    self.rd_count_list[i] += 1
                    # self.distr_change_list[i] += 1 / self.N

            # print("replace {} with {} sum {} {}".format(old, new, sum(self.rd_count_list), self.rd_count_list))
            rd_count_sum = sum(self.rd_count_list)
            self.distr_change_list = [ rd / rd_count_sum for rd in self.rd_count_list]

            # the sum of self.distr_change_list[i] is no longer 1, so we need to normalize
            # s = sum(self.distr_change_list)
            # for i in range(len(self.distr_change_list)):
            #     self.distr_change_list[i] = self.distr_change_list[i] / s


            # now calculate KL
            self.KL = KLDivergenceCalculator.cal_KL_Divergence(self.distr_const_list, self.distr_change_list, self.symmetric)

        return self.KL

    def get_KL(self):
        return self.KL


def run_KL(dat, dat_type, window=2560, cdf=False, symmetric=False,
           time_mode="v", real_time_rd_range=1000000,
           skip_ratio=0, reader=None,
           figname=None, folder=None):
    """

    :param dat:
    :param dat_type:
    :param window:
    :param cdf:
    :param symmetric:
    :param time_mode:
    :param real_time_rd_range: the max rd can be used in real time
    :param skip_ratio:
    :param reader:
    :param figname:
    :param folder:
    :return:
    """

    if folder is None and '/' in figname:
        folder = figname[: figname.rfind('/')]

    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if figname is None:
        figname = "{}KL_{}_{}{}{}.png".format(folder+"/" if folder else "",
                                              os.path.basename(dat), window,
                                              "_cdf" if cdf else "",
                                              "_symmetric" if symmetric else "")

    if NO_REPEAT_PLOTTING and os.path.exists(figname):
        return

    if reader is None:
        # reader = vscsiReader(dat)
        reader = get_reader(dat, dat_type)

    # rd_list = LRUProfiler(reader).use_precomputedRD()
    rd_list = CLRUProfiler(reader, no_load_rd=True).get_reuse_distance()
    if skip_ratio != 0:
        rd_list = rd_list[int(len(rd_list)*skip_ratio):]

    KL_list = []
    pos = 0

    # get distribution P
    l = []
    deq_Q = deque()
    if time_mode == 'v':
        for i in range(window):
            l.append(rd_list[i])
            deq_Q.append(rd_list[i])
        KL_cal = KLDivergenceCalculator(window, l, symmetric, cdf)

    elif time_mode == "r":
        ts_list = reader.get_timestamp_list()
        if skip_ratio != 0:
            ts_list = ts_list[int(len(ts_list) * skip_ratio):]
        pos = 0
        while ts_list[pos] - ts_list[0] < window:
            l.append(rd_list[pos])
            deq_Q.append(rd_list[pos])
            pos += 1
        print("real time mode, initial rd_list size: {}".format(len(l)))
        KL_cal = KLDivergenceCalculator(len(l), l, symmetric, cdf, distr_list_size=real_time_rd_range)
    else:
        print("ERROR")
        return

    # l_distribution = distribution(N, l, cdf).get_distr_list()

    # plot_distribution
    dl = KL_cal.distr_const_list
    for i in range(1, len(dl)):
        dl[i] = dl[i-1] + dl[i]
        if 1 - dl[i] < 0.001:
            break

    # plot the rd distribution for the first window
    # draw2d(dl[:i+1], figname="1010CPHY_KL_W92Test/rd_distribution_interval{}_skip{}.png".format(window // 1000000, skip_ratio),
    #        xlabel="reuse distance", ylabel="count percentage",
    #        # xlimit=(-20, len(rd_distr_list) + 20), ylimit=(0, 1),
    #        print_info=False)
    # return

    KL_list.append(KL_cal.get_KL())

    if time_mode == "v":
        # when time mode is "v", each time move one step, contrast to real time, each step move window time_units(microseconds)
        for i in range(window, len(rd_list)):
            old = deq_Q.popleft()
            new = rd_list[i]
            deq_Q.append(new)

            # if old > window-2 or old == -1:
            #     old = window - 1
            # if new > window-2 or new == -1:
            #     new = window - 1

            # new 1010
            if old > window * DISTRI_INC_FOLD - 2:
                old = -1
            if new > window * DISTRI_INC_FOLD - 2:
                new = -1

            # KL = distribution(N, l, cdf).cal_KL_Divergence(distribution(N, deq_Q, cdf), symmetric)
            KL = KL_list[-1]
            if old != new:
                KL = KL_cal.replace_one_entry(old, new)

            KL_list.append(KL)

    elif time_mode == "r":
        last_time = ts_list[pos]

        for i in range(pos, len(rd_list)):

            old = deq_Q.popleft()
            new = rd_list[i]
            deq_Q.append(new)

            if old > real_time_rd_range-1:
                old = -1
            if new > real_time_rd_range-1:
                new = -1

            if old != new:
                KL_cal.replace_one_entry(old, new)

            if ts_list[i] - last_time >= window:
                last_time = ts_list[i]
                KL_list.append(KL_cal.get_KL())


    plot_0901(KL_list, window, cdf, symmetric, figname, xlabel="time({})".format(time_mode))


def run_KL_sampling(dat, N, jump_distance=-1, cdf=True, symmetric=False, figname=None, folder=None):
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if figname is None:
        figname = "{}KL_{}_{}{}{}_sampling{}.png".format(folder+"/" if folder else "",
                                                          os.path.basename(dat), N,
                                                          "_cdf" if cdf else "",
                                                          "_symmetric" if symmetric else "",
                                                         jump_distance)

    if NO_REPEAT_PLOTTING and os.path.exists(figname):
        return
    reader = VscsiReader(dat)
    rd = CLRUProfiler(reader).get_reuse_distance()

    if jump_distance == -1:
        jump_distance = N

    KL_list = [0]

    # get distribution P
    p_rd_list = []
    for i in range(N):
        p_rd_list.append(rd[i])

    p_distribution = distribution(N, p_rd_list, cdf)

    # old
    # q_rd_list = []
    # i = 0
    # while i < len(rd):
    #     q_rd_list.append(rd[i])
    #     if len(q_rd_list) == N:
    #         q_distribution = distribution(N, q_rd_list, cdf)
    #         KL = p_distribution.cal_KL_Divergence(q_distribution, symmetric)
    #
    #         if abs(sum(q_distribution.get_distr_list()) - 1) > 1e-8:
    #             raise RuntimeError("distribution sum not 1")
    #
    #         # if i < 60 and i > 50:
    #         #     print("{} replacing {} with {}, \tKL {} sample distribution {}".format(i, "x", rd[i],
    #         #                                                                        KL, q_distribution.get_distr_list()))
    #         # if i>128:
    #         #     break
    #
    #         KL_list.append(KL)
    #         q_rd_list.clear()
    #         i += jump_distance
    #     i += 1

    # new
    begin = 0
    end = N
    while end < len(rd):
        q_distribution = distribution(N, rd[begin:end], cdf)
        KL = p_distribution.cal_KL_Divergence(q_distribution, symmetric)
        KL_list.append(KL)
        begin += jump_distance
        end = begin + N

    plot_0901(KL_list, N, cdf, symmetric, figname)


def run_JY(dat, N, figname=None, folder=None, type=None):
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if figname is None:
        figname = "{}{}JY_{}_{}.png".format(folder+"/" if folder else "",
                                            type if type is not None else "",
                                                          os.path.basename(dat), N)

    if NO_REPEAT_PLOTTING and os.path.exists(figname):
        return
    reader = VscsiReader(dat)
    rd = CLRUProfiler(reader).get_reuse_distance()

    JY_list = [0]
    JY_list2 = [0]
    # print(figname)

    # get distribution P
    l = []
    deq_Q = deque()
    for i in range(N):
        l.append(rd[i])
        deq_Q.append(rd[i])

    # l_distribution = distribution(N, l, cdf).get_distr_list()

    if type is None:
        jy_calculator = JYScore(distribution(N, l, False).get_distr_list(), type=None)
    elif type == "count":
        jy_calculator = JYScore(distribution(N, l, False).get_rd_count_list(), type=type)
    else:
        print("ERROR")
        sys.exit(1)

    for i in range(N, len(rd)):
        old = deq_Q.popleft()
        new = rd[i]
        deq_Q.append(new)

        if old > N-2 or old == -1:
            old = N - 1
        if new > N-2 or new == -1:
            new = N - 1


        jy = JY_list[-1]
        jy2 = JY_list2[-1]
        if old != new:
            # jy, jy2 = jy_calculator.replace_one_entry(old, new)
            jy = jy_calculator.replace_one_entry(old, new)

        # if i > 128:
        #     break
        #
        # if i > 50 and i < 60:
        #     print("{} replacing {} with {}, \tKL {} distribution {}".format(i, old, new, KL, KL_cal.distr_change_list))

        JY_list.append(jy)
        JY_list2.append(jy2)
    plot_0901(JY_list, N, None, None, figname)
    # plot_0901(JY_list2, N, None, None, figname.replace("JY", "JY2"))







def plot_0901(KL_list, N, cdf, symmetric, figname, xlabel="time", ylabel="KL Score", plot_partial=False):
    # print("size {}, {}".format(len(KL_list), KL_list[:100]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(KL_list)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0%}'.format((x + 1) / len(KL_list))))
    plt.text(0, plt.ylim()[1]*0.8, "N={}, cdf={}, symmetric={}".format(N, cdf, symmetric))
    plt.savefig(figname)
    # plt.savefig(figname.replace("png", "pdf"))
    plt.clf()
    print("overall plot done {}".format(figname))

    if plot_partial:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(KL_list[:len(KL_list)//20])
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0%}'.format((x + 1) / len(KL_list))))
        plt.text(0, plt.ylim()[1]*0.8, "N={}, cdf={}, symmetric={}".format(N, cdf, symmetric))
        figname2 = os.path.dirname(figname) + "/" + "partial_" + os.path.basename(figname)
        if figname2[0] == "/":
            figname2 = figname2[1:]
        plt.savefig(figname2)
        plt.clf()
        print("partial plot done {}".format(figname))


########################### RUNNABLE ###########################

def mytest1():
    # run_KL("small", "cphy", 25600, False, False, figname="smallTest.png")
    # for i in [256, 2560, 25600]:             # , 256000, 2560000
    #     run_KL("92", "cphy", i, False, False, figname="w92Test_{}_increasedDistrList.png".format(i))
    reader = VscsiReader("testData/test.vscsi")

    # for t in [60, 600, 1800, 3600, 7200, 3600*6]:
    for t in [100, 500, 1000, 2000, 5000, 10000]:
        for sr in [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3]:
            # run_KL("92", "cphy", t * 1000000, False, False, "r", 10000000,
            run_KL("92", "cphy", t, False, False, "v", 0,
                   skip_ratio=sr,           reader=reader,
                   figname="1016-CPHY-SanityTest/KL_{}_skip{}.png".format(t, sr))


def run_parallel(FOLDER_NAME = "0904KL-PARAM", type="KL_sampling"):
    TRACE_DIR, NUM_OF_THREADS = initConf("cphy", "variable")
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

    with ProcessPoolExecutor(max_workers=NUM_OF_THREADS) as ppe:
        futures = {}
        for dat in ["w92_vscsi1.vscsitrace", "w106_vscsi1.vscsitrace", "w78_vscsi1.vscsitrace"]:
            DAT = "{}/{}".format(TRACE_DIR, dat)
            if os.path.exists(DAT):
                print("yes {}".format(DAT))
            else:
                print("no {}".format(DAT))

            for i in range(3, 12):
                for cdf in [True, False]:
                    for symmetric in [True, False]:
                        if type == "KL":
                            futures[ppe.submit(run_KL, DAT, 2**i*50, cdf, symmetric, None, FOLDER_NAME)] = (2**i*50, cdf, symmetric)
                        elif type == "KL_sampling":
                            futures[ppe.submit(run_KL_sampling, DAT, 2**i*50, -1, cdf, symmetric, None, FOLDER_NAME)] = (2**i*50, -1, cdf, symmetric)
                        elif type == "JY":
                            continue

            if type == "JY":
                for i in range(2, 12):
                    futures[ppe.submit(run_JY, DAT, 2**i*50, None, FOLDER_NAME)] = 2**i*50


        finished_count = 0
        for _ in as_completed(futures):
            finished_count += 1
            print("{}/{}".format(finished_count, len(futures)))



def experiments():
    # run_parallel(FOLDER_NAME="0904KL-sampling", type="KL_sampling")
    # run_parallel(FOLDER_NAME="0904KL", type="KL")
    # run_parallel(FOLDER_NAME="0904JY-1DN", type="JY")
    # run_parallel(FOLDER_NAME="0904JY-1D100", type="JY")
    run_parallel(FOLDER_NAME="0906JY-COUNT", type="JY")


def run_for_lab_aux():
    count = 1
    # for dat in ["w92_vscsi1.vscsitrace", "w106_vscsi1.vscsitrace", "w78_vscsi1.vscsitrace"]:
    for dat_num in range(105, 20, -1):
        dat = "w{}_vscsi1.vscsitrace".format(dat_num)
        for i in range(1, 8):
            for cdf in [1, 0]:
                for symmetric in [1, 0]:
                    print("task{}=python3 0901CloudPhysics.py {} {} {} {}".format(count, dat, i, cdf, symmetric))
                    count += 1





if __name__ == "__main__":
    t1 = time.time()
    TRACE_DIR, NUM_OF_THREADS = initConf("cphy", "variable")

    mytest1()
    # plt.text(plt.xlim()[1]*0.2, plt.ylim()[1]*0.30, "place\nholder", {"fontsize": 72})
    # plt.savefig("holder.png")
    sys.exit(1)

    # run_JY(dat="../../data/trace.vscsi", N=20)
    # run_JY(dat="../../data/trace.vscsi", N=20, type="count")
    # run_JY(dat="{}/w92_vscsi1.vscsitrace".format(TRACE_DIR), N=200, type="count")
    # run_JY(dat="{}/w92_vscsi1.vscsitrace".format(TRACE_DIR), N=200, type=None)
    # run_JY(dat="{}/w106_vscsi1.vscsitrace".format(TRACE_DIR), N=200)
    # run_KL(dat="{}/w92_vscsi1.vscsitrace".format(TRACE_DIR), N=2560, cdf=True, symmetric=False)
    # run_KL(dat="../../data/trace.vscsi", N=2560, cdf=False, symmetric=False)
    # run_KL(dat="../../data/trace.vscsi", N=60, cdf=True, symmetric=False)
    # plot_0901(dat="../../data/trace.vscsi", N=2000, cdf=True, symmetric=False)
    # plot_0901(dat="../../data/trace.vscsi", N=2000, cdf=False, symmetric=False)
    # run_KL(dat="../../data/trace.vscsi", N=120, cdf=True, symmetric=False)
    # run_KL_sampling(dat="{}/w92_vscsi1.vscsitrace".format(TRACE_DIR), N=2560, jump_distance=-2000, cdf=True, symmetric=False)
    # run_KL_sampling(dat="{}/w106_vscsi1.vscsitrace".format(TRACE_DIR), N=25600, jump_distance=-1, cdf=True, symmetric=False)
    # run_KL_sampling(dat="../../data/trace.vscsi", N=2560, jump_distance=1, cdf=False, symmetric=False)
    # run_KL_sampling(dat="../../data/trace.vscsi", N=60, jump_distance=-59, cdf=True, symmetric=False)
    # run_KL_sampling(dat="../../data/trace.vscsi".format(TRACE_DIR), N=120, jump_distance=-119, cdf=True, symmetric=False)
    # plot_KL_sampling(dat="../../data/trace.vscsi".format(TRACE_DIR), N=120, jump_distance=120, cdf=True, symmetric=False)
    # test_params()
    # run1()
    # experiments()
    # run_for_lab_aux()
    # plot_0901("/home/jyan254/scratch/DATA/cphy/{}".format(sys.argv[1]),
    #         N=2**int(sys.argv[2])*50,
    #         cdf = True if sys.argv[3] == "1" else False,
    #         symmetric = True if sys.argv[4] == "1" else False,
    #         folder=sys.argv[1][:sys.argv[1].find("_")])
    print("using {} seconds".format(time.time() - t1))