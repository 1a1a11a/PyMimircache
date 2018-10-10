# coding=utf-8


"""
this is following of 0901CloudPhysics.py and JYScore.py

"""


import os
import sys
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from PyMimircache import *
from PyMimircache.profiler.twoDPlots import *
from PyMimircache.A1a1a11a.myUtils.prepareRun import *
from PyMimircache.bin.conf import *


DATA = "/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon"
CACHE_SIZE = 80000
NUM_OF_THREADS = 24


PLOT_ON_EXIST = True


def plot_rd_distribution(dat, dat_type, interval, cdf=True, output_folder="1008_cphy_KL_rd_distribution_at_time", figname=None):
    if figname is None:
        figname = "{}/{}{}/rd_distr_{}_at_time_{}_ind.png".format( output_folder, interval,
                                                                   "_cdf" if cdf else "",
                                                                   os.path.basename(dat), interval )

    if not os.path.exists(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    reader = get_reader(dat, dat_type)
    CLRUProfiler(reader)._del_reuse_dist_file()
    rd_list = CLRUProfiler(reader).use_precomputedRD()
    # rd_list = LRUProfiler(reader).get_reuse_distance()

    if not PLOT_ON_EXIST and os.path.exists(figname.replace("ind", str(len(rd_list)//interval))):
        return


    max_rd = max(rd_list)
    print("max rd {}".format(max_rd))

    rd_distr_dic = defaultdict(int)
    for n, rd in enumerate(rd_list):
        rd_distr_dic[rd] += 1
        if n % interval == 0 and n != 0:
            rd_distr_list = [0] * (max_rd + 1)
            for rd, rd_count in rd_distr_dic.items():
                rd_distr_list[rd] = rd_count

            for i in range(len(rd_distr_list)):
                rd_distr_list[i] = rd_distr_list[i]/interval

            # assert abs(sum(rd_distr_list) - 1) < 0.01, "sum {}".format(sum(rd_distr_list))

            if cdf:
                for i in range(1, len(rd_distr_list)):
                    rd_distr_list[i] = rd_distr_list[i-1] + rd_distr_list[i]

            draw2d(rd_distr_list, figname=figname.replace("ind", str(n//interval)),
                   xlabel="reuse distance", ylabel="count percentage",
                   # xlimit=(-20, len(rd_distr_list) + 20), ylimit=(0, 1),
                   print_info=False)
            INFO("{}/{}".format(n//interval, len(rd_list)//interval), end="\n")

            rd_distr_dic.clear()


def _get_percentile_replace_one_entry(rd_deq, old, new, percentiles, old_percentile_result_pos):
    """
    pop the left-most element of rd_deq and add the new one, calculate the new percentile
    :param rd_deq: sorted rd
    :param old:
    :param new:
    :param percentiles:
    :return:
    """
    percentile_results = []
    for i in range(len(percentiles)):
        percentile_results.append(rd_deq[old_percentile_result_pos[i]])

    old = rd_deq.popleft()
    rd_deq.append(new)

    for i in range(len(percentiles)):
        percentile = percentiles[i]


MAX_RD = 10000000000


def plot_RD_percentiles(dat, dat_type, window, percentiles=(0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99), figname=None):

    if figname is None:
        figname = "1011_rdPercentile/rd_percentile_time_{}_{}.png".format(os.path.basename(dat), window )

    if os.path.dirname(figname) and not os.path.exists(os.path.dirname(figname)):
        os.makedirs(os.path.dirname(figname))

    if not PLOT_ON_EXIST and os.path.exists(figname):
        return

    reader = get_reader(dat, dat_type)
    rd_list = list(LRUProfiler(reader).use_precomputedRD())
    rd_deq = deque()

    percentile_results = defaultdict(list)
    max_rd = max(rd_list) * 2

    for i in range(0, len(rd_list)//window):
        rd_sub_list = rd_list[i*window:(i+1)*window]
        for i in range(len(rd_sub_list)):
            if rd_sub_list[i] == -1:
                rd_sub_list[i] = max_rd
        rd_sub_list.sort()
        for p in percentiles:
            rd = rd_sub_list[int(window * p)]
            if rd == max_rd:
                rd = - int(max_rd * 0.1)
            percentile_results[p].append(rd)

    for p, l in percentile_results.items():
        plt.plot(l, label=p)
        plt.legend(loc="best")
        plt.xlabel("time")
        plt.ylabel("reuse distance")
        plt.title("reuse distance percentile")
        plt.tight_layout()
        plt.savefig(figname.replace(".png", "_"+str(p)+".png"))
        plt.clf()

def plot_required_cache_size(dat, dat_type, window, percentiles=(0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.99),
                             allowed_violation_percentage=0,
                             figname=None, folder="1018_CPHY_SLACacheSize/"):

    if figname is None:
        figname = "{}/{}_{}_percent{}.png".format(folder, os.path.basename(dat), window,
                                          "_vio{:.2f}".format(allowed_violation_percentage))


    if prepare(figname, folder=folder, plot_on_exist=PLOT_ON_EXIST):
        return

    reader = get_reader(dat, dat_type)
    rd_list = list(LRUProfiler(reader).use_precomputedRD())
    rd_deq = deque()

    uniq_item_window = []       # the number of uniq items per window
    item_set = set()
    for n, r in enumerate(reader):
        item_set.add(r)
        if n != 0 and n % window == 0:
            uniq_item_window.append(len(item_set))
            item_set.clear()


    percentile_results = defaultdict(list)
    max_rd = max(rd_list) * 2

    for i in range(0, len(rd_list) // window):
        rd_sub_list = rd_list[i * window:(i + 1) * window]
        for i in range(len(rd_sub_list)):
            if rd_sub_list[i] == -1:
                rd_sub_list[i] = max_rd
        rd_sub_list.sort()
        for p in percentiles:
            rd = rd_sub_list[int(window * p)]
            # if rd == max_rd:
            #     rd = - int(max_rd * 0.1)
            percentile_results[p].append(rd)

    for l in percentile_results.values():
        for i in range(len(l)):
            if l[i] == max_rd:
                l[i] = -1
        cold_miss_value = -0.2 * (max(l))
        for i in range(len(l)):
            if l[i] == -1:
                l[i] = cold_miss_value


    for p, l in percentile_results.items():
        # now trace backwards to beginning of the trace to reflect cache size
        cold_miss_value = min(l)
        violation_value = cold_miss_value // 2
        allowed_violations = int(len(l) * allowed_violation_percentage)
        l_sort = sorted(l, reverse=True)
        l_cutoff = l_sort[allowed_violations]
        print("violation value {}".format(violation_value))

        for i in range(len(l)-1, 0, -1):
            if l[i] < 0:
                continue
            if l[i] > l_cutoff:
                l[i] = violation_value
                continue

            go_back_N_uniq = l[i]
            j = i
            while go_back_N_uniq > 0:
                if l[j] > 0 and l[j] < l[i]:
                    l[j] = l[i]
                go_back_N_uniq -= uniq_item_window[j]
                j -= 1


        plt.plot(l, label=p)
        plt.legend(loc="best")
        plt.xlabel("time")
        plt.ylabel("cache_size")
        # plt.title("reuse distance percentile")
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / len(l))))
        plt.tight_layout()
        plt.savefig(figname.replace("percent", "_hr{:.2f}".format(p)))
        plt.clf()


################### RUNNABLE #####################
def cphy_run1():
    for dat in ["78", "92", "106"]:
        for interval in [2560, 25600, 51200, 102400]:         # 2560,
            for vio_percentage in [0, 0.01, 0.02, 0.05, 0.1, 0.2]:
            # plot_rd_distribution(dat, "cphy", interval)
            # plot_RD_percentiles(dat, "cphy", interval)
                plot_required_cache_size(dat, "cphy", interval, allowed_violation_percentage=vio_percentage)









if __name__ == "__main__":
    # plot_rd_distribution("small", "cphy", 51200)
    cphy_run1()
    # plot_RD_percentiles("small", "cphy", 12800, percentiles=(0.2, 0.5, 0.8))