# coding=utf-8
"""


"""

import os, sys, math, time, glob, socket, pickle, multiprocessing
import json
from PyMimircache import *
from PyMimircache.A1a1a11a.myUtils.prepareRun import *
from PyMimircache.bin.conf import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyMimircache.profiler.profilerUtils import draw_heatmap
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.ticker as ticker
from collections import defaultdict
from copy import copy



############################# CONST #############################
CACHE_SIZE = 2000000


############################ METRIC ##############################



############################ FUNC ################################
def plot_miss_freq_distribution_hm(cache_alg="Optimal", vtime_interval=20000, cache_size=8000, log_base=1.6):
    """
    this function plots a heatmap, y-axis is the frequency (in log scale), x-axis is (virtual) time,
    the color is the number of misses in the time_interval

    :param vtime_interval:
    :param cache_size:
    :param log_base:
    :return:
    """

    reader = VscsiReader("../../data/trace.vscsi")
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    p = CGeneralProfiler(reader, cache_size=cache_size, bin_size=cache_size, cache_alg=cache_alg)
    hit_result = p.get_hit_result()
    req_freq = []

    freq_dict = defaultdict(int)
    max_freq = 0
    for req in reader:
        freq_dict[req] += 1
        req_freq.append(freq_dict[req])
        if freq_dict[req] > max_freq:
            max_freq = freq_dict[req]
    reader.reset()


    # xydict = [[0] * int(math.ceil(math.log(max_freq, log_base)))] * (reader.num_of_req()//vtime_interval + 1)
    xydict = []

    current_distribution_interval = [0] * (int(math.ceil(math.log(max_freq, log_base)) + 1))

    for i in range(reader.get_num_of_req()):
        if i and i % vtime_interval == 0:
            for i in range(len(current_distribution_interval)):
                if sum(current_distribution_interval) != 0:
                    current_distribution_interval[i] = current_distribution_interval[i] / sum(current_distribution_interval)

            xydict.append(current_distribution_interval)
            current_distribution_interval = [0] * (int(math.ceil(math.log(max_freq, log_base)) + 1))

        if not hit_result[1][i]:
            freq = int(math.ceil(math.log(req_freq[i], log_base)))
            current_distribution_interval[freq] += 1

    xydict = np.array(xydict, dtype=np.double)
    plot_data = xydict.T
    np.ma.masked_where(plot_data == 0, plot_data, copy=False)
    print(xydict)


    plot_kwargs = {
        "xlabel": "virtual time",
        "ylabel": "frequency",
        "xticks": ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (plot_data.shape[1] - 1))),
        "yticks": ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x)),
        "figname": "miss_freq_{}_percent_hm.png".format(cache_alg)
    }


    draw_heatmap(plot_data, **plot_kwargs)
    reader.reset()
    return plot_data


def plot_miss_freq_diff_distr_hm(dat="/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1",
                                 cache_alg1="LRU", cache_alg2="Optimal",
                                 vtime_interval=20000, cache_size=8000, log_base=1.6,
                                 folder="0415"):
    """
    this function plots a heatmap, y-axis is the frequency (in log scale), x-axis is (virtual) time,
    the color is the number of misses in the time_interval

    :param vtime_interval:
    :param cache_size:
    :param log_base:
    :return:
    """

    # reader = VscsiReader("../../data/trace.vscsi")
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    # reader = VscsiReader(dat)
    dat_name = dat[dat.rfind("/")+1:dat.rfind(".")]
    p1 = CGeneralProfiler(reader, cache_size=cache_size, bin_size=cache_size, cache_alg=cache_alg1)
    p2 = CGeneralProfiler(reader, cache_size=cache_size, bin_size=cache_size, cache_alg=cache_alg2)

    hit_result = [p1.get_hit_result(), p2.get_hit_result()]
    req_freq = []

    freq_dict = defaultdict(int)
    max_freq = 0
    for req in reader:
        freq_dict[req] += 1
        req_freq.append(freq_dict[req])
        if freq_dict[req] > max_freq:
            max_freq = freq_dict[req]
    reader.reset()

    xydict = [[], []]


    for i0 in range(2):
        current_distribution_interval = [0] * (int(math.ceil(math.log(max_freq, log_base)) + 1))
        for i in range(reader.get_num_of_req()):
            if i and i % vtime_interval == 0:
                xydict[i0].append(current_distribution_interval)
                current_distribution_interval = [0] * (int(math.ceil(math.log(max_freq, log_base)) + 1))

            if not hit_result[i0][1][i]:
                freq = int(math.ceil(math.log(req_freq[i], log_base)))
                current_distribution_interval[freq] += 1
    # with open("xydic1.json", "w") as ofile:
    #     json.dump(xydict[0], ofile)
    # with open("xydic2.json", "w") as ofile:
    #     json.dump(xydict[1], ofile)

    xydict = [np.array(xydict[0], dtype=np.int), np.array(xydict[1], dtype=np.int)]
    plot_data = (xydict[0] - xydict[1]).T

    # np.savetxt("xydict1", xydict[0])
    # np.savetxt("xydict2", xydict[1])


    np.ma.masked_where(plot_data == 0, plot_data, copy=False)

    plot_kwargs = {
        "xlabel": "virtual time",
        "ylabel": "frequency",
        "xticks": ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (plot_data.shape[1] - 1))),
        "yticks": ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x)),
        "figname": "{}/miss_freq_{}_vti{}_size{}_{}_{}_hm.png".format(folder, dat_name, vtime_interval, cache_size, cache_alg1, cache_alg2)
    }


    draw_heatmap(plot_data, **plot_kwargs)
    reader.reset()
    return plot_data


############################ RUNNABLE HELPER ################################
def run_parallel(func, fixed_kwargs, change_kwargs_list, max_workers=os.cpu_count()):
    futures_dict = {}
    results_dict = {}
    print("{} threads".format(max_workers))

    with ProcessPoolExecutor(max_workers=max_workers) as ppe:
        for kwargs in change_kwargs_list:
            futures_dict[ppe.submit(func, **fixed_kwargs, **kwargs)] = kwargs.values()
        for futures in as_completed(futures_dict):
            results_dict[futures_dict[futures]] = futures.result()

    return results_dict


############################ RUNNABLE ################################
def run_akamai_seq(func, *args, **kwargs):
    dat_folder = "/home/jason/ALL_DATA/akamai3/layer/1/"
    dat_list = [f for f in glob.glob("{}/*.1".format(dat_folder))]
    print("{} dat".format(len(dat_list)))
    for f in dat_list:
        print(f)
        func(*args, **kwargs, dat="{}".format(f))


def run_akamai_parallel(func, fixed_kwargs={}, change_kwargs_list=[], threads=os.cpu_count()):
    dat_folder = "/home/jason/ALL_DATA/akamai3/layer/1/"
    dat_list = [f for f in glob.glob("{}/*.1".format(dat_folder))]
    print("{} dat".format(len(dat_list)))
    new_change_kwargs_list = []
    for dat in dat_list:
        for change_kwargs in change_kwargs_list:
            change_kwargs["dat"] = dat
            new_change_kwargs_list.append(copy(change_kwargs))
    print("{} combinations".format(len(new_change_kwargs_list)))
    run_parallel(func, fixed_kwargs, new_change_kwargs_list, max_workers=threads)


def run_akamai_big_file(func, fixed_kwargs, change_kwargs_list):
    for kwargs in change_kwargs_list:
        func(**fixed_kwargs, **kwargs, dat="/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")


def run_cphy_seq(func, *args, **kwargs):
    dat_list = [f for f in glob.glob("{}/*.vscsitrace".format("/home/cloudphysics/traces/"))]
    print("{} dat".format(len(dat_list)))
    for f in dat_list:
        print(f)
        func(*args, **kwargs, dat="{}".format(f))


def run_cphy_parallel(func, fixed_kwargs={}, change_kwargs_list=[], threads=os.cpu_count()):
    dat_list = [f for f in glob.glob("{}/*.vscsitrace".format("/home/cloudphysics/traces/"))]
    print("{} dat".format(len(dat_list)))
    new_change_kwargs_list = []
    for dat in dat_list:
        for change_kwargs in change_kwargs_list:
            change_kwargs["dat"] = dat
            new_change_kwargs_list.append(copy(change_kwargs))
    print("{} combinations".format(len(new_change_kwargs_list)))
    run_parallel(func, fixed_kwargs, new_change_kwargs_list, max_workers=threads)



############################ TEST ################################
def mytest1():
    for d in ["w94", "w100"]:
        c = Cachecow()
        c.vscsi("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(d))
        c.plotHRCs(["LRU", "Optimal", "LFUFast"], figname="{}.png".format(d))


def mytest2():
    with open("xydict1") as ifile:
        xydict1 = json.load(ifile)
    with open("xydict2") as ifile:
        xydict2 = json.load(ifile)
    print("{} {}".format(len(xydict1), len(xydict1[0])))
    print("{} {}".format(len(xydict2), len(xydict2[0])))
    print(xydict1)
    print(xydict2)


if __name__ == "__main__":
    plot_miss_freq_distribution_hm(cache_alg="LRU", vtime_interval=20000, log_base=2)
    # plot_miss_freq_distribution_hm(cache_alg="Optimal")
    # plot_miss_freq_diff_distr_hm(vtime_interval=20000, log_base=2)

    sys.exit(1)

    # run_cphy_seq(plot_miss_freq_diff_distr_hm)

    change_kwargs_list = [{"cache_size": 8000},
                          {"cache_size": 20000},
                          {"cache_size": 80000},
                          {"cache_size": 200000}]

    # run_cphy_parallel(plot_miss_freq_diff_distr_hm, fixed_kwargs={"log_base": 1.2}, change_kwargs_list=change_kwargs_list)
    run_akamai_parallel(plot_miss_freq_diff_distr_hm,
                        fixed_kwargs={"log_base": 1.2, "vtime_interval":200000},
                        change_kwargs_list=change_kwargs_list)
