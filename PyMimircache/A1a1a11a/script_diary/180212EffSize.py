# coding=utf-8
"""

    This module plots the OPT/LRU effective cache size heatmap

"""

import os, sys, time, glob, socket, pickle, multiprocessing
from PyMimircache import *
from PyMimircache.A1a1a11a.myUtils.prepareRun import *
from PyMimircache.bin.conf import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed






############################# CONST #############################
CACHE_SIZE = 2000000


############################ METRIC ##############################



############################ FUNC ################################
def hm_opt_eff_size(dat, dat_type, time_mode="v", time_interval=20000,
                    cache_size=200000, bin_size=2000, alg="Optimal", use_percent=True):

    figname = "0218hm_{}EffSize/hm_{}EffSize_{}_{}{}.png".format(alg, alg, dat, time_mode, time_interval)

    if os.path.exists(figname):
        print("skip {}".format(dat))
        return
    reader = get_reader(dat, dat_type)
    ch = CHeatmap()
    ch.heatmap(reader, time_mode, "effective_size", time_interval=time_interval, algorithm=alg,
               cache_size=cache_size, bin_size=bin_size, use_percent=use_percent, num_of_threads=48,
               figname=figname)
    reader.close()


############################ RUNNABLE ################################
def run_cphy_sequentially():
    for i in range(106, 0, -1):
        hm_opt_eff_size("w{}".format(i), "cphy", "v", 20000)
        hm_opt_eff_size("w{}".format(i), "cphy", "r", 20 * 60 * 1000000)

def run_cphy_parallel():
    change_args_list = []
    for i in range(106, 0, -1):
        change_args_list.append(("w{}".format(i), "cphy", "v", 20000))
        change_args_list.append(("w{}".format(i), "cphy", "r", 20 * 60 * 1000000))
    run_parallel(hm_opt_eff_size, fixed_args=[], change_args_list=change_args_list, max_workers=os.cpu_count()//12)



def run_akamai(dat="/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1",
                        time_mode="r", time_interval=800,
                        cache_size=200000, bin_size=2000, alg="Optimal", use_percent=True):
    figname = "0218hm_{}EffSize/hm_{}EffSize_akamai{}_{}{}.png".format(alg, alg, dat.split("/")[-1], time_mode, time_interval)

    if os.path.exists(figname):
        print("skip {}".format(dat))
        return
    reader = CsvReader(dat, init_params=AKAMAI_CSV3)
    ch = CHeatmap()
    ch.heatmap(reader, time_mode, "effective_size", time_interval=time_interval, algorithm=alg,
               cache_size=cache_size, bin_size=bin_size, use_percent=use_percent, num_of_threads=os.cpu_count(),
               figname=figname)



def run_akamai_big_file():
    func("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")


def run_parallel(func, fixed_args, change_args_list, max_workers=os.cpu_count()):
    futures_dict = {}
    results_dict = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ppe:
        for arg in change_args_list:
            futures_dict[ppe.submit(func, *fixed_args, *arg)] = arg
        for futures in as_completed(futures_dict):
            results_dict[futures_dict[futures]] = futures.result()

    return futures_dict

import random


def mytest1(alg="LRU", time_mode="v", time_interval=200, **kwargs):
    c = Cachecow()
    reader = VscsiReader("../../data/trace.vscsi")

    # with open("testDat", "w") as ofile:
    #     for i in range(20):
    #         ofile.write("{}\n".format(random.randint(1, 2)))
    # reader = PlainReader("testDat")

    ch = CHeatmap()
    # ch.heatmap(reader, time_mode, "effective_size", time_interval=time_interval,
    #            cache_size=1, bin_size=1,
    #            algorithm=alg,
    #            use_percent=True,
    #            figname="hm_effSize_small_T_{}{}_{}_{}.png".format(alg, time_mode, time_interval, 2000))


    # c.vscsi("../../data/trace.vscsi")
    # c.plotHRCs(["LRU", "Optimal", "LFU"], cache_size=50000, bin_size=20, auto_resize=False)
    # ph = PyHeatmap()
    # ph.heatmap(c.reader, "v", "KL_st_et", time_interval=2000)

    # ch = CHeatmap()
    ch.heatmap(reader, time_mode, "effective_size", time_interval=time_interval,
               cache_size=24000, bin_size=200,
               algorithm=alg,
               use_percent=True,
               figname="hm_effSize_small_T_{}{}_{}_{}.png".format(alg, time_mode, time_interval, 2000))

    # ch.heatmap(reader, "v", "effective_size", time_interval=2000,
    #            cache_size=2400, bin_size=2000,
    #            algorithm="Optimal",
    #            use_percent=False,
    #            figname="hm_optEffSize_small_T_v_{}_{}.png".format(alg, 2000))


    # for ti in [20, 50, 200, 1000, 8000]:
    #     ch.heatmap(reader, "v", "opt_effective_size", time_interval=ti,
    #                cache_size=24000, bin_size=200, use_percent=True,
    #                figname="hm_optEffSize_small_T_v_{}.png".format(ti))


if __name__ == "__main__":
    # mytest1()
    run_akamai()
    # run_cphy_parallel()
    # run_cphy_sequentially()
