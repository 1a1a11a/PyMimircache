# coding=utf-8
"""


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




############################ RUNNABLE ################################
def run_akamai_seq(dat_folder):
    dat_list = [f for f in glob.glob("{}/*.sort".format(dat_folder))]
    print("{} dat".format(len(dat_list)))
    for f in dat_list:
        print(f)
        func(dat="{}".format(f))



def run_akamai_parallel(dat_folder, threads=12):
    dat_list = [f for f in glob.glob(dat_folder)]
    print("{} dat".format(len(dat_list)))

    with multiprocessing.Pool(threads) as p:
        p.map(func, dat_list)

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

############################ TEST ################################
def mytest1():
    for d in ["w94", "w100"]:
        c = Cachecow()
        c.vscsi("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(d))prit
        c.plotHRCs(["LRU", "Optimal", "LFUFast"], figname="{}.png".format(d))


if __name__ == "__main__":
    mytest1()

