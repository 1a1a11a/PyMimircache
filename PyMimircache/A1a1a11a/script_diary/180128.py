# coding=utf-8
"""


"""

import os, sys, time, glob, socket, pickle, multiprocessing
from PyMimircache import *
from PyMimircache.cacheReader.multiReader import multiReader
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
from PyMimircache.bin.conf import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


TRACE_DIR, NUM_OF_THREADS = initConf(trace_type="Akamai", trace_format="csv")






############################# CONST #############################
CACHE_SIZE = 2000


############################ METRIC ##############################



############################ FUNC ################################
def plot_rd(dat, dat_type):
    reader = get_reader(dat, dat_type)
    



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






if __name__ == "__main__":
    run_akamai_seq()
    run_akamai_parallel()
    run_akamai_big_file()


