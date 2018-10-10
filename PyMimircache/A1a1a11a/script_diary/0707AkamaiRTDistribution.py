# coding=utf-8
"""
this module plots the reuse-time distribution for all the requests in the trace

"""

import os, sys, time, glob, socket, pickle, multiprocessing
from collections import defaultdict
import math
from PyMimircache import *
from PyMimircache.cacheReader.multiReader import MultiReader
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
from PyMimircache.bin.conf import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

TRACE_DIR, NUM_OF_THREADS = initConf(trace_type="Akamai", trace_format="csv")
TRACE_DIR += "/day/"
# TRACE_DIR = "/home/jason/ALL_DATA/akamai_new_logs"


############################# CONST #############################



############################ FUNC ################################
def plot_rt_distribution(dat, o_dir = "0707RT_Distribution", log_RT=True, log_count=True, cdf=False):
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    if cdf:
        figname = "{}/{}_cdf2.png".format(o_dir, os.path.basename(dat))
    else:
        figname = "{}/{}.png".format(o_dir, os.path.basename(dat))
    if os.path.exists(figname):
        return


    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    d_last_time = {}
    d_reuse_time = defaultdict(int)
    max_reuse_time = 0

    tr = reader.read_time_req()
    while tr:
        t, r = tr
        if r in d_last_time:
            d_reuse_time[int(t - d_last_time[r])] += 1
            if max_reuse_time < int(t - d_last_time[r]):
                max_reuse_time = int(t - d_last_time[r])

                # if log_RT:
            #     d_reuse_time[ int(math.log2( t - d_last_time[r] )) ] += 1
            #     if max_reuse_time < int(math.log2( t - d_last_time[r] )):
            #         max_reuse_time = int(math.log2( t - d_last_time[r] ))
            # else:
            #     print("non-log not supported")
        d_last_time[r] = t
        tr = reader.read_time_req()

    l = [0] * (max_reuse_time + 1)
    for reuse_time, count in d_reuse_time.items():
        l[reuse_time] = count

    l_sum = sum(l)
    if cdf:
        l[0] = l[0]/l_sum
        # for i in range(1, len(l)):
        #     l[i] = l[i-1] + l[i]/l_sum
    else:
        for i in range(len(l)):
            l[i] = l[i] / l_sum



    plt.xlabel("reuse time(s)")
    plt.ylabel("num of requests")
    if log_RT:
        if log_count:
            plt.loglog(l)
        plt.semilogx(l)
    else:
        if log_count:
            plt.semilogy(l)
        else:
            plt.plot(l)

    plt.savefig(figname, dpi=600)
    plt.clf()



    # plt.plot(l)
    # if not cdf:
    #     plt.xlabel("freq")
    #     plt.ylabel("num of obj")
    #     plt.loglog(l)
    #     plt.savefig("{}/{}.png".format(o_dir, os.path.basename(dat)), dpi=600)
    #     plt.clf()
    # else:
    #     plt.xlabel("cumulative ratio of objs")
    #     plt.ylabel("cumulative ratio of requests")
    #     plt.semilogy(l)
    #     plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format((x+1)/len(l))))
    #     plt.savefig("{}/{}_cdf.png".format(o_dir, os.path.basename(dat)), dpi=600)
    #     plt.clf()


############################ RUNNABLE ################################
def run_akamai_seq(dat_folder):
    dat_list = [f for f in glob.glob("{}/*.sort".format(dat_folder)) if 'key' not in f]
    print("{} dat".format(len(dat_list)))
    for f in dat_list:
        print(f)
        plot_rt_distribution(dat="{}".format(f), cdf=True)


def run_akamai_parallel(dat_folder, threads=12):
    dat_list = [f for f in glob.glob(dat_folder)]
    print("{} dat".format(len(dat_list)))

    with multiprocessing.Pool(threads) as p:
        p.map(func, dat_list)


def run_akamai_big_file():
    plot_rt_distribution("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")


if __name__ == "__main__":
    run_akamai_seq(dat_folder=TRACE_DIR)
    # run_akamai_parallel()
    run_akamai_big_file()


