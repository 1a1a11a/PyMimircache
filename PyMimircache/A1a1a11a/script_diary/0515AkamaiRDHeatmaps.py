# coding=utf-8
"""

This module plots the rd distribution heatmaps

"""

import os, sys, glob, pickle
from PyMimircache import *
from PyMimircache.bin.conf import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


TRACE_DIR, NUM_OF_THREADS = initConf(trace_type="Akamai", trace_format="csv")
TRACE_DIR += "/day/"





def heatmap_cachesize(dat):
    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    ch = CHeatmap()
    # ch.heatmap(reader, 'r', )



def heatmap_rd_dist(dat):
    output_file = "{}/hm_rt_{}_r.png".format("0622_hm_rt", os.path.basename(dat))

    if os.path.exists(output_file):
        print("{} exists".format(os.path.basename(dat)))
        return

    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    # reader = binaryReader(dat, init_params=AKAMAI_BIN_MAPPED2)
    # LRUProfiler(reader).load_reuse_dist("/home/jason/ALL_DATA/Akamai/day/rd/{}.rd".format(os.path.basename(dat).split('.')[0]),
    #                                     rd_type="rd")
    # LRUProfiler(reader).load_reuse_dist("/home/jason/ALL_DATA/Akamai/day/rd/201610.rd", rd_type="rd")

    ch = CHeatmap()
    # real time
    # ch.heatmap(reader, 'r', "rd_distribution",
    #            time_interval=600, num_of_threads=40, filter_rd=20,
    #            figname=output_file)

    # virtual time
    # ch.heatmap(reader, 'v', "rd_distribution",
    #            time_interval=100000, num_of_threads=40, filter_rd=20,
    #            figname=output_file)

    # distance
    # ch.heatmap(reader, 'r', "dist_distribution",
    #            time_interval=600, num_of_threads=40, filter_rd=20,
    #            figname=output_file)

    # reuse time
    ch.heatmap(reader, 'r', "reuse_time_distribution",
               time_interval=600, num_of_threads=40, filter_rd=20,
               figname=output_file)








############################ RUNNABLE #############################
def run_rd(file_type="binary"):
    if file_type == "binary":
        files = glob.glob("/home/jason/ALL_DATA/Akamai/day/*.mapped")
    elif file_type == "csv":
        files = glob.glob("/home/jason/ALL_DATA/Akamai/day/*.sort")
    else:
        print("unknown filetype {}".format(file_type))
        return

    for f in files:
        if 'test' in f:
            continue
        print(f)
        print(" ")
        heatmap_rd_dist(f)

def run_rd2():
    """
    for second batch of Akamai trace
    :return:
    """
    TRACE_DIR = "/home/jason/ALL_DATA/akamai_new_logs"
    for f in os.listdir(TRACE_DIR):
        if 'anon' in f:
            print(f)
            print(" ")
            if not os.path.exists("0622_hm_rt/hm_rt_{}_r.png".format(os.path.basename(f))):
                heatmap_rd_dist("{}/{}".format(TRACE_DIR, f))




def run_rd_big_file():
    heatmap_rd_dist("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")


def test_rd():
    heatmap_rd_dist("/home/jason/ALL_DATA/Akamai/day/test0.csv")


if __name__ == "__main__":
    # print(TRACE_DIR)
    # cal_perf(60, "{}/{}".format(TRACE_DIR, "test.csv"))
    # plt_perf("{}/{}".format(TRACE_DIR, "test.mapped.bin.LL"))

    # heatmap_rd_dist("{}/{}".format(TRACE_DIR, "test.mapped.bin.LL"))

    # run_rd(file_type="csv")
    # test_rd()
    # run_rd2()
    run_rd_big_file()

