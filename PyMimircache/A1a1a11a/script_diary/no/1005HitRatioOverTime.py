# coding=utf-8
""" this module calculates and plots interval hit ratio curve
"""


import os
import sys
import time
import math
from collections import defaultdict
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from PyMimircache import *
from PyMimircache.bin.conf import *
from PyMimircache.cacheReader.traceStat import TraceStat
from PyMimircache.utils.timer import MyTimer
from PyMimircache.profiler.twoDPlots import *
from concurrent.futures import ProcessPoolExecutor, as_completed


PLOT_ON_EXIST = True

def use_precomputeRD(p):
    """
    this function loads precomputed reuse distance into
    profiler to avoid O(NlogN) reuse distance computation,
    if the precomputed reuse distance does not exist,
    it will compute reuse distance and save to the specified location
    NOTE: this function has been integrated into PyMimircache when INTERNAL_USAGE is True
    :param p: an instance of LRUProfiler
    :return:
    """
    rd_dat_path = p.reader.file_loc.replace("/home/jason/ALL_DATA/",
                                            "/research/jason/preComputedData/RD/")
    if os.path.exists(rd_dat_path):
        p.load_reuse_dist(rd_dat_path, "rd")
    else:
        if not os.path.exists(os.path.dirname(rd_dat_path)):
            os.makedirs(os.path.dirname(rd_dat_path))
        print("compute reuse distance for {}".format(os.path.basename(p.reader.file_loc)))
        p.save_reuse_dist(rd_dat_path, "rd")
    return p.get_reuse_distance()


def hit_ratio_over_time(dat, dat_type, cache_size,
                        decay_coef=0.2, mode="v", interval=10000, figname=None):
    """
    this function plots hit ratio over time,
    y-axis is hit ratio and x-axis is time, the hit ratio is calculated
    with exponential decay
    :param dat: the path to data
    :param dat_type: the type of data, which is used for get_reader to
                        use correct reader type and initialization parameters
    :param cache_size: the size of cache
    :param decay_coef: exponential decay coefficient on the history hit ratio
    :param mode:       time mode, can be r(real time) or v(virtual time)
    :param interval:   time interval
    :param figname:    output name
    :return: a list of hit ratio
    """
    if figname is None:
        figname = "ihr_{}_{}_{}.png".format(os.path.basename(dat), cache_size, decay_coef)

    if not PLOT_ON_EXIST and os.path.exists(figname):
        return

    reader = CsvReader(dat, init_params=AKAMAI_CSV3)
    reader = CsvReader(dat, init_params={"header":False, "delimiter": "\t", "label":6, 'real_time':1})
    reader = get_reader(dat, dat_type=dat_type)
    p = LRUProfiler(reader)
    rd_list = p.use_precomputedRD()
    # rd_list = p.get_reuse_distance()

    hit_ratio_list = []
    hit_ratio_overall = 0
    hit_count_current_interval = 0

    if mode == "v":
        for n, rd in enumerate(rd_list):
            if rd > cache_size or rd == -1:
                pass
            else:
                hit_count_current_interval += 1
            if n % interval == 0:
                hit_ratio_current_interval = hit_count_current_interval / interval
                hit_ratio_overall = hit_ratio_overall * decay_coef + hit_ratio_current_interval * (1 - decay_coef)
                hit_count_current_interval = 0
                hit_ratio_list.append(hit_ratio_overall)

    elif mode == "r":
        ind = 0
        req_count_current_interval = 0
        line = reader.read_time_req()
        t, req = line
        last_time_interval_cutoff = t

        while line:
            last_time = t
            t, req = line
            if t - last_time_interval_cutoff > interval:
                hit_ratio_current_interval = hit_count_current_interval / req_count_current_interval
                hit_ratio_overall = hit_ratio_overall * decay_coef + hit_ratio_current_interval * (1 - decay_coef)
                hit_count_current_interval = 0
                last_time_interval_cutoff = last_time
                hit_ratio_list.append(hit_ratio_overall)

            rd = rd_list[ind]
            req_count_current_interval += 1
            if rd != -1 and rd <= cache_size:
                hit_count_current_interval += 1

            line = reader.read_time_req()
            ind += 1

    draw2d(hit_ratio_list, xlabel="{} time (interval {})".format({"r": "real", "v": "virtual"}.get(mode, ""), interval),
           ylabel="hit ratio (decay {})".format(decay_coef),
           logX=False, logY=False, figname=figname,
           xticks=ticker.FuncFormatter(lambda x, _: '{:.0%}'.format((x ) / len(hit_ratio_list))))
    return hit_ratio_list


def IHRC_heatmap(dat, dat_type, cache_size, decay_coef=0.2,
                 time_mode="v", time_interval=10000, reader=None, figname=None):
    """
    this function plots the heatmap version of IHRC
    :param dat:
    :param dat_type:
    :param cache_size:
    :param decay_coef:
    :param time_mode:
    :param time_interval:
    :param reader:
    :param figname:
    :return:
    """
    if figname is None:
        figname = "heatmap_IHRC_{}_{}_{}.png".format(os.path.basename(dat), cache_size, decay_coef)

    if not PLOT_ON_EXIST and os.path.exists(figname):
        return

    if reader is None:
        reader = get_reader(dat, dat_type=dat_type)
        p = LRUProfiler(reader)
        p.use_precomputedRD()
    ch = CHeatmap()
    print(reader.cReader)
    ch.heatmap(reader, time_mode, "hit_ratio_start_time_end_time",
               algorithm="LRU", cache_params=None,
               time_interval=time_interval, cache_size=cache_size,
               # interval_hit_ratio=True, decay_coefficient=decay_coef,
               num_of_threads=os.cpu_count(),
               figname=figname)
    plt.clf()







################################# RUNNABLE ################################
def akamai3_run(output_folder="1008Akamai3_IHRC_heatmap", parallel=True):
    TIME_MODE = "r"
    TIME_INTERVAL = 10

    trace_dir = "/home/jason/ALL_DATA/akamai3/layer/1/clean0922/"
    if not os.path.exists("{}/oneDay".format(output_folder)):
        os.makedirs("{}/oneDay".format(output_folder))
        os.makedirs("{}/twoDay".format(output_folder))

    if parallel:
        # parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=24) as ppe:
            for length in ["oneDay", "twoDay"]:
                for f in os.listdir("{}/{}".format(trace_dir, length)):
                    for size in [2000, 20000, 80000, 200000, 1000000]:
                        for decay_coef in [0.2, 0.5, 0.8]:
                            dat = "{}/{}/{}".format(trace_dir, length, f)
                            # 2d plot
                            # futures[ppe.submit(hit_ratio_over_time, dat, dat_type="akamai3", cache_size=size, decay_coef=decay_coef,
                            #                    mode=TIME_MODE, interval=TIME_INTERVAL,
                            #                     figname="{}/{}/IHRC_{}_{}_{}.png".format(output_folder, length, os.path.basename(dat), size, decay_coef))] = None
                            # heatmap
                            futures[ppe.submit(IHRC_heatmap, dat, dat_type="akamai3", cache_size=size, decay_coef=decay_coef,
                                               time_mode=TIME_MODE, time_interval=TIME_INTERVAL,
                                             figname="{}/{}/IHRC_heatmap_{}_{}_{}.png".format(output_folder, length, os.path.basename(dat), size, decay_coef))] = None


            for f in ["oneDayData.sort", "twoDayData.sort"]:
                for size in [2000, 20000, 80000, 200000, 1000000]:
                    for decay_coef in [0.2, 0.5, 0.8]:
                        dat = "{}/{}".format(trace_dir, f)
                        # futures[ppe.submit(hit_ratio_over_time, dat, dat_type="akamai3", cache_size=size, decay_coef=decay_coef,
                        #                    mode=TIME_MODE, interval=TIME_INTERVAL,
                        #                     figname="{}/{}/IHRC_{}_{}_{}.png".format(output_folder, length, os.path.basename(dat), size, decay_coef))] = None
                        # heatmap
                        futures[ppe.submit(IHRC_heatmap, dat, dat_type="akamai3", cache_size=size, decay_coef=decay_coef,
                                           time_mode=TIME_MODE, time_interval=TIME_INTERVAL,
                                           figname="{}/IHRC_heatmap_{}_{}_{}.png".format(output_folder,
                                                                                         os.path.basename(dat), size,
                                                                                         decay_coef))] = None


            finished_count = 0
            for _ in as_completed(futures):
                finished_count += 1
                print("{}/{}".format(finished_count, len(futures)))


    else:
        # sequential
        # for length in ["oneDay", "twoDay"]:
        #     for f in os.listdir("{}/{}".format(trace_dir, length)):
        #         for size in [2000, 20000, 80000, 200000, 1000000]:
        #             for decay_coef in [0.2, 0.5, 0.8]:
        #                 dat = "{}/{}/{}".format(trace_dir, length, f)
        #                 hit_ratio_over_time(dat, "akamai3", size, decay_coef, TIME_MODE, TIME_INTERVAL,
        #                                     figname="{}/{}/IHRC_{}_{}_{}.png".format(output_folder, length, os.path.basename(dat), size, decay_coef))
        #                 IHRC_heatmap(dat, "akamai3", size, decay_coef, TIME_MODE, TIME_INTERVAL,
        #                                     figname="{}/{}/IHRC_{}_{}_{}.png".format(output_folder, length, os.path.basename(dat), size, decay_coef))


        for f in ["oneDayData.sort", "twoDayData.sort"]:
            for size in [2000, 20000, 80000, 200000, 1000000]:
                for decay_coef in [0.2, 0.5, 0.8]:
                    dat = "{}/{}".format(trace_dir, f)
                    reader = get_reader(dat, dat_type="akamai3")
                    p = LRUProfiler(reader)
                    p.use_precomputedRD()

                    # hit_ratio_over_time(dat, "akamai3", size, decay_coef, TIME_MODE, TIME_INTERVAL,
                    #             figname="{}/IHRC_{}_{}_{}.png".format(output_folder, os.path.basename(dat), size, decay_coef))

                    IHRC_heatmap(dat, "akamai3", size, decay_coef, TIME_MODE, TIME_INTERVAL, reader=reader,
                                figname="{}/IHRC_heatmap_{}_{}_{}.png".format(output_folder, os.path.basename(dat), size, decay_coef))


def cphy_run(output_folder="1008CPHY_IHRC_Heatmap"):
    TIME_MODE = "r"
    TIME_INTERVAL = 20 * 60 * 1000000

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    DAT = {"78": [80000, 640000],
            "92": [20000, 640000],
            "106": [20000, 120000]}

    futures = {}
    with ProcessPoolExecutor(max_workers=24) as ppe:
        for dat, sizes in DAT.items():
            for size in sizes:
                for decay_coef in [0.2, 0.5, 0.8]:
                    # 2d plot
                    # futures[ppe.submit(hit_ratio_over_time, dat, dat_type="cphy", cache_size=size, decay_coef=decay_coef,
                    #                    mode=TIME_MODE, interval=TIME_INTERVAL,
                    #                  figname="{}/hr_time_{}_{}_{}.png".format(output_folder, os.path.basename(dat), size, decay_coef))] = None
                    # heatmap
                    futures[ppe.submit(IHRC_heatmap, dat, dat_type="cphy", cache_size=size, decay_coef=decay_coef,
                                       time_mode=TIME_MODE, time_interval=TIME_INTERVAL,
                                     figname="{}/IHRC_heatmap_{}_{}_{}.png".format(output_folder, os.path.basename(dat), size, decay_coef))] = None
        finished_count = 0
        for _ in as_completed(futures):
            finished_count += 1
            print("{}/{}".format(finished_count, len(futures)))

    # sequential 2d plot
    # for dat, sizes in DAT.items():
    #     for size in sizes:
    #         for decay_coef in [0.2, 0.5, 0.8]:
    #             hit_ratio_over_time(dat, dat_type="cphy", cache_size=size, decay_coef=decay_coef, mode=TIME_MODE, interval=TIME_INTERVAL,
    #                                  figname="{}/hr_time_{}_{}_{}.png".format(output_folder, os.path.basename(dat), size, decay_coef))


def my_test():
    # for i in range(1, 10):
    #     IHRC_heatmap("small", dat_type="cphy", cache_size=12000, decay_coef=i/10, time_mode="v", time_interval=200)

    IHRC_heatmap("small", dat_type="cphy", cache_size=3000, decay_coef=0.2, time_mode="v", time_interval=2000)
    IHRC_heatmap("small", dat_type="cphy", cache_size=3000, decay_coef=0.4, time_mode="v", time_interval=200)




if __name__ == "__main__":
    mt = MyTimer()
    # for i in range(0, 11):
    #     hit_ratio_over_time("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/testData", 2000, decay_coef=i/10, mode="v", interval=10000)
    # hit_ratio_over_time("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/testData", 2000, decay_coef=0.2, mode="v", interval=10000)
    # hit_ratio_over_time("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", 20000)
    # hit_ratio_over_time("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/twoDayData.sort", 20000)

    # hit_ratio_over_time("", dat_type="cphy", cache_size=size, decay_coef=decay_coef, mode=TIME_MODE,
    #                     interval=TIME_INTERVAL,
    #                     figname="1006CPHY_IHRC/hr_time_{}_{}_{}.png".format(os.path.basename(dat), size, decay_coef))

    # my_test()
    # cphy_run()
    # akamai3_run(parallel=False)
    my_test()
    mt.tick()