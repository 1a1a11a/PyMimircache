# coding=utf-8

"""
    This function tries to characterize the trace from more aspects


"""

import os
import sys
import time
import math
import bisect
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pprint import pprint, pformat
from concurrent.futures import ProcessPoolExecutor, as_completed


from PyMimircache.profiler.cHeatmap import CHeatmap
from PyMimircache.utils.timer import MyTimer
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import *


def characterize_by_freq_interval(reader, time_interval, dat_name, time_mode="v", cutoff_freq=12):
    """
    an interval plot of future dist classified by freq, PLUS
    an interval plot of hit probability classified by freq

    """

    access_time = defaultdict(list)
    fdist_avg = defaultdict(list)
    fdist_median = defaultdict(list)
    hit_prob = defaultdict(list)


    for n, req in enumerate(reader):
        access_time[req].append(n)
    reader.reset()
    bp = CHeatmap.get_breakpoints(reader, time_mode, time_interval)

    for i in range(len(bp)-1):
        fdist_interval = defaultdict(list)
        hit_miss_interval = {}
        for vts in range(bp[i], bp[i+1], 1):
            rts, req = reader.read_time_req()
            # freq of current req before request
            pos = bisect.bisect_left(access_time[req], vts)
            freq = pos
            if freq > cutoff_freq:
                continue
            hit_miss_freq = hit_miss_interval.get(freq, [0, 0])
            if len(access_time[req]) > pos+1:
                fdist_interval[freq].append(access_time[req][pos+1] - access_time[req][pos])
                hit_miss_freq[0] += 1
            else:
                hit_miss_freq[1] += 1
            hit_miss_interval[freq] = hit_miss_freq

        for freq, hit_miss_count in hit_miss_interval.items():
            hit_prob[freq].append(hit_miss_count[0] / (sum(hit_miss_count)))
        for freq, dist_list in fdist_interval.items():
            fdist_avg[freq].append(sum(dist_list)/len(dist_list))
            # use median
            dist_list.sort()
            fdist_median[freq].append(dist_list[len(dist_list)//2])
    reader.reset()

    if not os.path.exists("1218ClassByFreq/{}_{}{}".format(dat_name, time_mode, time_interval)):
        os.makedirs("1218ClassByFreq/{}_{}{}".format(dat_name, time_mode, time_interval))

    # plot hit prob
    for freq, hit_prob_list in hit_prob.items():
        # plot for each freq
        plt.plot(hit_prob_list, label=str(freq))
        plt.xlabel("Time")
        plt.ylabel("Hit Probability")
        plt.legend(loc="best")
        plt.savefig("1218ClassByFreq/{}_{}{}/Hit_Prob_{}.png".format(dat_name, time_mode, time_interval, freq))
        plt.clf()

    # plot all freq on one
    for freq, hit_prob_list in hit_prob.items():
        plt.plot(hit_prob_list, label=str(freq))
    plt.xlabel("Time")
    plt.ylabel("Hit Probability")
    plt.legend(loc="best")
    plt.savefig("1218ClassByFreq/{}_Hit_Prob_{}{}.png".format(dat_name, time_mode, time_interval))
    plt.clf()

    # plot dist
    for name, fdist in {"avg": fdist_avg, "median": fdist_median}.items():
        for freq, fdist_list in fdist.items():
            plt.plot(fdist_list, label=str(freq))
            plt.xlabel("Time")
            plt.ylabel("Future Distance")
            plt.legend(loc="best")
            plt.savefig("1218ClassByFreq/{}_{}{}/fdist_{}_{}.png".format(dat_name, time_mode, time_interval, name, freq))
            plt.clf()

        for freq, fdist_list in fdist.items():
            plt.plot(fdist_list, label=str(freq))
        plt.xlabel("Time")
        plt.ylabel("Future Distance")
        plt.legend(loc="best")
        plt.savefig("1218ClassByFreq/{}_fdist_{}_{}{}.png".format(dat_name, name, time_mode, time_interval))
        plt.clf()


def dist_popularity_freq(reader, dat_name, freq={2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}):
    access_time = defaultdict(list)
    dist_list_dict = defaultdict(list)


    for n, req in enumerate(reader):
        access_time[req].append(n)
    reader.reset()

    for req, ts_list in access_time.items():
        if len(ts_list) in freq:
            for i in range(len(ts_list)-1):
                dist_list_dict[len(ts_list)].append(ts_list[i+1] - ts_list[i])

    if not os.path.exists("1218ClassByFreqDistPopularity/{}".format(dat_name)):
        os.makedirs("1218ClassByFreqDistPopularity/{}".format(dat_name))

    for freq, dist_list in dist_list_dict.items():
        dist_count_list = [0] * (max(dist_list)+1)
        for dist in dist_list:
            dist_count_list[dist] += 1

        # CDF
        for i in range(1, len(dist_count_list)):
            dist_count_list[i] += dist_count_list[i-1]
        for i in range(len(dist_count_list)):
            dist_count_list[i] = dist_count_list[i] / dist_count_list[-1]

        plt.semilogx(dist_count_list, label=str(freq))
        plt.xlabel("Dist")
        plt.ylabel("CDF Percentage")
        plt.legend(loc="best")
        plt.savefig("1218ClassByFreqDistPopularity/{}/distPop_{}.png".format(dat_name, freq))
        plt.clf()

        plt.plot(dist_count_list, label=str(freq))
        plt.xlabel("Dist")
        plt.ylabel("CDF Percentage")
        plt.legend(loc="best")
        plt.savefig("1218ClassByFreqDistPopularity/{}/distPop2_{}.png".format(dat_name, freq))
        plt.clf()



if __name__ == "__main__":
    from PyMimircache.bin.conf import AKAMAI_CSV3
    from PyMimircache.cacheReader.vscsiReader import VscsiReader
    from PyMimircache.cacheReader.csvReader import CsvReader


    reader0 = VscsiReader("/home/jason/pycharm/mimircache/data/trace.vscsi")
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    reader1 = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    reader2 = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)

    dist_popularity_freq(reader0, "small")

    for dat_name, reader in {"185.232.99.68.anon.1": reader1, "oneDayData.sort": reader2}.items():
        # characterize_by_freq_interval(reader, time_interval=20000, dat_name=dat_name)
        # characterize_by_freq_interval(reader, time_interval=200000, dat_name=dat_name)
        # characterize_by_freq_interval(reader, time_interval=2000000, dat_name=dat_name)
        dist_popularity_freq(reader, dat_name)
