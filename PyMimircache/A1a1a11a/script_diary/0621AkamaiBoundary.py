# coding=utf-8
"""
this module tries to calculate the performance of different 1st/2nd level partition
the metric used for performance evaluation:

Overall Latency
Traffic to Origin
Traffic between tiers



"""

import os, sys, pickle
from PyMimircache import *
from PyMimircache.bin.conf import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


TRACE_DIR, NUM_OF_THREADS = initConf(trace_type="Akamai", trace_format="csv")
TRACE_DIR += "/day/"






############################# CONST #############################
latency_L1      =   10
latency_L2      =   20
latency_ori     =   50

copy_L1         =   20
copy_L2         =   2

CACHE_SIZE = 2000000


############################ METRIC ##############################
latency_overall = -1
traffic_to_origin = 1
traffic_bet_tiers = 1



############################ FUNC ################################
def cal_perf(L1_percentage, dat):
    L1_size = int(CACHE_SIZE * L1_percentage / copy_L1)
    L2_size = int(CACHE_SIZE * (1 - L1_percentage) / copy_L2)


    # reader = csvReader(dat, init_params=AKAMAI_CSV)
    reader = BinaryReader(dat, init_params=AKAMAI_BIN_MAPPED2)

    # p = LRUProfiler(reader, cache_size=L1_size+L2_size)
    if L1_size != 0:
        p_L1 = cGeneralProfiler(reader, cache_size=L1_size, bin_size=L1_size, cache_name="LRU")
        hr_L1 = p_L1.get_hit_rate()[1]
    else:
        hr_L1 = 0

    p_L12 = cGeneralProfiler(reader, cache_size=L1_size+L2_size,
                             bin_size=L1_size+L2_size, cache_name="LRU")
    hr_L12 = p_L12.get_hit_rate()[1]

    latency_overall = hr_L1 * latency_L1 + (hr_L12 - hr_L1) * latency_L2 + (1 - hr_L12) * latency_ori
    traffic_to_origin = 1 - hr_L12 # - hr_L1
    traffic_bet_tiers = hr_L12 - hr_L1

    print("L1 {}, size {} {}, hr1 {}, hr2 {}".format(L1_percentage, L1_size, L2_size, hr_L1, hr_L12))

    return latency_overall,traffic_to_origin, traffic_bet_tiers


def plt_perf(dat):

    # obtain data
    l_percent = [i/100 for i in range(0, 100, 3)]
    l_latecy = [0] * len(l_percent)
    l_traf_ori = [0] * len(l_percent)
    l_traf_tier = [0] * len(l_percent)

    # for i in l_percent:
    #     lat, to, tt = cal_perf(i, dat)
    #     l_latecy.append(lat)
    #     l_traf_ori.append(to)
    #     l_traf_tier.append(tt)


    with ProcessPoolExecutor(max_workers=NUM_OF_THREADS) as e:
        futures = {e.submit(cal_perf, l_percent[i], dat): i for i in range(len(l_percent))}
        for future in as_completed(futures):
            lat, to, tt = future.result()
            i = futures[future]

            l_latecy[i] = lat
            l_traf_ori[i] = to
            l_traf_tier[i] = tt

    # plot
    plt.plot(l_percent, l_latecy, marker="o")
    plt.grid(True)
    plt.xlabel("Cache Size Boundary/L1 Percentage")
    plt.ylabel("Latency (ms)")
    plt.savefig("0621Boundary/latency_{}_size{}.png".format(os.path.basename(dat), CACHE_SIZE))
    plt.clf()

    plt.plot(l_percent, l_traf_ori, marker="o")
    plt.grid(True)
    plt.xlabel("Cache Size Boundary/L1 Percentage")
    plt.ylabel("Traffic to Origin")
    plt.savefig("0621Boundary/traf_ori_{}_size{}.png".format(os.path.basename(dat), CACHE_SIZE))
    plt.clf()

    plt.plot(l_percent, l_traf_tier, marker="o")
    plt.grid(True)
    plt.xlabel("Cache Size Boundary/L1 Percentage")
    plt.ylabel("Traffic between Tiers")
    plt.savefig("0621Boundary/traf_tier_{}_size{}.png".format(os.path.basename(dat), CACHE_SIZE))
    plt.clf()






def find_best_boundary(dat):
    pass



############################ RUNNABLE ############################
def run_plt_perf():
    for f in os.listdir(TRACE_DIR):
        if 'mapped' in f:
            print(f)
            plt_perf("{}/{}".format(TRACE_DIR, f))





if __name__ == "__main__":
    run_plt_perf()
