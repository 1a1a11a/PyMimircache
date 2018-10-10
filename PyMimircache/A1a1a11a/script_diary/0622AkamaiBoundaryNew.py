# coding=utf-8
"""
this is the new module for evaluating performance, the difference is the model

this module tries to calculate the performance of different 1st/2nd level partition
the metric used for performance evaluation:

Overall Latency
Traffic to Origin
Traffic between tiers



"""

import os, sys, time, glob, socket, pickle
from PyMimircache import *
from PyMimircache.cacheReader.multiReader import multiReader
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
from PyMimircache.bin.conf import *
# from PyMimircache.profiler.cGeneralProfiler import
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

# copy_L1         =   20
copy_L2         =   2

# CACHE_SIZE = 2000000


############################ METRIC ##############################
latency_overall = -1
traffic_to_origin = 1
traffic_bet_tiers = 1



############################ FUNC ################################
def get_hit_rate(dat, cache_size):
    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    p = cGeneralProfiler(reader, cache_size=cache_size, bin_size=cache_size, cache_name="LRU")
    return p.get_hit_rate()[1]





def cal_perf(L1_percentage, dat, cache_size, dat_type="csv"):
    if len(dat) == 0:
        raise RuntimeError("dat is empty")
    L1_size = int(cache_size * L1_percentage / len(dat))
    L2_size = int(cache_size * (1 - L1_percentage) / copy_L2)
    # assert L1_size!=0 and L2_size!=0, "L1 size {}, L2 size {}".format(L1_size, L2_size)

    print("L1 percent {}, cache_size {}".format(L1_percentage, cache_size))
    # print("L1 percent {}, cache_size {}, dat {}".format(L1_percentage, cache_size, dat))


    if dat_type == "binary":
        reader_type = BinaryReader
        init_params = AKAMAI_BIN_MAPPED2
    elif dat_type == "csv":
        reader_type = CsvReader
        init_params = AKAMAI_CSV
    else:
        raise RuntimeError("does not recognize dat_type")

    reader_list = []

    for datum in dat:
        reader_list.append(reader_type(datum, init_params=init_params))

    if len(reader_list) == 0:
        raise RuntimeError("reader list is empty")


    L1_hitrates = [0] * len(dat)


    if L1_size != 0:
        # L1_hitrates = [cGeneralProfiler(reader, cache_size=L1_size, bin_size=L1_size, cache_name="LRU").get_hit_rate()[1] \
        #                 for reader in reader_list]

        with ProcessPoolExecutor(max_workers=8) as ppe:
            futures = {ppe.submit(get_hit_rate, dat[i], L1_size): i for i in range(len(dat))}
            for future in as_completed(futures):
                hr = future.result()
                i = futures[future]
                L1_hitrates[i] = hr
    else:
        # L1 hitrate all are 0, now prepare data for L2
        mReader = multiReader(reader_list, reading_type="real_time")
        trace_writer = TraceBinaryWriter("/home/jason/special/final_percent{}".format(L1_percentage), fmt="Q")
        mapping_hashtable = {}
        for req in mReader:
            if req not in mapping_hashtable:
                mapping_hashtable[req] = len(mapping_hashtable)
            trace_writer.write((mapping_hashtable[req], ))

    for reader in reader_list:
        reader.close()
    reader_list.clear()

    if L2_size != 0:
        if L1_size != 0:
            # the most common case, take the miss from all the servers in L1, form the data for L2
            for f in [f for f in glob.glob("/home/jason/special/*_size{}".format(L1_size)) if 'final_percent' not in f]:
                reader_list.append(CsvReader(f, init_params={"real_time_column":1, "label_column":2}))
            mReader = multiReader(reader_list, reading_type="real_time")
            trace_writer = TraceBinaryWriter("/home/jason/special/final_percent{}".format(L1_percentage), fmt="Q")
            mapping_hashtable = {}
            for req in mReader:
                if req not in mapping_hashtable:
                    mapping_hashtable[req] = len(mapping_hashtable)
                trace_writer.write((mapping_hashtable[req], ))
            mReader.close_all_readers()
        else:
            # L2!=0, L1=0, the data have already been prepared in last step
            pass

        # get reader for L2
        reader = BinaryReader("/home/jason/special/final_percent{}".format(L1_percentage), init_params={"fmt": "Q", "label":1})
        L2_hitrate = cGeneralProfiler(reader, cache_size=L2_size, bin_size=L2_size, cache_name="LRU").get_hit_rate()[1]
    else:
        L2_hitrate = 0

    hr_L1 = sum(L1_hitrates)/len(L1_hitrates)
    hr_L2 = L2_hitrate

    print("{} L1 size {}, L2 size {}, L1 hit rate {}, L2 hit rate {}".format(L1_percentage, L1_size, L2_size, L1_hitrates, L2_hitrate))

    latency_overall = hr_L1 * latency_L1 + (1-hr_L1) * hr_L2 * latency_L2 + (1 - hr_L1) * (1 - hr_L2) * latency_ori
    traffic_to_origin = (1 - hr_L1) * (1 - hr_L2)
    traffic_bet_tiers = 1 - hr_L1

    print("L1 {}, size {} {}, hr1 {}, hr2 {}".format(L1_percentage, L1_size, L2_size, hr_L1, hr_L2))

    return latency_overall,traffic_to_origin, traffic_bet_tiers


def plt_perf(dat, cache_size, figname="test"):
    # if os.path.exists("0622Boundary/traf_tier_{}_size{}.png".format(figname, cache_size)):
    #     return

    for f in glob.glob("/home/jason/special/*"):
        if os.path.isfile(f):
            os.remove(f)
    # to make sure all processes are synchronized
    num_of_threads = NUM_OF_THREADS // 2
    print("number of threads {}".format(num_of_threads))

    # obtain data
    # l_percent = [i/100 for i in range(48, 80, 4)]
    l_percent = [i/100 for i in range(0, 101, 10)]
    l_latecy = [0] * len(l_percent)
    l_traf_ori = [0] * len(l_percent)
    l_traf_tier = [0] * len(l_percent)

    with ProcessPoolExecutor(max_workers=num_of_threads) as e:
        futures = {e.submit(cal_perf, l_percent[i], dat, cache_size): i for i in range(len(l_percent))}
        for future in as_completed(futures):
            lat, to, tt = future.result()
            i = futures[future]

            l_latecy[i] = lat
            l_traf_ori[i] = to
            l_traf_tier[i] = tt

    # for i in range(len(l_percent)):
    #     # if l_percent[i] in [0, 0.1, 0.2, 0.3]:
    #     #     continuem
    #     print("######################### begin {} ##########################".format(i))
    #     lat, to, tt = cal_perf(l_percent[i], dat, cache_size)
    #     l_latecy[i] = lat
    #     l_traf_ori[i] = to
    #     l_traf_tier[i] = tt



    # plot

    # latency
    plt.figure(1)
    plt.plot(l_percent, l_latecy, marker="o", label="cache size {}".format(cache_size))
    plt.ylim([0, latency_ori])
    plt.grid(True)
    # ylimit = plt.ylim()
    # plt.text(0.4, (ylimit[0]+ylimit[1])/2, "cache size {}".format(cache_size))
    plt.xlabel("Cache Size Boundary/L1 Percentage")
    plt.ylabel("Latency (ms)")
    plt.legend(ncol=2)
    plt.savefig("0622Boundary/latency_{}_size{}.png".format(figname, cache_size))
    # plt.clf()

    plt.figure(2)
    plt.plot(l_percent, l_traf_ori, marker="o", label="cache size {}".format(cache_size))
    plt.ylim([0, 1])
    plt.grid(True)
    # plt.text(0.4, 0.2, "cache size {}".format(cache_size))
    plt.xlabel("Cache Size Boundary/L1 Percentage")
    plt.ylabel("Traffic to Origin")
    plt.legend(ncol=2)
    plt.savefig("0622Boundary/traf_ori_{}_size{}.png".format(figname, cache_size))
    # plt.clf()
    #

    plt.figure(3)
    plt.plot(l_percent, l_traf_tier, marker="o", label="cache size {}".format(cache_size))
    plt.ylim([0, 1])
    plt.grid(True)
    # plt.text(0.4, 0.2, "cache size {}".format(cache_size))
    plt.xlabel("Cache Size Boundary/L1 Percentage")
    plt.ylabel("Traffic between Tiers")
    plt.legend(ncol=2)
    plt.savefig("0622Boundary/traf_tier_{}_size{}.png".format(figname, cache_size))
    # # plt.clf()






def find_best_boundary(dat):
    pass



############################ RUNNABLE ############################
def run_plt_perf():
    folders = ["test"]
    # folders = ["1", "2", "3", "4", "5", "6", "7", "8"]
    for folder in folders:
        # plt_perf(glob.glob("/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/{}/*".format(folder)), cache_size=200000000, figname="temp3")
        for i in range(1, 10):
            # try:
            plt_perf(glob.glob("/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp3/{}/*".format(folder)),
                         cache_size=4**i*100, figname=folder)
            # except Exception as e:
            #     print("ERROR {}".format(e))

def run_plt_perf_aka_data2():
    for i in range(4, 12):
    # i = 2
    # if i == 2:
        print("size {}".format(4**i*100))
        plt_perf(glob.glob("/home/jason/ALL_DATA/akamai_new_logs/*anon"),
                 cache_size=4**i*100, figname="akamai_new")



def mytest():
    reader = CsvReader("/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/ext_80_120/1.all.sort", init_params=AKAMAI_CSV)
    LRUProfiler(reader, cache_size=1200).plotHRC()
    # print(cGeneralProfiler(reader, cache_size=2000, bin_size=2000, cache_name="LRU").get_hit_rate()[1])

def mytest2():
    folder = "1"
    plt_perf(glob.glob("/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/ext_80_120/{}/*".format(folder)),
             cache_size=2 ** 24 * 100, figname=folder)

def verify():
    reader = BinaryReader("/home/jason/special/final_percent0.0", init_params={"fmt": "Q", "label": 1})
    # print(cGeneralProfiler(reader, cache_size=20000000, bin_size=200000, cache_name="LRU", num_of_threads=48).get_hit_rate())
    LRUProfiler(reader, cache_size=2000000).plotHRC()

def performance_analysis():
    import time, cProfile
    t = time.time()
    reader_list = []
    for f in glob.glob("/home/jason/temp2/*_size{}".format(8450)):
    # for f in glob.glob("/home/jason/special/temp/*_size{}".format(8450)):
        reader_list.append(CsvReader(f, init_params={"real_time_column": 1, "label_column": 2}))
    mReader = multiReader(reader_list, reading_type="real_time")
    # trace_writer = traceBinaryWriter("/home/jason/special/temp/final_percent{}".format("test"), fmt="Q")
    trace_writer = TraceBinaryWriter("/home/jason/special/final_percent{}".format("test"), fmt="Q")
    mapping_hashtable = {}

    # cProfile.run()
    for req in mReader:
        if req not in mapping_hashtable:
            mapping_hashtable[req] = len(mapping_hashtable)
        trace_writer.write((mapping_hashtable[req],))


    mReader.close_all_readers()

    print("using time {}s".format(time.time() - t))

if __name__ == "__main__":
    # mytest()
    # cal_perf(0.1, glob.glob("/home/jason/ALL_DATA/Akamai/dataCenterSplitted/csv/temp2/*"), dat_type="csv")
    # run_plt_perf()
    run_plt_perf_aka_data2()
    # verify()
    # mytest2()

    # performance_analysis()