# coding=utf-8

import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyMimircache.bin.conf import AKAMAI_CSV3
from PyMimircache import Cachecow
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.traceStat import TraceStat
from PyMimircache.utils.timer import MyTimer
from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
from PyMimircache.cache.INTERNAL.ASig import ASig



def HRC():
    CACHE_SIZE = 200000
    BIN_SIZE = 5000

    mt = MyTimer()
    # reader = VscsiReader("/home/jason/pycharm/mimircache/data/trace.vscsi")
    # reader = VscsiReader("/home/cloudphysics/traces/w106_vscsi1.vscsitrace")
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)

    # lp = CLRUProfiler(reader, cache_size=200000)
    # lp.plotHRC("akamaiLRU.png")
    p0 = PyGeneralProfiler(reader, "LRU", cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=os.cpu_count())
    p0.plotHRC(no_clear=True, figname="185.232.99.68.anon.1_LRU.png")

    reader.reset()
    mt.tick()

    g = PyGeneralProfiler(reader, ASig, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=os.cpu_count())
    g.plotHRC(figname="185.232.99.68.anon.1_ASig_0.98_noDyn.png")
    mt.tick()


def get_list_of_obj_with_freq():
    """
    this function gets the freq of each obj and write "freq: obj" in ascending order into file

    """
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    d = defaultdict(int)
    for r in reader:
        d[r] += 1

    with open("19.28.122.183.anon.freq", "w") as ofile:
        for r, freq in sorted(d.items(), key=lambda x: x[1]):
            ofile.write("{:-8d}\t:\t{}\n".format(freq, r))


def transform_into_accesstime_list():
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    access_time = defaultdict(list)
    for n, r in enumerate(reader):
        access_time[r].append(n)

    with open("19.28.122.183.anon.accesstime.pickle", "wb") as ofile:
        pickle.dump(access_time, ofile)


def analyze_low_freq_obj(cutoff_freq_min = 2, cutoff_freq_max = 2):
    with open("19.28.122.183.anon.accesstime.pickle", "rb") as ifile:
        access_time = pickle.load(ifile)

    dist_list = [0]


    with open("19.28.122.183.anon.freq", "r") as ifile:
        for line in ifile:
            freq, obj = line.split(":")
            freq = int(freq.strip())
            obj = obj.strip()
            if freq < cutoff_freq_min:
                continue
            if freq > cutoff_freq_max:
                break
            access_time_list = access_time[obj]
            print(", ".join([str(access_time_list[i+1] - access_time_list[i]) for i in range(len(access_time_list)-1)]))

            assert len(access_time_list) == freq, "access_time_list {}, freq {}".format(len(access_time_list), freq)
            for i in range(len(access_time_list)-1):
                dist = access_time_list[i+1] - access_time_list[i]
                if dist > len(dist_list):
                    dist_list.extend([0] * (dist - len(dist_list)))
                dist_list[dist - 1] += 1

    # d = {}
    # for n, i in enumerate(dist_list):
    #     if i>2:
    #         d[n] = i
    # print(d)
    # plt.hist(dist_list, bins=200)

    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.loglog(dist_list)
    plt.savefig("hist_{}_{}.png".format(cutoff_freq_min, cutoff_freq_max))
    plt.clf()

    for i in range(len(dist_list)-2, -1, -1):
        dist_list[i] += dist_list[i+1]
    plt.xlabel("Distance")
    plt.ylabel("CCDF Count")
    plt.loglog(dist_list)
    plt.savefig("hist_CCDF_{}_{}.png".format(cutoff_freq_min, cutoff_freq_max))
    plt.clf()




def characterize():
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)

    # ts = TraceStat(reader)
    # print(ts.get_stat())

    c = Cachecow()
    # c.csv("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    # c.csv("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    c.csv("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)
    figname_prefix = os.path.basename(c.reader.file_loc)
    c.plotHRCs(["LRU", "LFUFast", "ARC", "SLRU", "Optimal"], figname="HRC_{}.png".format(figname_prefix), save_gradually=True)
    c.twoDPlot("request_rate", time_mode="r", time_interval=600, figname="{}_requestRate.png".format(figname_prefix))
    c.twoDPlot("cold_miss_count", time_mode="r", time_interval=600, figname="{}_coldMissCount.png".format(figname_prefix))
    c.twoDPlot("popularity", figname="{}_popularity.png".format(figname_prefix))
    c.twoDPlot("rd_popularity", figname="{}_rd_popularity.png".format(figname_prefix))
    c.twoDPlot("rd_popularity", plot_type="req", figname="{}_rd_req_popularity.png".format(figname_prefix))
    c.twoDPlot("rt_popularity", figname="{}_rt_popularity.png".format(figname_prefix))
    c.twoDPlot("mapping")


def mytest_1208():
    CACHE_SIZE = 20000
    BIN_SIZE = 5000

    mt = MyTimer()
    reader = VscsiReader("/home/jason/pycharm/mimircache/data/trace.vscsi")

    lp = CLRUProfiler(reader, cache_size=CACHE_SIZE)
    lp.plotHRC("akamaiLRU.png", no_clear=True, no_save=True)

    p0 = PyGeneralProfiler(reader, "LRU", cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=os.cpu_count())
    p0.plotHRC(no_clear=True, figname="test.png")

    reader.reset()
    mt.tick()

    mt.tick()




if __name__ == "__main__":
    mt = MyTimer()
    mytest_1208()
    # HRC()
    # characterize()
    # get_list_of_obj_with_freq()
    # transform_into_accesstime_list()

    # analyze_low_freq_obj(20, 20)


    # for i in range(3, 80):
    #     analyze_low_freq_obj(i, i)
    # for i in range(3, 60):
    #     analyze_low_freq_obj(2, i)




    mt.tick()