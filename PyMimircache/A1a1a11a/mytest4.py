# coding=utf-8


import mmap
from collections import defaultdict
from multiprocessing import Process
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.cacheReader.traceStat import TraceStat
from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
from PyMimircache.profiler.cHeatmap import CHeatmap
from PyMimircache.bin.conf import *
from PyMimircache.utils.timer import MyTimer
from PyMimircache import Cachecow, CHeatmap
import random

def mytest1(ewma_coef=8):
    c = Cachecow()
    c.csv("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", init_params=AKAMAI_CSV3)
    reader = c.reader

    last_ts_dict = {}
    d = defaultdict(list)
    for n, i in enumerate(reader):
        last_ts = last_ts_dict.get(i, 0)
        if last_ts != 0:
            d[i].append(n - last_ts)
        last_ts_dict[i] = n
    print("loaded")
    # verify
    correction_list1 = []
    correction_list2 = []
    correction_list3 = []

    multiplier = 2 / (ewma_coef + 1)
    for k, dist_list in d.items():
        if len(dist_list) < 20: continue
        ewma_dist = sum(dist_list[:ewma_coef])/ewma_coef

        dist_error_range = [0] * 10       # 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20
        for i in range(ewma_coef, len(dist_list)):
            dist = dist_list[i]
            dif = dist / ewma_dist
            ewma_dist = ewma_dist * (1 - multiplier) + dist * multiplier
            if dif < 0.01:
                dist_error_range[0] += 1
            elif dif < 0.1:
                dist_error_range[1] += 1
            elif dif < 0.2:
                dist_error_range[2] += 1
            elif dif < 0.5:
                dist_error_range[3] += 1
            elif dif < 1:
                dist_error_range[4] += 1
            elif dif < 2:
                dist_error_range[5] += 1
            elif dif < 5:
                dist_error_range[6] += 1
            elif dif < 10:
                dist_error_range[7] += 1
            elif dif < 20:
                dist_error_range[8] += 1
            else:
                dist_error_range[9] += 1


        # whether we can link this abnormal behavior of one obj with others?


        correction = (dist_error_range[4] + dist_error_range[5]) / (len(dist_list) - 1)
        correction2 = (dist_error_range[3] + dist_error_range[4] + dist_error_range[5] + dist_error_range[6]) / (len(dist_list) - 1)
        correction3 = sum(dist_error_range[:5]) / (len(dist_list) - 1)
        correction_list1.append(correction)
        correction_list2.append(correction2)
        correction_list3.append(correction3)
        # print(dist_error_range)
    print("average correction {:.2f} {:.2f} {:.2f}, max {} min {}".format(
        sum(correction_list1)/len(correction_list1),
        sum(correction_list2)/len(correction_list2),
        sum(correction_list3)/len(correction_list3),
        max(correction_list1),
        min(correction_list1)
    ))


def mytest2(dat="/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.86.anon.1"):
    c = Cachecow()
    c.csv(dat, init_params=AKAMAI_CSV3)
    # c.vscsi("../../data/trace.vscsi")
    # c.plotHRCs(["LRU", "FIFO", "Optimal"], cache_params=[None, None, None], cache_size=200, figname="test.png")
    # c.plotHRCs(["LRU", "FIFO", "Optimal"], cache_params=[None, None, None], cache_size=2000000, figname="185.232.99.68_HRC1.png")
    # c.plotHRCs(["LRU", "Optimal", "ASig0430", "LHD"], cache_params=[None, None, {"lifetime_prob": 0.9999},
    #             {"update_interval": 200000, "coarsen_age_shift": 5, "n_classes":20, "max_coarsen_age":800000, "dat_name": "A"},],
    #            cache_size=2000000, figname="185.232.99.68_HRC2.png")


    cH = CHeatmap()
    cH.heatmap(c.reader, "v", "hr_st_et",
                algorithm="LRU",
                time_interval=20000,
               cache_size=2000000,
                cache_params=None,
               info_on_fig=False,
               figname="heatmap.pdf")


def mytest3(dat, dat_type):
    reader = get_reader(dat, dat_type)
    with open(dat, "w") as ofile:
        for i in reader:
            ofile.write("{}\n".format(i))

def mytest5(dat):

    xlabels = None
    ylabels = None
    X = []
    Y = []
    with open("script_diary/180612Features/{}.X.noColdMissnoScale".format(dat)) as ifilex:
        with open("script_diary/180612Features/{}.Y.noColdMissnoScale".format(dat)) as ifiley:
            for line in ifilex:
                if xlabels is None:
                    xlabels = line.strip("\n").split(",")
                else:
                    X.append(line.strip())
            for line in ifiley:
                if ylabels is None:
                    ylabels = line.strip("\n").split(",")
                else:
                    Y.append(line.strip())

    assert len(X) == len(Y)

    while True:
        r1 = random.randint(0, len(X))
        r2 = random.randint(0, len(X))
        x1 = X[r1].split(",")
        x2 = X[r2].split(",")
        y1 = Y[r1].split(",")
        y2 = Y[r2].split(",")
        print("{} {}".format(r1, r2))
        for i in range(len(x1)):
            print("{:>24}:      {:<24s} {:<24s}".format(xlabels[i], x1[i], x2[i]))
        x = input("wait: ")
        while x != "go":
            print("input " + x)
            if x == "exit":
                sys.exit(1)
        for i in range(len(y1)):
            print("{:>24}:      {:<24s} {:<24s}".format(ylabels[i], y1[i], y2[i]))
        x = input("wait: ")
        if x == "exit":
            sys.exit(1)







if __name__ == "__main__":
    # mytest3("w106", "cphy")
    mytest5("small")