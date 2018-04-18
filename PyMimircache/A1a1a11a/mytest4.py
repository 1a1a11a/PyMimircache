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
from PyMimircache.profiler.generalProfiler import generalProfiler
from PyMimircache.profiler.cHeatmap import CHeatmap
from PyMimircache.bin.conf import *
from PyMimircache.utils.timer import MyTimer
from PyMimircache import Cachecow


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

if __name__ == "__main__":
    mytest1()