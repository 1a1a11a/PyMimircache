

import os, sys, time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from PyMimircache import CsvReader
from PyMimircache.bin.conf import *
from PyMimircache.cache.optimal import Optimal
from PyMimircache.cache.lru import LRU


# def plt_class_percentage(dat=None, class_boundary=(12, 2000), time_interval=20000):
def plt_class_percentage(dat=None, class_boundary=(12, 12), time_interval=20000):
    """
    this function plots the percentage of low, mid, high freq obj as time evolves
    :param dat:
    :param class_boundary:
    :param time_interval:
    :return:
    """


    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)
    # reader = VscsiReader("../../data/trace.vscsi")

    low_freq_portion_list = []
    mid_freq_portion_list = []
    high_freq_portion_list = []

    freq_dict = defaultdict(int)
    count = [0, 0, 0]

    for n, req in enumerate(reader):
        if n % time_interval == 0:
            if n != 0:
                low_freq_portion_list.append(count[0]/time_interval)
                mid_freq_portion_list.append(count[1]/time_interval)
                high_freq_portion_list.append(count[2]/time_interval)
            if n % (time_interval * 20) == 0:
                plt.plot(low_freq_portion_list, label="low")
                plt.plot(mid_freq_portion_list, label="mid")
                plt.plot(high_freq_portion_list, label="high")
                plt.ylabel("Percentage")
                plt.xlabel("Time (v)")
                plt.legend(loc="best")
                plt.savefig("class_percent.png")
                plt.clf()

            count = [0, 0, 0]
        freq_dict[req] += 1
        if freq_dict[req] <= class_boundary[0]:
            # low-freq
            count[0] += 1
        elif class_boundary[0] <= freq_dict[req] <= class_boundary[1]:
            # mid-freq
            count[1] += 1
        elif freq_dict[req] > class_boundary[1]:
            # high-freq
            count[2] += 1
        else:
            raise RuntimeError("unknown")

    plt.plot(low_freq_portion_list, label="low")
    plt.plot(mid_freq_portion_list, label="mid")
    plt.plot(high_freq_portion_list, label="high")
    plt.ylabel("Percentage")
    plt.xlabel("Time (v)")
    plt.legend(loc="best")
    plt.savefig("class_percent.png")
    plt.clf()


def plt_class_percentage2(dat=None, class_boundary=(12, 2000), time_interval=20000, cache_size=200000, decay=0.6):
    """
    this function plots the percentage of low, mid, high freq obj as time evolves
    :param dat:
    :param class_boundary:
    :param time_interval:
    :return:
    """


    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)
    # reader = VscsiReader("../../data/trace.vscsi")

    low_freq_portion_list = []
    mid_freq_portion_list = []
    high_freq_portion_list = []

    freq_dict = defaultdict(int)
    count = [0, 0, 0]

    for n, req in enumerate(reader):
        if n % time_interval == 0:
            if n != 0:
                low_freq_portion_list.append(count[0]/(sum(count)))
                mid_freq_portion_list.append(count[1]/sum(count))
                high_freq_portion_list.append(count[2]/sum(count))
            if n % (time_interval * 20) == 0:
                plt.plot(low_freq_portion_list, label="low")
                plt.plot(mid_freq_portion_list, label="mid")
                plt.plot(high_freq_portion_list, label="high")
                plt.ylabel("Percentage")
                plt.xlabel("Time (v)")
                plt.legend(loc="best")
                plt.savefig("class_percent2OneDay.png")
                plt.clf()
        if n % (cache_size*class_boundary[0]) == 0 and n != 0:
            for k in freq_dict.keys():
                freq_dict[k] = freq_dict[k] * decay

        freq_dict[req] += 1
        if freq_dict[req] <= class_boundary[0]:
            # low-freq
            count[0] += 1
        elif class_boundary[0] <= freq_dict[req] <= class_boundary[1]:
            # mid-freq
            count[1] += 1
        elif freq_dict[req] > class_boundary[1]:
            # high-freq
            count[2] += 1
        else:
            raise RuntimeError("unknown")

    plt.plot(low_freq_portion_list, label="low")
    plt.plot(mid_freq_portion_list, label="mid")
    plt.plot(high_freq_portion_list, label="high")
    plt.ylabel("Percentage")
    plt.xlabel("Time (v)")
    plt.legend(loc="best")
    plt.savefig("class_percent2OneDay.png")
    plt.clf()


def cal_conversion_rate(dat, freq_boundary):
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)
    # reader = VscsiReader("../../data/trace.vscsi")

    obj_freq_first_part = defaultdict(int)
    obj_freq_all = defaultdict(int)

    total_num = reader.get_num_of_req()
    for n, req in enumerate(reader):
        obj_freq_all[req] += 1
        if n < total_num // (freq_boundary + 2):
            obj_freq_first_part[req] += 1


    conversion_rate = [0] * freq_boundary
    converted = [0] * freq_boundary
    non_converted = [0] * freq_boundary

    for obj, freq in obj_freq_first_part.items():
        if freq > freq_boundary:
            continue
        freq_all = obj_freq_all[obj]
        if freq_all > freq_boundary:
            converted[freq-1] += 1
        else:
            non_converted[freq-1] += 1

    for i in range(0, freq_boundary):
        conversion_rate[i] = converted[i] / (converted[i] + non_converted[i])

    for i in range(12):
        print("{} :{}".format(i+1, conversion_rate[i]))


    return conversion_rate

def low_freq_distr_time(freq=2, window_size=20000, num_window=6):
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)


def plot_eviction_freq(dat="/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", freq_boundary=12, time_interval=80000, alg="Optimal", cache_size=8000):
    """
    of all the objs evicted by alg, how many are low-freq, mid-freq
    :param dat:
    :param freq:
    :return:
    """
    reader = CsvReader(dat, init_params=AKAMAI_CSV3)
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", init_params=AKAMAI_CSV3)
    # reader = VscsiReader("../data/trace.vscsi")
    if alg == "Optimal":
        cache = Optimal(cache_size, reader)
    elif alg == "LRU":
        cache = LRU(cache_size)
    else:
        raise RuntimeError("not support")


    eviction_list = []
    evict_freq_list = []
    freq_dict = defaultdict(int)
    for n, r in enumerate(reader):
        freq_dict[r] += 1

        cache.access(r, evict_item_list=eviction_list)
        if len(eviction_list):
            evict_freq_list.append(freq_dict[eviction_list[0]])
            eviction_list.pop()

        # if r in cache:
        #     cache._update(r)
        # else:
        #     cache._insert(r)
        #     # print("insert {} into {}".format(r, cache))
        #     if len(cache) > cache_size:
        #         obj = cache.evict()
        #         eviction_list.append(obj)
        #         evict_freq_list.append(freq_dict[obj])
        #     else:
        #         eviction_list.append(None)
        #         # evict_freq_list.append(-1)

    print("{} {}".format(len(evict_freq_list), evict_freq_list[:200]))

    low_mid_percent_list = []
    n_low = 0
    n_mid = 0
    for n, freq in enumerate(evict_freq_list):
        if freq <= freq_boundary:
            n_low += 1
        else:
            n_mid += 1
        if n and n % time_interval == 0:
            low_mid_percent_list.append(n_low/(n_low + n_mid))
            n_low = 0
            n_mid = 0

    print(low_mid_percent_list[:200])
    plt.plot(low_mid_percent_list)
    plt.xlabel("Time (interval={})".format(time_interval))
    plt.ylabel("Low Freq Percentage ({})".format(alg))
    plt.savefig("low_mid_{}_{}2.png".format(alg, os.path.basename(reader.file_loc)))






if __name__ == "__main__":
    # plt_class_percentage(class_boundary=(12, 2000), time_interval=2000000)
    # plt_class_percentage2(class_boundary=(12, 2000), time_interval=200000, cache_size=200000)
    # conversion_rate = cal_conversion_rate(dat=None, freq_boundary=12)
    # plot_eviction_freq(alg="Optimal")
    plot_eviction_freq(alg="LRU")
