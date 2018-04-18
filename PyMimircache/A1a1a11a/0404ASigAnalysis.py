

import os, sys, time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from PyMimircache import CsvReader
from PyMimircache.bin.conf import *


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




if __name__ == "__main__":
    # plt_class_percentage(class_boundary=(12, 2000), time_interval=2000000)
    # plt_class_percentage2(class_boundary=(12, 2000), time_interval=200000, cache_size=200000)
    conversion_rate = cal_conversion_rate(dat=None, freq_boundary=12)

