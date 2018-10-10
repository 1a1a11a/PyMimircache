# coding=utf-8
# from PyMimircache import *
from collections import deque, defaultdict
from PyMimircache.utils.printing import print_list
import csv
import math
import numpy as np
from matplotlib import pyplot as plt

import os, sys, time, pickle
from file_read_backwards import FileReadBackwards
from multiprocessing import Pool, Process, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
from PyMimircache.bin.conf import initConf
from PyMimircache.A1a1a11a.ML.featureGenerator import *

################################## Global variable and initialization ###################################

TRACE_TYPE = "cloudphysics"

TRACE_DIR, NUM_OF_THREADS = initConf(TRACE_TYPE, trace_format='variable')


###################################### main funcitons ########################################

def get_frd(reader):
    frd = c_LRUProfiler.get_future_reuse_dist(reader.cReader)
    print("frd len = {}".format(len(frd)))
    reader.reset()
    return frd


def get_rd(reader):
    rd = c_LRUProfiler.get_reuse_dist_seq(reader.cReader)
    print("rd len = {}".format(len(rd)))
    reader.reset()
    return rd

def get_frt(reader):
    pass
    # t = reader.read_time_request()  # t: (time, request)
    # d = {}
    # frt = []
    # for i in reader:
    #     if i in d:
    #         freq_list.append(d[i])
    #         d[i] += 1
    #     else:
    #         freq_list.append(1)
    #         d[i] = 1
    # reader.reset()
    # print("freq len = {}".format(len(freq_list)))
    # return freq_list


def get_reuse_time(reader):
    t = reader.read_time_request()  # t: (time, request)
    d = {}
    reuse_time_list = []
    while t:
        if t[1] in d:
            reuse_time_list.append( int((t[0] - d[t[1]])/1e6) )  # current time - last time seen it
            d[t[1]] = t[0]
        else:
            reuse_time_list.append(-1)
            d[t[1]] = t[0]
        t = reader.read_time_request()
    reader.reset()
    print("reuse real time len = {}".format(len(reuse_time_list)))
    return reuse_time_list




def get_freq(reader):
    d = {}
    freq_list = []
    for i in reader:
        if i in d:
            freq_list.append(d[i])
            d[i] += 1
        else:
            freq_list.append(1)
            d[i] = 1
    reader.reset()
    print("freq len = {}".format(len(freq_list)))
    return freq_list


def get_request_rate(reader, time_interval):
    request_rate_list = []
    init_time = -1
    time_queue = deque()
    last_time = -1
    t = reader.read_time_request()
    num = 0

    while t:
        time_queue.append(t[0])

        if init_time == -1:
            init_time = t[0]
        if last_time == -1:
            last_time = t[0]
        while t[0] - last_time > time_interval:
            last_time = time_queue.popleft()

        request_rate_list.append(len(time_queue))
        t = reader.read_time_request()
        num += 1


    reader.reset()
    print("request rate len = {}".format(len(request_rate_list)))
    return request_rate_list


def get_cold_miss_rate(reader, time_interval):
    cold_miss_rate_list = []
    seen_set = set()
    init_time = -1
    time_queue = deque()
    last_time = -1
    cold_miss_count = 0
    t = reader.read_time_request()
    while t:
        if t[1] not in seen_set:
            time_queue.append( (t[0], 1) )
            seen_set.add(t[1])
            cold_miss_count +=1
        else:
            time_queue.append( (t[0], 0) )

        if init_time == -1:
            init_time = t[0]
        if last_time == -1:
            last_time = t[0]
        while t[0] - last_time > time_interval:
            last_time, seen_or_not = time_queue.popleft()
            cold_miss_count -= seen_or_not

        cold_miss_rate_list.append(cold_miss_count)
        t = reader.read_time_request()

    reader.reset()
    print("cold_miss_rate_list rate len = {}".format(len(cold_miss_rate_list)))
    # print(cold_miss_rate_list)
    return cold_miss_rate_list


def get_distance_second_to_last_request(reader):
    seen_first_set = set()
    ts_2_to_last = {}
    result_list =[]
    for num, element in enumerate(reader):
        if element in seen_first_set:
            if element in ts_2_to_last:
                # result_list.append(num - ts_2_to_last[element])
                result_list.append(0)
                ts_2_to_last[element] = num
            else:
                ts_2_to_last[element] = num
                result_list.append(1)      # seen once not twice
        else:
            result_list.append(2)
            seen_first_set.add(element)

    reader.reset()
    print("distance second to last request len = {}".format(len(result_list)))
    print(result_list[100])
    # print(result_list)
    return result_list


def get_feature(reader, time_interval, cache_size, train_ratio, cutoff_head=0.2, cutoff_tail=0.2):
    """

    :param reader:
    :param time_interval:
    :param train_ratio:
    :param cutoff_head:  discard the head part because reuse distance at this part is no use (pollute the result)
    :param cutoff_tail:  discard the tail part because future reuse distance at this part is no use (pollute the result)
    :return:
    """
    frd = get_frd(reader)
    # frd_new = [int(math.log(i+1)/2) if i!=-1  else int(math.log(10000000000)) for i in frd]
    # frd_new = [50000 if i==-1 else i for i in frd]
    frd_new = frd

    # frd_new = [int(math.log(i+2)/2) for i in frd]


    rd = get_rd(reader)
    # rd_new1 = [1 if i==-1 or i>20000 else 0 for i in rd]
    # rd_new2 = [i+1 if i!=-1 and i<=20000 else 0 for i in rd]

    freq = get_freq(reader)
    request_rate = get_request_rate(reader, time_interval)
    cold_miss_rate = get_cold_miss_rate(reader, time_interval)
    second_distance = get_distance_second_to_last_request(reader)
    # print(frd)
    # print(rd)
    # print(freq)
    # print(request_rate)
    # print(cold_miss_rate)
    # print(second_distance)

    # determine the number of samples for training and evaluation
    total_num = reader.get_num_of_total_requests()
    cutoff_head_num = int(total_num * cutoff_head)
    cutoff_tail_num = int(total_num * cutoff_tail)

    train_num = int((total_num - cutoff_head_num - cutoff_tail_num) * train_ratio)
    eval_num  = (total_num - cutoff_head_num - cutoff_tail_num) - train_num
    print("train: {}, evaluation: {}".format(train_num, eval_num))

    with open('train.csv', 'w') as ofile_train:
        with open('evaluation.csv', 'w') as ofile_eval:

            writer_train = csv.writer(ofile_train)
            writer_eval = csv.writer(ofile_eval)

            for i in range(len(frd)):
                if i < cutoff_head_num:
                    continue
                if i > total_num - cutoff_tail_num:
                    break
                row = [frd_new[i], rd[i], freq[i], request_rate[i], cold_miss_rate[i]]  # rd_new1[i], rd_new2[i],
                if i <= train_num:
                    writer_train.writerow(row)
                else:
                    writer_eval.writerow(row)



def get_feature(reader, time_interval, cache_size, train_ratio, cutoff_head=0.2, cutoff_tail=0.2):
    """

    :param cache_size:
    :param reader:
    :param time_interval:
    :param train_ratio:
    :param cutoff_head:  discard the head part because reuse distance at this part is no use (pollute the result)
    :param cutoff_tail:  discard the tail part because future reuse distance at this part is no use (pollute the result)
    :return:
    """
    frd = get_frd(reader)
    # frd_new = [int(math.log(i+1)/2) if i!=-1  else int(math.log(10000000000)) for i in frd]
    # frd_new = [50000 if i==-1 else i for i in frd]
    frd_new = frd

    # frd_new = [int(math.log(i+2)/2) for i in frd]


    rd = get_rd(reader)
    # rd_new1 = [1 if i==-1 or i>20000 else 0 for i in rd]
    # rd_new2 = [i+1 if i!=-1 and i<=20000 else 0 for i in rd]

    freq = get_freq(reader)
    request_rate = get_request_rate(reader, time_interval)
    cold_miss_rate = get_cold_miss_rate(reader, time_interval)
    second_distance = get_distance_second_to_last_request(reader)


    # determine the number of samples for training and evaluation
    total_num = reader.get_num_of_req()
    cutoff_head_num = int(total_num * cutoff_head)
    cutoff_tail_num = int(total_num * cutoff_tail)

    train_num = int((total_num - cutoff_head_num - cutoff_tail_num) * train_ratio)
    eval_num  = (total_num - cutoff_head_num - cutoff_tail_num) - train_num
    print("train: {}, evaluation: {}".format(train_num, eval_num))

    with open('train.csv', 'w') as ofile_train:
        with open('evaluation.csv', 'w') as ofile_eval:

            writer_train = csv.writer(ofile_train)
            writer_eval = csv.writer(ofile_eval)

            for i in range(len(frd)):
                if i < cutoff_head_num:
                    continue
                if i > total_num - cutoff_tail_num:
                    break
                row = [frd_new[i], rd[i], freq[i], request_rate[i], cold_miss_rate[i]]  # rd_new1[i], rd_new2[i],
                if i - cutoff_head_num <= train_num:
                    writer_train.writerow(row)
                else:
                    writer_eval.writerow(row)


def plot_reuse_time(reader, interval):
    VERTICAL_POINTS = 20
    rt = get_reuse_time(reader)

    pos = 0
    count = 0
    # rd_array = np.ma.masked_less(rd_array, 0)
    matrix = np.zeros( (int( (len(rt)/interval+1 )), VERTICAL_POINTS), dtype=float )

    minus_one_counter = 0
    for i in range(len(rt)):
        count += 1
        realtime = rt[i]
        if realtime == -1:
            minus_one_counter += 1

        elif realtime == 0:
            realtime = 1
            matrix[pos][0] += 1

        elif realtime < 0:
            print(realtime)
            realtime = 1
            matrix[pos][0] += 1

        else:
            matrix[pos][int(math.log2(realtime))] += 1

        if count == interval:
            pos += 1
            count = 0
            minus_one_counter = 0
            for j in range(VERTICAL_POINTS):
                matrix[pos-1][j] /= float(interval - minus_one_counter)


    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(x * 100 ))
    yticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(2**x))
    # yticks = ticker.IndexFormatter([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # plt.imshow(matrix.T, interpolation='nearest', origin='lower', aspect='auto')

    plt.gca().xaxis.set_major_formatter(xticks)
    plt.gca().yaxis.set_major_formatter(yticks)

    img = plt.imshow(matrix.T, interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=1)
    cb = plt.colorbar(img)
    plt.savefig("reuse_time_heat", dpi=600)
    plt.clf()





if __name__ == "__main__":
    from pprint import pprint
    dat = "../data/trace.vscsi"
    trace_num = "test"
    t_virtual = 2000 # 60 * 1000000
    t_real = 60 * 1000000

    ################################ temporary generation functions ##################################
    # trace_num = "w101"
    # dat = "{}/{}_vscsi1.vscsitrace".format(TRACE_DIR, trace_num)

    t1 = time.time()
    # get_feature_specific(dat, time_interval=t_virtual,
    #                      outfile="features/specific_rd/{}_t{}.csv".format(trace_num, t_virtual))
    get_feature_interval(dat, time_interval=t_real,
                         outfile="features/interval/{}_t{}.csv".format(trace_num, t_real))
    print(time.time() - t1)
    # get_feature(reader, 1000000, cache_size=20000, train_ratio=0.6, cutoff_head=0.2, cutoff_tail=0.2)
    sys.exit(0)

    ################################## cal correlation coefficient ##################################
    # with Pool(cpu_count()) as p:
    #     p.map(rd_frd_coefficient,
    #           ["{}/{}".format(TRACE_DIR, f)
    #                        for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace")])
    # sys.exit(0)

    ################################## BATCH JOB: generate features #################################
    # with ProcessPoolExecutor(4) as p:
    #     futures = {p.submit(get_feature_specific,
    #                         "{}/{}".format(TRACE_DIR, f),
    #                         t,
    #                         "features/specific/{}_t{}.csv".format(f[:f.find('_')], t)):
    #                    f for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace")}
    #     for future in as_completed(futures):
    #         print(future)

    with ProcessPoolExecutor(4) as p:
        futures = {p.submit(get_feature_specific,
                            "{}/{}".format(TRACE_DIR, f),
                            t,
                            "features/specific_rd/{}_t{}".format(f[:f.find('_')], t_virtual)):
                       f for f in os.listdir(TRACE_DIR) if f.endswith("vscsitrace")}
        for future in as_completed(futures):
            print(futures[future])


