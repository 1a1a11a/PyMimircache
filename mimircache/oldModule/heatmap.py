'''
this module provides the heatmap ploting engine, it supports both virtual time (fixed number of trace requests) and
real time, under both modes, it support using multiprocessing to do the plotting 
'''

from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cache.LRU import LRU
from mimircache.profiler.pardaProfiler import pardaProfiler
import pickle
import numpy as np
import sys
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue, Pool
import time


def prepare_heatmap_dat(bin_size=1000, cache_size=2000):
    reader = csvCacheReader("../data/trace_CloudPhysics_txt", 4, delimiter=' ')
    total_line = reader.get_num_total_lines()
    p = pardaProfiler(30000, reader)
    c_reuse_dist_long_array = p.get_reuse_distance()

    array_len = len(c_reuse_dist_long_array) // bin_size
    result = np.empty((array_len - 1, array_len), dtype=np.float32)
    result[:] = np.nan
    # (x, y) -> heat, x, y is the left, lower point of heat square
    # (x,y) means from time x to time y

    print(len(c_reuse_dist_long_array))
    # print((c_reuse_dist_long_array))


    with open('reuse.dat', 'w') as ofile:
        for i in c_reuse_dist_long_array:
            print(i)
            ofile.write(str(i) + '\n')
    sys.exit(1)

    for (x, y) in _get_xy_distribution(bin_size, len(c_reuse_dist_long_array)):
        print('({},{})'.format(x, y))
        hr = _calc_hit_rate(c_reuse_dist_long_array, cache_size, x, y)
        result[x // bin_size][y // bin_size] = hr

    print(result)
    with open('temp2', 'wb') as ofile:
        pickle.dump(result, ofile)


def prepare_heatmap_dat_multiprocess(bin_size=1000, cache_size=2000, num_of_process=8):
    reader = plainCacheReader("../data/parda.trace")
    total_line = reader.get_num_total_lines()
    p = pardaProfiler(30000, reader)
    c_reuse_dist_long_array = p.get_reuse_distance()

    array_len = len(c_reuse_dist_long_array) // bin_size
    result = np.empty((array_len - 1, array_len), dtype=np.float32)
    result[:] = np.nan
    # (x, y) -> heat, x, y is the left, lower point of heat square
    # (x,y) means from time x to time y

    print(len(c_reuse_dist_long_array))
    # print((c_reuse_dist_long_array[-2]))



    # for i in c_reuse_dist_long_array:
    #     print(i)

    l = _get_xy_distribution_list(bin_size, len(c_reuse_dist_long_array))

    divided_len = len(l) // num_of_process
    q = Queue()
    process_list = []
    for i in range(num_of_process):
        if i != num_of_process - 1:
            p = Process(target=_calc_hit_rate_multiprocess,
                        args=(c_reuse_dist_long_array, cache_size, l[divided_len * i:divided_len * (i + 1)], q))
            process_list.append(p)
            p.start()
        else:
            p = Process(target=_calc_hit_rate_multiprocess,
                        args=(c_reuse_dist_long_array, cache_size, l[divided_len * i:len(l)], q))
            process_list.append(p)
            p.start()

    num_left = len(l)
    while (num_left):
        t = q.get()
        result[t[0] // bin_size][t[1] // bin_size] = t[2]
        num_left -= 1
        print(num_left)

    print(result)
    with open('temp', 'wb') as ofile:
        pickle.dump(result, ofile)


def prepare_heatmap_dat_multiprocess_ts(datapath="../data/trace_CloudPhysics_bin",
                                        time_interval=10000000, cache_size=2000, num_of_process=8, calculate=True):
    reader = vscsiCacheReader(datapath)
    reuse_dist_python_list = []
    break_points = None

    if calculate:
        p = pardaProfiler(30000, reader)
        c_reuse_dist_long_array = p.get_reuse_distance()

        # print(len(c_reuse_dist_long_array))
        for i in c_reuse_dist_long_array:
            reuse_dist_python_list.append(i)

        with open('reuse.dat', 'wb') as ofile:
            pickle.dump(reuse_dist_python_list, ofile)
        print(len(reuse_dist_python_list))

    else:
        with open('reuse.dat', 'rb') as ifile:
            reuse_dist_python_list = pickle.load(ifile)

        if os.path.exists('break_points_' + str(time_interval) + '.dat'):
            with open('break_points_' + str(time_interval) + '.dat', 'rb') as ifile:
                break_points = pickle.load(ifile)

    # xy_list is a two dimensional square matrix, the returned size is the size of one dimension

    # old
    # xy_list, xy_dict, size = _get_xy_distribution_list_timestamp(reader, time_interval)

    # new
    if not break_points:
        break_points = _get_xy_distribution_list_timestamp(reader, time_interval)
        with open('break_points_' + str(time_interval) + '.dat', 'wb') as ifile:
            pickle.dump(break_points, ifile)

    # old
    # array_len = size

    # new
    array_len = len(break_points)
    result = np.empty((array_len, array_len), dtype=np.float32)
    result[:] = np.nan
    # (x, y) -> heat, x, y is the left, lower point of heat square
    # (x,y) means from time x to time y


    # divided_len = len(xy_list)//num_of_process      # old
    q = Queue()
    process_list = []

    # efficiency can be further improved by porting into Cython and improve parallel logic
    map_list = [i for i in range(len(break_points) - 1)]
    print(len(map_list))

    count = 0
    async_result_list = []
    with Pool(processes=num_of_process, initializer=_cal_hit_rate_multiprocess_ts2_init, \
              initargs=[reuse_dist_python_list, cache_size, break_points, q]) as p:
        for l in p.imap_unordered(_calc_hit_rate_multiprocess_ts2, map_list):
            count += 1
            print("%2.2f%%" % (count / len(map_list) * 100))
            for t in l:  # l is a list of (x, y, hr)
                result[t[0]][t[1]] = t[2]

    # old version not efficient
    # for i in range(num_of_process):
    #     if i!=num_of_process-1:
    #         p = Process(target=_calc_hit_rate_multiprocess_ts,
    #                     args=(reuse_dist_python_list, cache_size, xy_list[divided_len*i:divided_len*(i+1)], q))
    #         process_list.append(p)
    #         p.start()
    #     else:
    #         p = Process(target=_calc_hit_rate_multiprocess_ts,
    #                     args=(reuse_dist_python_list, cache_size, xy_list[divided_len*i:len(xy_list)], q))
    #         process_list.append(p)
    #         p.start()
    #
    # num_left = len(xy_list)
    # while (num_left):
    #     t = q.get()
    #     result[t[0]][t[1]] = t[2]
    #     num_left -= 1
    #     print(num_left)


    # print(result)
    with open('temp', 'wb') as ofile:
        pickle.dump(result, ofile)


def _get_xy_distribution(bin_size, total_length):
    for i in range(0, total_length - bin_size + 1, bin_size):
        # if the total length is not multiple of bin_size, discard the last partition
        for j in range(i + bin_size, total_length - bin_size + 1, bin_size):
            yield (i, j)


def _get_xy_distribution_list(bin_size, total_length):
    l = []
    for i in range(0, total_length - bin_size + 1, bin_size):
        # if the total length is not multiple of bin_size, discard the last partition
        for j in range(i + bin_size, total_length - bin_size + 1, bin_size):
            l.append((i, j))
    return l


def _get_xy_distribution_list_timestamp(reader, time_interval):
    xy_list = []
    xy_dict = dict()
    # generate break point
    # reader.reset()
    break_points = []
    prev = 0
    for num, line in enumerate(reader.lines()):
        if (line[0] - prev) > time_interval:
            break_points.append(num)
            prev = line[0]
    # print(num)
    if line[0] != prev:
        break_points.append(num)
    # print(break_points)
    # print(len(break_points))

    if (len(break_points)) > 10000:
        print("number of bins more than 10000, exact size: %d, it may be too slow" % len(break_points))

    return break_points

    # old
    # generate the list for x,y coordinate
    # for i in range(len(break_points)):
    #     # if the total length is not multiple of bin_size, discard the last partition
    #     for j in range(i+1, len(break_points)):
    #         xy_list.append((i, j, break_points[i], break_points[j]))
    #         xy_dict[(i, j)] = (break_points[i], break_points[j])
    #
    # return (xy_list, xy_dict, len(break_points))


def _calc_hit_rate(reuse_dist_array, cache_size, begin_pos, end_pos):
    hit_count = 0
    miss_count = 0
    for i in range(begin_pos, end_pos):
        if reuse_dist_array[i] == -1:
            # never appear
            miss_count += 1
            continue
        if reuse_dist_array[i] - (i - begin_pos) < 0 and reuse_dist_array[i] < cache_size:
            # hit
            hit_count += 1
        else:
            # miss
            miss_count += 1
    # print("hit+miss={}, total size:{}, hit rage:{}".format(hit_count+miss_count, end_pos-begin_pos, hit_count/(end_pos-begin_pos)))
    return hit_count / (end_pos - begin_pos)


def _calc_hit_count(reuse_dist_array, cache_size, begin_pos, end_pos, real_start):
    '''

    :rtype: count of hit 
    :param reuse_dist_array:
    :param cache_size:
    :param begin_pos:
    :param end_pos:
    :param real_start: the real start position of cache trace
    :return:
    '''
    hit_count = 0
    miss_count = 0
    for i in range(begin_pos, end_pos):
        if reuse_dist_array[i] == -1:
            # never appear
            miss_count += 1
            continue
        if reuse_dist_array[i] - (i - real_start) < 0 and reuse_dist_array[i] < cache_size:
            # hit
            hit_count += 1
        else:
            # miss
            miss_count += 1
    # print("hit+miss={}, total size:{}, hit rage:{}".format(hit_count+miss_count, end_pos-begin_pos, hit_count/(end_pos-begin_pos)))
    return hit_count


def _calc_hit_rate_multiprocess(reuse_dist_array, cache_size, xy_list, q):
    for (begin_pos, end_pos) in xy_list:
        hr = _calc_hit_rate(reuse_dist_array, cache_size, begin_pos, end_pos)
        # print('({},{}): {}'.format(begin_pos, end_pos, hr))
        q.put((begin_pos, end_pos, hr))


def _calc_hit_rate_multiprocess_ts(reuse_dist_array, cache_size, xy_list, q):
    for (i, j, begin_pos, end_pos) in xy_list:
        # print(i, j, begin_pos, end_pos)
        hr = _calc_hit_rate(reuse_dist_array, cache_size, begin_pos, end_pos)
        # print('({},{}): {}'.format(i, j, hr))
        q.put((i, j, hr))


def _cal_hit_rate_multiprocess_ts2_init(reuse_dist_array, cache_size, break_points, q):
    _calc_hit_rate_multiprocess_ts2.reuse_dist_array = reuse_dist_array
    _calc_hit_rate_multiprocess_ts2.cache_size = cache_size
    _calc_hit_rate_multiprocess_ts2.break_points = break_points
    _calc_hit_rate_multiprocess_ts2.q = q


def _calc_hit_rate_multiprocess_ts2(order):
    # break_points[i] points to the ith position in reuse_dist_array
    l = []
    total_hc = 0
    for i in range(order + 1, len(_calc_hit_rate_multiprocess_ts2.break_points)):
        hc = _calc_hit_count(_calc_hit_rate_multiprocess_ts2.reuse_dist_array,
                             _calc_hit_rate_multiprocess_ts2.cache_size,
                             _calc_hit_rate_multiprocess_ts2.break_points[i - 1],
                             _calc_hit_rate_multiprocess_ts2.break_points[i],
                             _calc_hit_rate_multiprocess_ts2.break_points[order])
        total_hc += hc
        hr = total_hc / (
        _calc_hit_rate_multiprocess_ts2.break_points[i] - _calc_hit_rate_multiprocess_ts2.break_points[order])
        l.append((order, i, hr))
    # _calc_hit_rate_multiprocess_ts2.q.put(1)
    return l


def draw(filename="heatmap.pdf"):
    with open('temp', 'rb') as ofile:
        result = pickle.load(ofile)
    print("load data successfully, begin plotting")
    # print(result)

    masked_array = np.ma.array(result, mask=np.isnan(result))

    # print(masked_array)
    cmap = plt.cm.jet
    cmap.set_bad('w', 1.)

    plt.Figure()
    heatmap = plt.pcolor(masked_array.T, cmap=cmap)

    # heatmap = plt.pcolor(result2.T, cmap=plt.cm.Blues, vmin=np.min(result2[np.nonzero(result2)]), vmax=result2.max())
    try:
        heatmap = plt.pcolor(masked_array.T, cmap=cmap)
        # heatmap = plt.pcolor(result2.T, cmap=plt.cm.Blues, vmin=np.min(result2[np.nonzero(result2)]), vmax=result2.maxrc())
        plt.show()
        plt.savefig(filename)
    except:
        import matplotlib
        matplotlib.use('pdf')
        heatmap = plt.pcolor(masked_array.T, cmap=cmap)
        plt.savefig(filename)


import os

print(os.getcwd())
# prepare_heatmap_dat(3000, 200)
# prepare_heatmap_dat_multiprocess(1000, 20000, 48)

prepare_heatmap_dat_multiprocess_ts("../data/trace_CloudPhysics_bin", 100000000, 20000, 4, True)
draw()

# for i in range(10, 200, 10):
#     prepare_heatmap_dat_multiprocess_ts("../data/traces/w02_vscsi1.vscsitrace", 1000000000, 200*i*i, 48, True)
#     draw("heatmap_"+str(1000000000)+'_'+str(200*i*i)+'.pdf')
