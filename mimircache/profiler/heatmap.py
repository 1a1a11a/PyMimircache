"""
this module provides the heatmap ploting engine, it supports both virtual time (fixed number of trace requests) and
real time, under both modes, it support using multiprocessing to do the plotting
currently, it only support LRU
the mechanism it works is it gets reuse distance from LRU profiler (parda for now)
"""

import logging
import pickle
from collections import deque
from multiprocessing import Pool, Array, Process, Queue

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from mimircache.cache.LRU import LRU
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.profiler.pardaProfiler import pardaProfiler
import matplotlib.ticker as ticker
from mimircache.utils.printing import *

DEBUG = True

__all__ = ("run", "draw", "prepare_heatmap_dat")





def _calc_hit_count(reuse_dist_array, cache_size, begin_pos, end_pos, real_start):
    """

    :rtype: count of hit
    :param reuse_dist_array:
    :param cache_size:
    :param begin_pos:
    :param end_pos: end pos of trace in current partition (not included)
    :param real_start: the real start position of cache trace
    :return:
    """
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
    return hit_count


def _calc_hit_rate_subprocess(order, cache_size, break_points_share_array, reuse_dist_share_array, q):
    """
    the child process for calculating hit rate, each child process will calculate for
    a column with fixed starting time
    :param order: the order of column the child is working on
    :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
    """

    result_list = []
    total_hc = 0
    for i in range(order + 1, len(break_points_share_array)):
        hc = _calc_hit_count(reuse_dist_share_array, cache_size, break_points_share_array[i - 1],
                             break_points_share_array[i], break_points_share_array[order])
        total_hc += hc
        hr = total_hc / (break_points_share_array[i] - break_points_share_array[order])
        result_list.append((order, i, hr))
    q.put(result_list)
    # return result_list





class heatmap:
    def __init__(self, cache_class=LRU):
        self.cache_class = cache_class


    def prepare_heatmap_dat(self, mode, reader, calculate=True):
        """
        prepare the data for plotting heatmap
        :param mode:   mode can be either virtual time(v) or real time(r)
        :param reader: for reading the trace file
        :param calculate: flag for checking whether we need to recalculate reuse distance and distribution table
        :return:
        """

        assert isinstance(reader, vscsiCacheReader), "Currently only supports vscsiReader"
        reader.reset()

        reuse_dist_python_list = []
        break_points = []
        
        if calculate:
            # build profiler for extracting reuse distance (For LRU)
            p = pardaProfiler(30000, reader)
            # reuse distance, c_reuse_dist_long_array is an array in dtype C long type
            c_reuse_dist_long_array = p.get_reuse_distance()

            for i in c_reuse_dist_long_array:
                reuse_dist_python_list.append(i)

            if not os.path.exists('temp/'):
                os.makedirs('temp/')
            with open('temp/reuse.dat', 'wb') as ofile:
                pickle.dump(reuse_dist_python_list, ofile)

        else:
            # the reuse distance has already been calculated, just load it
            with open('temp/reuse.dat', 'rb') as ifile:
                reuse_dist_python_list = pickle.load(ifile)

            breakpoints_filename = 'temp/break_points_' + mode + str(self.interval) + '.dat'
            # check whether the break points distribution table has been calculated
            if os.path.exists(breakpoints_filename):
                with open(breakpoints_filename, 'rb') as ifile:
                    break_points = pickle.load(ifile)

        # check break points are loaded or not, if not need to calculate it
        if not break_points:
            if mode == 'r':
                break_points = self._get_breakpoints_realtime(reader, self.interval)
            elif mode == 'v':
                break_points = self._get_breakpoints_virtualtime(self.interval, len(reuse_dist_python_list))
            with open('temp/break_points_' + mode + str(self.interval) + '.dat', 'wb') as ifile:
                pickle.dump(break_points, ifile)

        # create share memory for child process
        reuse_dist_share_array = Array('l', len(reuse_dist_python_list), lock=False)
        for i, j in enumerate(reuse_dist_python_list):
            reuse_dist_share_array[i] = j

        break_points_share_array = Array('i', len(break_points), lock=False)
        for i, j in enumerate(break_points):
            break_points_share_array[i] = j


        # create the array for storing results
        # result is a dict: (x, y) -> heat, x, y is the left, lower point of heat square
        # (x,y) means from time x to time y
        array_len = len(break_points)
        result = np.empty((array_len, array_len), dtype=np.float32)
        result[:] = np.nan

        # Jason: efficiency can be further improved by porting into Cython and improve parallel logic
        # Jason: memory usage can be optimized by not copying whole reuse distance array in each sub process
        map_list = deque()
        for i in range(len(break_points) - 1):
            map_list.append(i)

        # new 0510
        q = Queue()
        process_pool = []
        process_count = 0
        result_count = 0
        map_list_pos = 0
        while result_count < len(map_list):
            if process_count < self.num_of_process and map_list_pos < len(map_list):
                p = Process(target=_calc_hit_rate_subprocess, args=(map_list[map_list_pos], self.cache_size,
                                                                    break_points_share_array, reuse_dist_share_array,
                                                                    q))
                p.start()
                process_pool.append(p)
                process_count += 1
                map_list_pos += 1
            else:
                rl = q.get()
                for r in rl:
                    result[r[0]][r[1]] = r[2]
                process_count -= 1
                result_count += 1
            print("%2.2f%%" % (result_count / len(map_list) * 100), end='\r')
        for p in process_pool:
            p.join()
        del reuse_dist_share_array
        del break_points_share_array

        # old 0510
        # with Pool(processes=self.num_of_process) as p:
        #     for ret_list in p.imap_unordered(_calc_hit_rate_subprocess, map_list,
        #                                      chunksize=10):
        #         count += 1
        #         print("%2.2f%%" % (count / len(map_list) * 100), end='\r')
        #         for r in ret_list:  # l is a list of (x, y, hr)
        #             result[r[0]][r[1]] = r[2]

        # print(result)
        with open('temp/draw', 'wb') as ofile:
            pickle.dump(result, ofile)

        return result

    @staticmethod
    def _get_breakpoints_virtualtime(bin_size, total_length):
        """

        :param bin_size: the size of the group (virtual time slice)
        :param total_length: the total length of trace requests
        :return: a list of break points
        """
        break_points = []
        for i in range(total_length // bin_size):
            break_points.append(i * bin_size)
        if break_points[-1] != total_length - 1:
            break_points.append(total_length - 1)

        if (len(break_points)) > 10000:
            colorfulPrintWithBackground('yellow', 'blue', "number of pixels in one dimension are more than 10000, \
            exact size: %d, it may take a very long time, if you didn't intend to do it, \
            please try with a larger time stamp" % len(break_points))

        return break_points

    @staticmethod
    def _get_breakpoints_realtime(reader, interval):
        """
        given reader(vscsi) and time interval, split the data according to the time interval,
        save the order into break_points(list)
        :param reader:
        :param interval:
        :return:
        """
        reader.reset()
        break_points = []
        prev = 0
        for num, line in enumerate(reader.lines()):
            if (line[0] - prev) > interval:
                break_points.append(num)
                prev = line[0]

        # noinspection PyUnboundLocalVariable
        if line[0] != prev:
            # noinspection PyUnboundLocalVariable
            break_points.append(num)

        if (len(break_points)) > 10000:
            colorfulPrintWithBackground('yellow', 'blue', "number of pixels in one dimension are more than 10000, \
            exact size: %d, it may take a very long time, if you didn't intend to do it, \
            please try with a larger time stamp" % len(break_points))

        return break_points

    @staticmethod
    def _calc_hit_rate(reuse_dist_array, cache_size, begin_pos, end_pos):
        """
        the original function for calculating hit rate, given the reuse_dist_array, cache_size,
        begin position and end position in the reuse_dist_array, return hit rate
        now isolate as a separate function
        :param reuse_dist_array:
        :param cache_size:
        :param begin_pos:
        :param end_pos:
        :return:
        """
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
        # print("hit+miss={}, total size:{}, hit rage:{}".format(hit_count+miss_count, \
        # end_pos-begin_pos, hit_count/(end_pos-begin_pos)))
        return hit_count / (end_pos - begin_pos)


    @staticmethod
    def draw(xydict, **kargs):
        if 'figname' in kargs:
            filename = kargs['figname']
        else:
            filename = 'heatmap.png'

        plt.clf()
        masked_array = np.ma.array(xydict, mask=np.isnan(xydict))

        # print(masked_array)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        # heatmap = plt.pcolor(result2.T, cmap=plt.cm.Blues, vmin=np.min(result2[np.nonzero(result2)]), \
        # vmax=result2.max())
        try:
            # ax.pcolor(masked_array.T, cmap=cmap)
            if 'fixed_range' in kargs and kargs['fixed_range']:
                img = plt.imshow(masked_array.T, vmin=0, vmax=1, interpolation='nearest', origin='lower')
            else:
                img = plt.imshow(masked_array.T, interpolation='nearest', origin='lower')
            cb = plt.colorbar(img)
            if 'text' in kargs:
                (length1, length2) = masked_array.shape
                ax = plt.gca()
                ax.text(length2 // 3, length1 // 8, kargs['text'], fontsize=20)  # , color='blue')

            # change tick from arbitrary number to real time
            if 'change_label' in kargs and 'interval' in kargs:
                ticks = ticker.FuncFormatter(lambda x, pos: '{:2.2f}'.format(x * kargs['interval'] / (10 ** 6) / 3600))
                plt.gca().xaxis.set_major_formatter(ticks)
                plt.gca().yaxis.set_major_formatter(ticks)
                plt.xlabel("time/hour")
                plt.ylabel("time/hour")

            # plt.show()
            plt.savefig(filename)
            colorfulPrint("red", "plot is saved at the same directory")
        except Exception as e:
            logging.warning(str(e))
            plt.pcolormesh(masked_array.T, cmap=cmap)
            plt.savefig(filename)

    def __del_manual__(self):
        """
        cleaning
        :return:
        """
        if os.path.exists('temp/'):
            for filename in os.listdir('temp/'):
                os.remove('temp/' + filename)
            os.rmdir('temp/')

    def run(self, mode, interval, cache_size, reader, **kargs):
        """

        :param mode:
        :param interval:
        :param cache_size:
        :param reader:
        :param kargs: include num_of_process, figname
        :return:
        """
        self.cache_size = cache_size
        self.interval = interval
        self.mode = mode
        if 'num_of_process' in kargs:
            self.num_of_process = kargs['num_of_process']
        else:
            self.num_of_process = 4


        if 'calculate' in kargs:
            calculate = kargs['calculate']
        else:
            calculate = True

        if mode == 'r':
            xydict = self.prepare_heatmap_dat('r', reader, calculate)
            kargs['interval'] = interval
        elif mode == 'v':
            xydict = self.prepare_heatmap_dat('v', reader, calculate)
        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        self.draw(xydict, **kargs)

        # self.__del_manual__()


def server_plot_all():
    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    for filename in os.listdir("../data/cloudphysics"):
        print(filename)
        if filename.endswith('.vscsitrace'):
            if int(filename.split('_')[0][1:]) in [1, 4, 5, 51, 99, 83, 87]:
                continue
            if os.path.exists(filename + '_v.png'):
                continue
            hm = heatmap()
            mem_size = mem_sizes[int(filename.split('_')[0][1:]) - 1] * 16
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            hm.run('r', 1000000000, mem_size, reader, num_of_process=45, figname=filename + '_v.png',
                   fixed_range="True", change_label='True')


def server_size_plot():
    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    for filename in os.listdir("../data/cloudphysics"):
        if filename.endswith('.vscsitrace'):
            if int(filename.split('_')[0][1:]) in [1, 4, 5, 51, 99, 83, 87]:
                continue
            if os.path.exists('time_size/' + filename + '_512.png'):
                continue
            hm = heatmap()
            mem_size = mem_sizes[int(filename.split('_')[0][1:])]
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            print(filename)
            size = 512
            while size < mem_size * 1024 * 128:
                print(size)
                if size == 512:
                    hm.run('r', 1000000000, size, reader, num_of_process=42, figname='time_size/' + filename
                                                                                     + '_' + str(size) + '.png',
                           calculate=True, fixed_range=True, text='size=' + str(size))
                else:
                    hm.run('r', 1000000000, size, reader, num_of_process=42, figname='time_size/' + filename
                                                                                     + '_' + str(size) + '.png',
                           calculate=False, fixed_range=True, text='size=' + str(size))
                size *= 2


def server_request_num_plot():
    for filename in os.listdir("../data/cloudphysics"):
        print(filename)
        if filename.endswith('.vscsitrace'):
            if int(filename.split('_')[0][1:]) in [1, 4, 5, 51, 99, 83, 87]:
                continue
            if os.path.exists(filename + '_request_num.png'):
                continue
            hm = heatmap()
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            break_points = hm._get_breakpoints_realtime(reader, 1000000000)

            array_len = len(break_points)
            result = np.empty((array_len, array_len), dtype=np.float32)
            result[:] = np.nan

            for o1, i in enumerate(break_points):
                for o2, j in enumerate(break_points):
                    if i >= j:
                        continue
                    else:
                        result[o1][o2] = j - i
            hm.draw(result, figname=filename + '_request_num.png')


def localtest():
    # reader1 = plainCacheReader("../data/parda.trace")
    reader2 = vscsiCacheReader("../data/trace_CloudPhysics_bin")

    hm = heatmap()
    hm.run('r', 10000000, 2000, reader2, num_of_process=8, fixed_range="True", text="Hello word", change_label='True')





if __name__ == "__main__":
    localtest()
    # server_plot_all()
    # server_size_plot()
    # server_request_num_plot()
