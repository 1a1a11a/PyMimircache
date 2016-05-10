"""
this module provides the heatmap ploting engine, it supports both virtual time (fixed number of trace requests) and
real time, under both modes, it support using multiprocessing to do the plotting
currently, it only support LRU
the mechanism it works is it gets reuse distance from LRU profiler (parda for now)
the whole module is not externally thread safe, if you want to use it in multi-threading,
create multiple heatmap objects
"""

import logging
import pickle
from multiprocessing import Pool

import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from mimircache.cache.LRU import LRU
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.profiler.pardaProfiler import pardaProfiler
from mimircache.utils.printing import *

DEBUG = True


class heatmap:
    def __init__(self, cache_class=LRU):
        self.cache_class = cache_class
        self.reuse_dist_python_list = []
        self.break_points = None

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

        if calculate:
            # build profiler for extracting reuse distance (For LRU)
            p = pardaProfiler(LRU, 30000, reader)
            # reuse distance, c_reuse_dist_long_array is an array in dtype C long type
            c_reuse_dist_long_array = p.get_reuse_distance()

            for i in c_reuse_dist_long_array:
                self.reuse_dist_python_list.append(i)

            if not os.path.exists('temp/'):
                os.makedirs('temp/')
            with open('temp/reuse.dat', 'wb') as ofile:
                pickle.dump(self.reuse_dist_python_list, ofile)

        else:
            # the reuse distance has already been calculated, just load it
            with open('temp/reuse.dat', 'rb') as ifile:
                self.reuse_dist_python_list = pickle.load(ifile)

            breakpoints_filename = 'temp/break_points_' + mode + str(self.interval) + '.dat'
            # check whether the break points distribution table has been calculated
            if os.path.exists(breakpoints_filename):
                with open(breakpoints_filename, 'rb') as ifile:
                    self.break_points = pickle.load(ifile)

        # new
        if not self.break_points:
            # the break points are not loaded, needs to calculate it
            if mode == 'r':
                self.break_points = self._get_breakpoints_realtime(reader, self.interval)
            elif mode == 'v':
                self.break_points = self._get_breakpoints_virtualtime(self.interval, len(self.reuse_dist_python_list))
            with open('temp/break_points_' + mode + str(self.interval) + '.dat', 'wb') as ifile:
                pickle.dump(self.break_points, ifile)

        # create the array for storing results
        # result is a dict: (x, y) -> heat, x, y is the left, lower point of heat square
        # (x,y) means from time x to time y
        array_len = len(self.break_points)
        result = np.empty((array_len, array_len), dtype=np.float32)
        result[:] = np.nan

        # Jason: efficiency can be further improved by porting into Cython and improve parallel logic
        # Jason: memory usage can be optimized by not copying whole reuse distance array in each sub process
        map_list = [i for i in range(len(self.break_points) - 1)]

        count = 0
        with Pool(processes=self.num_of_process) as p:
            for ret_list in p.imap_unordered(self._calc_hit_rate_subprocess_realtime, map_list):
                count += 1
                print("%2.2f%%" % (count / len(map_list) * 100), end='\r')
                for r in ret_list:  # l is a list of (x, y, hr)
                    result[r[0]][r[1]] = r[2]

        # print(result)
        with open('temp/draw', 'wb') as ofile:
            pickle.dump(result, ofile)

        return result

    def _get_breakpoints_virtualtime(self, bin_size, total_length):
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

    def _get_breakpoints_realtime(self, reader, interval):
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

    def _calc_hit_count(self, reuse_dist_array, cache_size, begin_pos, end_pos, real_start):
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

    def _calc_hit_rate_subprocess_realtime(self, order):
        """
        the child process for calculating hit rate, each child process will calculate for
        a column with fixed starting time
        :param order: the order of column the child is working on
        :return: a list of result in the form of (x, y, hit_rate) with x as fixed value
        """
        result_list = []
        total_hc = 0
        for i in range(order + 1, len(self.break_points)):
            # break points does not
            hc = self._calc_hit_count(self.reuse_dist_python_list, self.cache_size, self.break_points[i - 1],
                                      self.break_points[i], self.break_points[order])
            total_hc += hc
            hr = total_hc / (self.break_points[i] - self.break_points[order])
            result_list.append((order, i, hr))
        return result_list

    def draw(self, xydict, filename="heatmap.png"):
        # with open('temp/draw', 'rb') as ofile:
        #     xydict = pickle.load(ofile)
        # print("load data successfully, begin plotting")
        # print(result)

        masked_array = np.ma.array(xydict, mask=np.isnan(xydict))

        # print(masked_array)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        plt.Figure()
        plt.pcolor(masked_array.T, cmap=cmap)

        # heatmap = plt.pcolor(result2.T, cmap=plt.cm.Blues, vmin=np.min(result2[np.nonzero(result2)]), \
        # vmax=result2.max())
        try:
            plt.pcolor(masked_array.T, cmap=cmap)
            # plt.show()
            plt.savefig(filename)
        except Exception as e:
            logging.warning(str(e))
            import matplotlib
            matplotlib.use('pdf')
            plt.pcolor(masked_array.T, cmap=cmap)
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
        self.cache_size = cache_size
        self.interval = interval
        # self.reader = reader
        self.mode = mode
        figname = None
        if 'num_of_process' in kargs:
            self.num_of_process = kargs['num_of_process']
        else:
            self.num_of_process = 4

        if 'figname' in kargs:
            figname = kargs['figname']

        if mode == 'r':
            xydict = self.prepare_heatmap_dat('r', reader, True)
        elif mode == 'v':
            xydict = self.prepare_heatmap_dat('v', reader, True)
            self.draw(xydict)
        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        if figname:
            self.draw(xydict, figname)
        else:
            self.draw(xydict)

        self.__del_manual__()


if __name__ == "__main__":

    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    # reader1 = plainCacheReader("../data/parda.trace")
    # reader2 = vscsiCacheReader("../data/trace_CloudPhysics_bin")


    hm = heatmap()
    for filename in os.listdir("../data/cloudphysics"):
        if filename.endswith('.vscsitrace'):
            mem_size = mem_sizes[int(filename.split('_')[0][1:])] * 16
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            print(filename)
            hm.run('r', 100000000, mem_size, reader, num_of_process=42, figname=filename + '.png')




    # hm.run('r', 100000000, 2000, reader2, 1)
            # hm.run('v', 1000, 2000, reader2, 1)


    # for i in range(10, 200, 10):
    #     prepare_heatmap_dat_multiprocess_ts("../data/traces/w02_vscsi1.vscsitrace", 1000000000, 200*i*i, 48, True)
    #     draw("heatmap_"+str(1000000000)+'_'+str(200*i*i)+'.pdf')
