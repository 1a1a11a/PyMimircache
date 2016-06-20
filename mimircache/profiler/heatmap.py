"""
this module provides the heatmap ploting engine, it supports both virtual time (fixed number of trace requests) and
real time, under both modes, it support using multiprocessing to do the plotting
currently, it only support LRU
the mechanism it works is it gets reuse distance from LRU profiler (parda for now)
currently supported plot type:

        # 1. hit_rate_start_time_end_time
        # 2. hit_rate_start_time_cache_size
        # 3. avg_rd_start_time_end_time
        # 4. cold_miss_count_start_time_end_time
        # 5. rd_distribution
"""

import logging
import os
import pickle
from collections import deque
from multiprocessing import Array, Process, Queue

import matplotlib
import matplotlib.ticker as ticker
import mimircache.c_heatmap as c_heatmap
import numpy as np
from matplotlib import pyplot as plt

from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.oldModule.pardaProfiler import pardaProfiler
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.heatmap_subprocess import *
from mimircache.profiler.heatmap_subprocess2d import *
from mimircache.utils.printing import *

DEBUG = True

__all__ = ("run", "draw", "prepare_heatmap_dat")


# heatmap gradient, only works in _calc_rd_count_subprocess
# HEATMAP_GRADIENT = 10000
# LOG_NUM = 0




class heatmap:
    def __init__(self, cache_class=LRU):
        self.cache_class = cache_class
        if not os.path.exists('temp/'):
            os.mkdir('temp')

    def _prepare_reuse_distance_and_break_points(self, mode, reader, calculate, save=False):
        # assert isinstance(reader, vscsiCacheReader), "Currently only supports vscsiReader"
        reader.reset()

        break_points = []

        if calculate:
            # build profiler for extracting reuse distance (For LRU)
            # p = pardaProfiler(30000, reader)
            p = LRUProfiler(30000, reader)
            # reuse distance, c_reuse_dist_long_array is an array in dtype C long type
            # c_reuse_dist_long_array = p.get_reuse_distance()
            reuse_dist = p.get_reuse_distance()

            if save:
                # needs to save the data
                # for i in c_reuse_dist_long_array:
                #     reuse_dist_python_list.append(i)

                if not os.path.exists('temp/'):
                    os.makedirs('temp/')
                with open('temp/reuse.dat', 'wb') as ofile:
                    # pickle.dump(reuse_dist_python_list, ofile)
                    pickle.dump(reuse_dist, ofile)

        else:
            # the reuse distance has already been calculated, just load it
            with open('temp/reuse.dat', 'rb') as ifile:
                # reuse_dist_python_list = pickle.load(ifile)
                reuse_dist = pickle.load(ifile)

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
                break_points = self._get_breakpoints_virtualtime(self.interval, reader.num_of_line)
            if save:
                with open('temp/break_points_' + mode + str(self.interval) + '.dat', 'wb') as ifile:
                    pickle.dump(break_points, ifile)

        return reuse_dist, break_points

        # if reuse_dist_python_list:
        #     return reuse_dist_python_list, break_points
        # else:
        #     return c_reuse_dist_long_array, break_points

    def _prepare_multiprocess_params_LRU(self, plot_type, break_points, **kwargs):
        # 1. hit_rate_start_time_end_time
        # 2. hit_rate_start_time_cache_size
        # 3. avg_rd_start_time_end_time
        # 4. cold_miss_count_start_time_end_time
        # 5. rd_distribution

        # create the array for storing results
        # result is a dict: (x, y) -> heat, x, y is the left, lower point of heat square
        kwargs_dict = {}  # the dictionary for passing parameters to subprocess
        if plot_type == "hit_rate_start_time_cache_size":
            max_rd = max(kwargs['reuse_distance_list'])
            kwargs_dict['max_rd'] = max_rd
            if 'bin_size' in kwargs:
                kwargs_dict["bin_size"] = kwargs['bin_size']
            else:
                kwargs_dict['bin_size'] = 1

            result = np.empty((len(break_points), max_rd // kwargs_dict['bin_size'] + 1), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_hit_rate_start_time_cache_size_subprocess
            kwargs_dict['yticks'] = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(kwargs_dict['bin_size'] * x))
            kwargs_dict['xlabel'] = 'time'
            kwargs_dict['ylabel'] = 'cache size'
            kwargs_dict['zlabel'] = 'hit rate'
            plt.gca().yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(kwargs_dict['bin_size'] * x)))

        elif plot_type == "rd_distribution":
            max_rd = max(kwargs['reuse_distance_list'])
            # print("max rd = %d"%max_rd)
            kwargs_dict['max_rd'] = max_rd
            kwargs_dict['log_num'] = self._cal_log_num(max_rd, len(break_points))
            plt.xlabel("time")
            plt.ylabel('reuse distance')
            plt.gca().yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(kwargs_dict['log_num'] ** x)))

            result = np.empty((len(break_points), int(math.log(max_rd, kwargs_dict['log_num'])) + 1), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_rd_distribution_subprocess

        elif plot_type == "hit_rate_start_time_end_time":
            # (x,y) means from time x to time y
            reader = kwargs['reader']
            kwargs_dict['cache_size'] = self.cache_size
            array_len = len(break_points)
            result = np.empty((array_len, array_len), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_hit_rate_start_time_end_time_subprocess

            kwargs_dict['yticks'] = ticker.FuncFormatter(
                lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))

            last_access = c_heatmap.get_last_access_dist(reader.cReader)
            last_access_array = Array('l', len(last_access), lock=False)
            for i, j in enumerate(last_access):
                last_access_array[i] = j
            kwargs_dict['last_access_array'] = last_access_array


        elif plot_type == "avg_rd_start_time_end_time":
            # (x,y) means from time x to time y
            kwargs_dict['cache_size'] = self.cache_size

            array_len = len(break_points)
            result = np.empty((array_len, array_len), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_avg_rd_start_time_end_time_subprocess

        elif plot_type == "cold_miss_count_start_time_end_time":
            # (x,y) means from time x to time y
            kwargs_dict['cache_size'] = self.cache_size
            array_len = len(break_points)
            result = np.empty((array_len, array_len), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_cold_miss_count_start_time_end_time_subprocess

        else:
            raise RuntimeError("cannot recognize plot type")

        kwargs_dict['xticks'] = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))

        return result, kwargs_dict, func_pointer

    def calculate_heatmap_dat(self, mode, reader, plot_type, **kwargs):
        """
            calculate the data for plotting heatmap

        :param mode: mode can be either virtual time(v) or real time(r)
        :param reader: for reading the trace file
        :param plot_type: the type of heatmap to generate, possible choices are:
        1. hit_rate_start_time_end_time
        2. hit_rate_start_time_cache_size
        3. avg_rd_start_time_end_time
        4. cold_miss_count_start_time_end_time
        5. rd_distribution
        6.

        :param kwargs: possible key-arg includes the type of cache replacement algorithms(cache),

        :return: a two-dimension list, the first dimension is x, the second dimension is y, the value is the heat value
        """
        if 'calculate' in kwargs:
            calculate = kwargs['calculate']
        else:
            calculate = True

        if 'LRU' not in kwargs or ('LRU' in kwargs and kwargs['LRU']):
            # reuse_dist_python_list, break_points = self._prepare_reuse_distance_and_break_points(mode, reader, calculate)
            reuse_dist, break_points = self._prepare_reuse_distance_and_break_points(mode, reader, calculate)

            # create shared memory for child process
            reuse_dist_share_array = Array('l', len(reuse_dist), lock=False)
            for i, j in enumerate(reuse_dist):
                reuse_dist_share_array[i] = j
            kwargs['reuse_distance_list'] = reuse_dist_share_array

            kwargs['reader'] = reader
            result, kwargs_dict, func_pointer = self._prepare_multiprocess_params_LRU(plot_type, break_points,
                                                                                      **kwargs)  # reuse_distance_list=reuse_dist_share_array,

        else:
            # if it is not LRU, then there should be a cache replacement algorithm in kwargs
            assert 'cache' in kwargs, "you didn't provide any cache replacement algorithm"
            cache_type = kwargs['cache']
            cache_size = kwargs['cache_size']

            # for general cache replacement algorithm, uncomment the following, it is just for
            # transform the data from other format to plain txt
            # p = pardaProfiler(30000, reader)
            p = LRUProfiler(cache_size, reader)

            # prepare break points
            if mode == 'r':
                break_points = self._get_breakpoints_realtime(reader, self.interval)
            elif mode == 'v':
                break_points = self._get_breakpoints_virtualtime(self.interval, p.num_of_lines)
            else:
                raise RuntimeError("unrecognized mode, it can only be r or v")

            kwargs_dict = {}
            array_len = len(break_points)
            result = np.empty((array_len, array_len), dtype=np.float32)
            result[:] = np.nan

        # print(break_points)

        # create shared memory for break points
        break_points_share_array = Array('i', len(break_points), lock=False)
        for i, j in enumerate(break_points):
            break_points_share_array[i] = j

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
                # LRU related heatmaps
                if 'LRU' not in kwargs or ('LRU' in kwargs and kwargs['LRU']):
                    # p = Process(target=_calc_hit_rate_subprocess, args=(map_list[map_list_pos], self.cache_size,
                    #                                                     break_points_share_array, reuse_dist_share_array,
                    #                                                     q))
                    # print(kwargs_dict)
                    p = Process(target=func_pointer, args=(map_list[map_list_pos], break_points_share_array,
                                                           reuse_dist, q), kwargs=kwargs_dict)
                else:
                    p = Process(target=calc_hit_rate_start_time_end_time_subprocess_general,
                                args=(map_list[map_list_pos], cache_type,
                                      break_points_share_array, reader, q), kwargs={'cache_size': cache_size})

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

        return result, kwargs_dict

    def _cal_log_num(self, max_rd, length):
        """
        find out the LOG_NUM that makes the number of pixels in y axis to match the number of pixels in x axis
        :param max_rd: maxinum reuse distance
        :param length: length of break points (also number of pixels along x axis)
        :return:
        """
        init_log_num = 10
        l2_prev = length
        while True:
            l2 = math.log(max_rd, init_log_num)
            if l2 > length > l2_prev:
                return init_log_num
            l2_prev = l2
            init_log_num = (init_log_num - 1) / 2 + 1

    @staticmethod
    def _get_breakpoints_virtualtime(bin_size, total_length):
        """

        :param bin_size: the size of the group (virtual time slice)
        :param total_length: the total length of trace requests
        :return: a list of break points
        """
        break_points = []
        assert total_length // bin_size > 10, "bin size too large, please choose a smaller one, " \
                                              "total length %d, bin size %d" % (total_length, bin_size)
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
            if (float(line[0]) - prev) > interval:
                break_points.append(num)
                prev = float(line[0])

        # noinspection PyUnboundLocalVariable
        if line[0] != prev:
            # noinspection PyUnboundLocalVariable
            break_points.append(num)

        if (len(break_points)) > 10000:
            colorfulPrintWithBackground('yellow', 'blue', "number of pixels in one dimension are more than 10000, \
            exact size: %d, it may take a very long time, if you didn't intend to do it, \
            please try with a larger time stamp" % len(break_points))

        # print(break_points)
        return break_points

    @staticmethod
    def _calc_hit_rate(reuse_dist_array, last_access_dist_array, cache_size, begin_pos, end_pos):
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
            if last_access_dist_array[i] - (i - begin_pos) <= 0 and reuse_dist_array[i] < cache_size:
                # hit
                hit_count += 1
                print("{}: {}".format(i, reuse_dist_array[i]))
            else:
                # miss
                miss_count += 1
        print("hit={}, hit+miss={}, total size:{}, hit rate:{}".format(hit_count, hit_count + miss_count,
                                                                       end_pos - begin_pos,
                                                                       hit_count / (end_pos - begin_pos)))
        return hit_count / (end_pos - begin_pos)

    @staticmethod
    def draw_heatmap(xydict, plot_type, **kwargs):
        if 'figname' in kwargs:
            filename = kwargs['figname']
        else:
            filename = 'heatmap.png'

        masked_array = np.ma.array(xydict, mask=np.isnan(xydict))

        # print(masked_array)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        try:
            # ax.pcolor(masked_array.T, cmap=cmap)
            if plot_type == "rd_distribution":  # or plot_type == "cold_miss_count_start_time_end_time"
                img = plt.imshow(masked_array.T, interpolation='nearest', origin='lower', aspect='auto'
                                 , norm=matplotlib.colors.LogNorm())
            else:
                if 'fixed_range' in kwargs and kwargs['fixed_range']:
                    img = plt.imshow(masked_array.T, vmin=0, vmax=1, interpolation='nearest', origin='lower',
                                     aspect='auto')
                else:
                    img = plt.imshow(masked_array.T, interpolation='nearest', origin='lower', aspect='auto')

            cb = plt.colorbar(img)
            if 'text' in kwargs:
                (length1, length2) = masked_array.shape
                ax = plt.gca()
                ax.text(length2 // 3, length1 // 8, kwargs['text'], fontsize=20)  # , color='blue')

            if 'xlabel' in kwargs:
                plt.xlabel(kwargs['xlabel'])
            if 'ylabel' in kwargs:
                plt.ylabel(kwargs['ylabel'])
            if 'xticks' in kwargs:
                plt.gca().xaxis.set_major_formatter(kwargs['xticks'])
            if 'yticks' in kwargs:
                plt.gca().yaxis.set_major_formatter(kwargs['yticks'])

            # # change tick from arbitrary number to real time
            if 'change_label' in kwargs and kwargs['change_label'] and 'interval' in kwargs:
                ticks = ticker.FuncFormatter(lambda x, pos: '{:2.2f}'.format(x * kwargs['interval'] / (10 ** 6) / 3600))
                plt.gca().xaxis.set_major_formatter(ticks)
                plt.gca().yaxis.set_major_formatter(ticks)
                plt.xlabel("time/hour")
                plt.ylabel("time/hour")

            # plt.show()
            plt.savefig(filename, dpi=600)
            colorfulPrint("red", "plot is saved at the same directory")
            plt.clf()
        except Exception as e:
            logging.warning(str(e))
            plt.pcolormesh(masked_array.T, cmap=cmap)
            plt.savefig(filename)

    @staticmethod
    def draw2d(l, **kwargs):
        if 'figname' in kwargs:
            filename = kwargs['figname']
        else:
            filename = '2d_plot.png'

        try:
            # print(l)
            plt.plot(l)

            if 'xlabel' in kwargs:
                plt.xlabel(kwargs['xlabel'])
            if 'ylabel' in kwargs:
                plt.ylabel(kwargs['ylabel'])
            if 'xticks' in kwargs:
                plt.gca().xaxis.set_major_formatter(kwargs['xticks'])
            if 'yticks' in kwargs:
                plt.gca().yaxis.set_major_formatter(kwargs['yticks'])

            # plt.show()
            plt.savefig(filename)
            colorfulPrint("red", "plot is saved at the same directory")
            plt.clf()
        except Exception as e:
            logging.warning(str(e))
            plt.savefig(filename)

    @staticmethod
    def draw_3D(xydict, **kwargs):
        from mpl_toolkits.mplot3d import Axes3D

        if 'figname' in kwargs:
            filename = kwargs['figname']
        else:
            filename = 'heatmap_3D.png'

        fig = plt.figure()
        ax = Axes3D(fig)
        # X = np.arange(-4, 4, 0.25)
        X = np.arange(0, len(xydict))
        # Y = np.arange(-4, 4, 0.25)
        Y = np.arange(0, len(xydict[0]))
        X, Y = np.meshgrid(X, Y)
        # R = np.sqrt(X ** 2 + Y ** 2)
        # Z = np.sin(R)
        Z = np.array(xydict).T

        print(X.shape)
        print(Y.shape)
        print(Z.shape)

        # print(X)
        # print(Y)
        # print(Z)

        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        if 'xticks' in kwargs:
            print("setx")
            ax.xaxis.set_major_formatter(kwargs['xticks'])
        if 'yticks' in kwargs:
            print("sety")
            ax.yaxis.set_major_formatter(kwargs['yticks'])

        ax.plot_surface(X[:, :-3], Y[:, :-3], Z[:, :-3], rstride=10000, cstride=2, cmap=plt.cm.jet)  # plt.cm.hot
        ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=plt.cm.jet)
        ax.set_zlim(0, 1)
        # ax.set_zlim(-2, 2)

        plt.show()
        plt.savefig(filename, dpi=600)

    def __del_manual__(self):
        """
        cleaning
        :return:
        """
        if os.path.exists('temp/'):
            for filename in os.listdir('temp/'):
                os.remove('temp/' + filename)
            os.rmdir('temp/')

    def run(self, mode, interval, plot_type, reader, **kwargs):
        """

        :param plot_type:
        :param mode:
        :param interval:
        :param reader:
        :param kwargs: include num_of_process, figname
        :return:
        """
        self.cache_size = kwargs['cache_size']
        self.interval = interval
        self.mode = mode
        reader.reset()
        if 'num_of_process' in kwargs:
            self.num_of_process = kwargs['num_of_process']
        else:
            self.num_of_process = 4

        if mode == 'r' or mode == 'v':
            xydict, kwargs_dict = self.calculate_heatmap_dat(mode, reader, plot_type, interval=interval, **kwargs)
        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        kwargs_dict.update(kwargs)
        self.draw_heatmap(xydict, plot_type, interval=interval, **kwargs_dict)
        reader.reset()
        # self.draw_test(xydict, **kwargs_dict)

        # self.__del_manual__()

    def draw_diff(self, xydict1, xydict2, plot_type, **kwargs):

        # print(xydict1.shape)
        # print(xydict2.shape)
        print(xydict1)
        print(xydict2)

        xydict = xydict1 - xydict2

        for i in range(len(xydict)):
            for j in range(len(xydict[i])):
                if xydict[i][j] < 0:
                    print("({}, {})".format(i, j))

        self.draw_heatmap(xydict, plot_type, **kwargs)

    def run_diff_optimal(self, mode, interval, plot_type, reader, **kwargs):
        """

        :param plot_type:
        :param mode:
        :param interval:
        :param reader:
        :param kwargs: include num_of_process, figname
        :return:
        """
        self.cache_size = kwargs['cache_size']
        self.interval = interval
        self.mode = mode
        if 'num_of_process' in kwargs:
            self.num_of_process = kwargs['num_of_process']
        else:
            self.num_of_process = 4

        if mode == 'r':
            xydict1, kwargs_dict1 = self.calculate_heatmap_dat('r', reader, plot_type, LRU=False, cache='optimal',
                                                               num_of_process=self.num_of_process,
                                                               cache_size=kwargs['cache_size'])

            reader.reset()
            xydict2, kwargs_dict2 = self.calculate_heatmap_dat('r', reader, plot_type, **kwargs)

        elif mode == 'v':
            xydict1, kwargs_dict1 = self.calculate_heatmap_dat('v', reader, plot_type, LRU=False, cache='optimal',
                                                               num_of_process=self.num_of_process,
                                                               cache_size=kwargs['cache_size'])
            reader.reset()
            xydict2, kwargs_dict2 = self.calculate_heatmap_dat('v', reader, plot_type, **kwargs)
        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)
        kwargs_dict = {}
        kwargs_dict.update(kwargs_dict1)
        kwargs_dict.update(kwargs_dict2)
        kwargs_dict.update(kwargs)
        self.draw_diff(xydict1, xydict2, plot_type, **kwargs_dict)

        # self.__del_manual__()


def server_plot_all():
    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    for filename in os.listdir("../data/cloudphysics"):
        print(filename)
        if filename.endswith('.vscsitrace'):
            # if int(filename.split('_')[0][1:]) in [1, 3, 4, 5, 51, 99, 83, 87]:
            #     continue

            hm = heatmap()
            mem_size = mem_sizes[int(filename.split('_')[0][1:]) - 1] * 16
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            if os.path.exists("0601/" + filename + "LRU_Optimal_" + str(mem_size) + "_r.png"):
                continue

            # cold_miss_2d(reader, 'r', 10000000, figname='0601/' + filename + '_cold_miss_r.png')
            # request_num_2d(reader, 'r', 10000000, figname='0601/' + filename + '_request_num_r.png')
            #
            # hm.run('r', 1000000000, "hit_rate_start_time_cache_size", reader, num_of_process=48, cache_size=mem_size,
            #        save=True, bin_size=1000, text="size = " + str(mem_size), fixed_range=True,
            #        figname='0601/' + filename + '_hit_rate_start_time_cache_size_r.png')
            # # ,change_label='False'     ,   10000000
            #
            #
            # hm.run('r', 1000000000, "avg_rd_start_time_end_time", reader, num_of_process=48,
            #        # LRU=False, cache='optimal',
            #        calculated=True, cache_size=mem_size,
            #        figname='0601/' + filename + "_avg_rd_start_time_end_time_r.png")
            #
            # hm.run('r', 1000000000, "rd_distribution", reader, num_of_process=48,  # LRU=False, cache='optimal',
            #        calculated=True, cache_size=mem_size,
            #        figname='0601/' + filename + "_rd_distribution_r.png")
            #
            # hm.run('r', 1000000000, "cold_miss_count_start_time_end_time", reader, num_of_process=48,
            #        calculated=True, cache_size=mem_size,
            #        figname='0601/' + filename + "_cold_miss_count_start_time_end_time_r.png")
            #
            # hm.run('r', 1000000000, "hit_rate_start_time_end_time", reader, num_of_process=48,
            #        cache_size=mem_size, calculated=True, fixed_range=True, change_label=True,
            #        figname='0601/' + filename + "_hit_rate_start_time_end_time_" + str(mem_size) + "_r.png")
            # reader.reset()


            hm.run_diff_optimal('r', 1000000000, "hit_rate_start_time_end_time", reader, num_of_process=48,
                                cache_size=mem_size, save=False, change_label=True,
                                figname='0601/' + filename + "LRU_Optimal_" + str(mem_size) + "_r.png")

            del hm
            del reader


def server_size_plot():
    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    for filename in os.listdir("../data/cloudphysics"):
        if filename.endswith('.vscsitrace'):
            if int(filename.split('_')[0][1:]) in [1, 4, 5, 51, 99, 83, 87]:
                continue
            if os.path.exists('time_size_0523/' + filename + '_128.png'):
                continue
            hm = heatmap()
            mem_size = mem_sizes[int(filename.split('_')[0][1:]) - 1]
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            print(filename)
            size = 128
            while size < mem_size * 1024:
                print(size)
                if size == 128:
                    hm.run('r', 10000000, "hit_rate_start_time_end_time", reader, num_of_process=48,
                           figname='time_size/' + filename + '_' + str(size) + '.png',
                           cache_size=size, change_label=True, save=True,
                           calculate=True, fixed_range=True, text='size=' + str(size))
                else:
                    hm.run('r', 10000000, "hit_rate_start_time_end_time", reader, num_of_process=48,
                           figname='time_size/' + filename + '_' + str(size) + '.png',
                           cache_size=size, change_label=True, save=True,
                           calculate=False, fixed_range=True, text='size=' + str(size))
                size *= 2


def server_plot_all_redis():
    for filename in os.listdir("../data/redis/"):
        print(filename)
        if filename.endswith('.csv'):
            # print(reader)
            # for line in reader:
            #     print(line)
            #     break

            if filename in []:
                continue
            if os.path.exists(filename + '_r.png'):
                continue
            hm = heatmap()
            reader = csvCacheReader("../data/redis/" + filename, column=1)
            hm.run('r', 20, 2000, reader, num_of_process=48, figname=filename + '_r.png',
                   fixed_range="True")  # , change_label='True')


def request_num_2d(reader, mode, interval, **kwargs):
    # reader = vscsiCacheReader("../data/trace_CloudPhysics_bin")

    hm = heatmap()
    if mode == 'r':
        break_points = hm._get_breakpoints_realtime(reader, interval)
    elif mode == 'v':
        # p = pardaProfiler(2000, reader2)
        p = LRUProfiler(2000, reader)
        break_points = hm._get_breakpoints_virtualtime(interval, p.num_of_lines)

    l = []
    for i in range(len(break_points)):
        if i == 0:
            continue
        else:
            l.append(break_points[i] - break_points[i - 1])
    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    hm.draw2d(l, figname=kwargs['figname'], xticks=xticks)

    reader.reset()


def cold_miss_2d(reader, mode, interval, **kwargs):
    # reader2 = vscsiCacheReader("../data/trace_CloudPhysics_bin")

    hm = heatmap()
    break_points = None
    if mode == 'r':
        break_points = hm._get_breakpoints_realtime(reader, interval)
    elif mode == 'v':
        p = pardaProfiler(2000, reader)
        break_points = hm._get_breakpoints_virtualtime(interval,
                                                       reader.get_num_total_lines())  # p.num_of_trace_elements)

    l = calc_cold_miss_count(break_points, reader)
    xticks = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
    hm.draw2d(l, figname=kwargs['figname'], xticks=xticks)
    reader.reset()


def localtest():
    CACHE_SIZE = 2000
    # reader1 = plainCacheReader("../data/parda.trace")



    reader2 = vscsiCacheReader("../data/trace_CloudPhysics_bin")

    hm = heatmap()

    # cold_miss_2d('r', 10000000, filename='cold_miss.png')
    # request_num_2d('r', 10000000, filename='request_num.png')

    # hm.run('r', 100000000, "hit_rate_start_time_cache_size", reader2, num_of_process=8,  # LRU=False, cache='optimal',
    #        save=False, fixed_range=True, cache_size=CACHE_SIZE, bin_size=200,
    #        figname="0529/3D_small_LRU_hit_rate_start_time_cache_size.png")
    # reader2.reset()

    # hm.run('r', 10000000, "avg_rd_start_time_end_time", reader2, num_of_process=8,  # LRU=False, cache='optimal',
    #        save=False, cache_size=CACHE_SIZE,
    #        figname="0529/small_LRU_avg_rd_start_time_end_time.png")
    # reader2.reset()

    hm.run('r', 10000000, "rd_distribution", reader2, num_of_process=8,  # LRU=False, cache='optimal',
           save=False, cache_size=CACHE_SIZE,
           figname="small_LRU_rd_distribution1.png")
    reader2.reset()

    # hm.run('r', 10000000, "cold_miss_count_start_time_end_time", reader2, num_of_process=8,
    #        # LRU=False, cache='optimal',
    #        save=False, cache_size=CACHE_SIZE,
    #        figname="0529/small_LRU_cold_miss_count_start_time_end_time.png")
    # reader2.reset()


    # hm.run('r', 10000000, "hit_rate_start_time_end_time", reader2, num_of_process=8,  # LRU=False, cache='optimal',
    #        cache_size=CACHE_SIZE, save=False, fixed_range=True, change_label=True,
    #        figname="small_LRU_hit_rate_start_time_end_time_" + str(CACHE_SIZE) + ".png")


    reader2.reset()

    # hm.run_diff_optimal('r', 100000000, "hit_rate_start_time_end_time", reader2, num_of_process=47,
    #                     # LRU=False, cache="LRU",
    #                     cache_size=CACHE_SIZE, save=False, change_label=True,
    #                     figname="LRU_NONPARDA_Optimal_" + str(CACHE_SIZE) + "3.png")

    # reader2.reset()


'''
    for CACHE_SIZE in range(30, 1200, 120):
        hm.run('r', 10000000, "hit_rate_start_time_end_time", reader2, num_of_process=8, # LRU=False, cache='optimal',
               cache_size=CACHE_SIZE, save=False, fixed_range=True, change_label=True,
               figname="0529/small_LRU_hit_rate_start_time_end_time_" + str(CACHE_SIZE) + ".png")
        reader2.reset()




        hm.run_diff_optimal('r', 10000000, "hit_rate_start_time_end_time", reader2, num_of_process=8, LRU=False, cache="LRU",
                            cache_size=CACHE_SIZE, save=False, change_label=True,
                            figname="0529/LRU_NONPARDA_Optimal_" + str(CACHE_SIZE) + ".png")
        reader2.reset()
        hm.run_diff_optimal('r', 10000000, "hit_rate_start_time_end_time", reader2, num_of_process=8, LRU=False,
                            cache="ARC",
                            cache_size=CACHE_SIZE, save=False, change_label=True,
                            figname="0529/ARC_NONPARDA_Optimal_" + str(CACHE_SIZE) + ".png")
        reader2.reset()

        hm.run_diff_optimal('r', 10000000, "hit_rate_start_time_end_time", reader2, num_of_process=8, LRU=False,
                            cache="RR",
                            cache_size=CACHE_SIZE, save=False, change_label=True,
                            figname="0529/RR_NONPARDA_Optimal_" + str(CACHE_SIZE) + ".png")
        reader2.reset()

        hm.run_diff_optimal('r', 10000000, "hit_rate_start_time_end_time", reader2, num_of_process=8, LRU=False,
                            cache="SLRU",
                            cache_size=CACHE_SIZE, save=False, change_label=True,
                            figname="0529/SLRU_NONPARDA_Optimal_" + str(CACHE_SIZE) + ".png")
        reader2.reset()



    # hm.run('r', 10000000, "hit_rate_start_time_end_time", reader2, LRU=False, num_of_process=8,
    #        cache='LFU_RR', cache_size=500, save=False, change_label=True, figname="LFU_RR.png")
    # reader2.reset()
    # hm.run('r', 10000000, "hit_rate_start_time_end_time", reader2, LRU=False, num_of_process=8,
    #        cache='ARC', cache_size=500, save=False, change_label=True, figname="ARC.png")
    # reader2.reset()
    # hm.run('r', 10000000, "hit_rate_start_time_end_time", reader2, LRU=False, num_of_process=8,
    #        cache='optimal', cache_size=500, save=False, change_label=True, figname="optimal.png")
    # hm.run('r', 10000000, "hit_rate_start_time_end_time", reader2, LRU=True, num_of_process=8,
    #       cache_size=500, save=False, change_label=True, figname="LRU.png")



    # p = pardaProfiler(20000, reader2)
    # p.run()
'''

if __name__ == "__main__":
    import time

    t1 = time.time()
    localtest()
    # server_plot_all()
    # cold_miss_2d('v', 100)
    t2 = time.time()
    print(t2 - t1)

    # server_plot_all()
    # server_plot_all_redis()
    # server_size_plot()
    # server_request_num_plot()
