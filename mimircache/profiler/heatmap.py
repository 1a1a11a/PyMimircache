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

import os
import pickle
from collections import deque
from multiprocessing import Array, Process, Queue

from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.cHeatmap import cHeatmap
import mimircache.c_heatmap as c_heatmap
from mimircache.profiler.heatmap_subprocess import *
from mimircache.utils.printing import *
from mimircache.const import *

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt

# DEBUG = True

__all__ = ("run", "draw")


class heatmap:
    def __init__(self):
        # if not os.path.exists('temp/'):
        #     os.mkdir('temp')
        pass

    def _prepare_reuse_distance_and_break_points(self, mode, reader, time_interval, calculate=True, save=False,
                                                 **kwargs):
        reader.reset()

        break_points = []

        if calculate:
            # build profiler for extracting reuse distance (For LRU)
            p = LRUProfiler(reader)
            reuse_dist = p.get_reuse_distance()

            if save:
                # needs to save the data
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

            breakpoints_filename = 'temp/break_points_' + mode + str(time_interval) + '.dat'
            # check whether the break points distribution table has been calculated
            if os.path.exists(breakpoints_filename):
                with open(breakpoints_filename, 'rb') as ifile:
                    break_points = pickle.load(ifile)

        # check break points are loaded or not, if not need to calculate it
        if not break_points:
            break_points = cHeatmap().gen_breakpoints(reader, mode, time_interval)
            if save:
                with open('temp/break_points_' + mode + str(time_interval) + '.dat', 'wb') as ifile:
                    pickle.dump(break_points, ifile)

        return reuse_dist, break_points

    def _prepare_multiprocess_params_LRU(self, mode, plot_type, break_points, **kwargs):
        # create the array for storing results
        # result is a dict: (x, y) -> heat, x, y is the left, lower point of heat square

        kwargs_subprocess = {}  # kw dictionary for passing parameters to subprocess
        kwargs_plot = {}  # kw dictionary passed to plotting functions
        if plot_type == "hit_rate_start_time_cache_size":
            max_rd = max(kwargs['reuse_dist'])
            kwargs_subprocess['max_rd'] = max_rd
            if 'bin_size' in kwargs:
                kwargs_subprocess["bin_size"] = kwargs['bin_size']
            else:
                kwargs_subprocess['bin_size'] = int(max_rd / DEFAULT_BIN_NUM_PROFILER)

            result = np.zeros((len(break_points) - 1, max_rd // kwargs_subprocess['bin_size'] + 1), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_hit_rate_start_time_cache_size_subprocess
            kwargs_plot['yticks'] = ticker.FuncFormatter(
                lambda x, pos: '{:2.0f}'.format(kwargs_subprocess['bin_size'] * x))
            kwargs_plot['xlabel'] = 'time({})'.format(mode)
            kwargs_plot['ylabel'] = 'cache size'
            kwargs_plot['zlabel'] = 'hit rate'
            kwargs_plot['title'] = "hit rate start time cache size"

            # plt.gca().yaxis.set_major_formatter(
            #     ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(kwargs_subprocess['bin_size'] * x)))

        elif plot_type == "hit_rate_start_time_end_time":
            # (x,y) means from time x to time y
            assert 'reader' in kwargs, 'please provide reader as keyword argument'
            assert 'cache_size' in kwargs, 'please provide cache_size as keyword argument'

            reader = kwargs['reader']
            kwargs_subprocess['cache_size'] = kwargs['cache_size']
            array_len = len(break_points) - 1
            result = np.zeros((array_len, array_len), dtype=np.float32)
            # result[:] = np.nan
            func_pointer = calc_hit_rate_start_time_end_time_subprocess

            kwargs_plot['xlabel'] = 'time({})'.format(mode)
            kwargs_plot['ylabel'] = 'time({})'.format(mode)
            kwargs_plot['yticks'] = ticker.FuncFormatter(
                lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))
            kwargs_plot['title'] = "hit_rate_start_time_end_time"

            last_access = c_heatmap.get_last_access_dist(reader.cReader)
            last_access_array = Array('l', len(last_access), lock=False)
            for i, j in enumerate(last_access):
                last_access_array[i] = j
            kwargs_subprocess['last_access_array'] = last_access_array


        elif plot_type == "avg_rd_start_time_end_time":
            # (x,y) means from time x to time y
            kwargs_plot['xlabel'] = 'time({})'.format(mode)
            kwargs_plot['ylabel'] = 'time({})'.format(mode)
            kwargs_plot['title'] = "avg_rd_start_time_end_time"

            array_len = len(break_points) - 1
            result = np.zeros((array_len, array_len), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_avg_rd_start_time_end_time_subprocess

        elif plot_type == "cold_miss_count_start_time_end_time":
            # (x,y) means from time x to time y
            kwargs_plot['xlabel'] = 'time({})'.format(mode)
            kwargs_plot['ylabel'] = 'cold miss count'
            kwargs_plot['title'] = "cold_miss_count_start_time_end_time"
            array_len = len(break_points) - 1
            result = np.zeros((array_len, array_len), dtype=np.float32)
            result[:] = np.nan
            func_pointer = calc_cold_miss_count_start_time_end_time_subprocess

        else:
            raise RuntimeError("cannot recognize plot type")

        kwargs_plot['xticks'] = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / len(break_points)))

        return result, kwargs_subprocess, kwargs_plot, func_pointer

    def calculate_heatmap_dat(self, reader, mode, time_interval, plot_type,
                              algorithm="LRU", cache_params=None, num_of_threads=4, **kwargs):
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

        if algorithm == "LRU":
            reuse_dist, break_points = self._prepare_reuse_distance_and_break_points(
                mode, reader, time_interval, **kwargs)

            # create shared memory for child process
            reuse_dist_share_array = Array('l', len(reuse_dist), lock=False)
            for i, j in enumerate(reuse_dist):
                reuse_dist_share_array[i] = j

            result, kwargs_subprocess, kwargs_plot, func_pointer = self._prepare_multiprocess_params_LRU(mode,
                                                                                                         plot_type,
                                                                                                         break_points,
                                                                                                         reuse_dist=reuse_dist_share_array,
                                                                                                         reader=reader,
                                                                                                         **kwargs)

        else:
            assert 'cache_size' in kwargs, "you didn't provide cache_size parameter"
            if isinstance(algorithm, str):
                algorithm = cache_name_to_class(algorithm)

            # prepare break points
            if mode[0] == 'r' or mode[0] == 'v':
                break_points = cHeatmap().gen_breakpoints(reader, mode[0], time_interval)
            else:
                raise RuntimeError("unrecognized mode, it can only be r or v")

            result, kwargs_subprocess, kwargs_plot, func_pointer = self._prepare_multiprocess_params_LRU(mode,
                                                                                                         plot_type,
                                                                                                         break_points,
                                                                                                         reader=reader,
                                                                                                         **kwargs)
            kwargs_subprocess['cache_params'] = cache_params

        # create shared memory for break points
        break_points_share_array = Array('i', len(break_points), lock=False)
        for i, j in enumerate(break_points):
            break_points_share_array[i] = j

        # Jason: memory usage can be optimized by not copying whole reuse distance array in each sub process
        # Jason: change to process pool for efficiency
        map_list = deque()
        for i in range(len(break_points) - 1):
            map_list.append(i)

        q = Queue()
        process_pool = []
        process_count = 0
        result_count = 0
        map_list_pos = 0
        while result_count < len(map_list):
            if process_count < num_of_threads and map_list_pos < len(map_list):
                # LRU related heatmaps
                if algorithm == LRU:
                    p = Process(target=func_pointer,
                                args=(map_list[map_list_pos], break_points_share_array, reuse_dist_share_array, q),
                                kwargs=kwargs_subprocess)
                else:

                    p = Process(target=calc_hit_rate_start_time_end_time_subprocess_general,
                                args=(map_list[map_list_pos], algorithm, break_points_share_array, reader, q),
                                kwargs=kwargs_subprocess)

                p.start()
                process_pool.append(p)
                process_count += 1
                map_list_pos += 1
            else:
                rl = q.get()
                for r in rl:
                    # print("[{}][{}] = {}".format(r[0], r[1], r[2]))
                    result[r[0]][r[1]] = r[2]
                process_count -= 1
                result_count += 1
            print("%2.2f%%" % (result_count / len(map_list) * 100), end='\r')
        for p in process_pool:
            p.join()

        # old 0510
        # with Pool(processes=self.num_of_threads) as p:
        #     for ret_list in p.imap_unordered(_calc_hit_rate_subprocess, map_list,
        #                                      chunksize=10):
        #         count += 1
        #         print("%2.2f%%" % (count / len(map_list) * 100), end='\r')
        #         for r in ret_list:  # l is a list of (x, y, hr)
        #             result[r[0]][r[1]] = r[2]

        return result.T, kwargs_plot


    @staticmethod
    def draw_heatmap(xydict, **kwargs):
        if 'figname' in kwargs:
            filename = kwargs['figname']
        else:
            filename = 'heatmap.png'

        masked_array = np.ma.array(xydict, mask=np.isnan(xydict))

        # print(masked_array)
        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        if 'fixed_range' in kwargs and kwargs['fixed_range']:
            img = plt.imshow(masked_array, vmin=0, vmax=1, interpolation='nearest', origin='lower',
                             aspect='auto')
        else:
            img = plt.imshow(masked_array, interpolation='nearest', origin='lower', aspect='auto')

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
        if 'title' in kwargs:
            plt.title(kwargs['title'])

        # # # change tick from arbitrary number to real time
        # if 'change_label' in kwargs and kwargs['change_label'] and 'interval' in kwargs:
        #     ticks = ticker.FuncFormatter(lambda x, pos: '{:2.2f}'.format(x * kwargs['interval'] / (10 ** 6) / 3600))
        #     plt.gca().xaxis.set_major_formatter(ticks)
        #     plt.gca().yaxis.set_major_formatter(ticks)
        #     plt.xlabel("time/hour")
        #     plt.ylabel("time/hour")

        # plt.show()
        plt.savefig(filename, dpi=600)
        colorfulPrint("red", "plot is saved at the same directory")
        plt.clf()


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

    def heatmap(self, reader, mode, time_interval, plot_type, algorithm="LRU", cache_params=None, **kwargs):
        """

        :param plot_type:
        :param mode:
        :param interval:
        :param reader:
        :param kwargs: include num_of_threads, figname
        :return:
        """
        reader.reset()

        if mode == 'r' or mode == 'v':
            xydict, kwargs_plot = self.calculate_heatmap_dat(reader, mode, time_interval, plot_type, algorithm,
                                                             cache_params=None, **kwargs)
        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        kwargs_plot.update(kwargs)
        self.draw_heatmap(xydict, **kwargs_plot)
        reader.reset()

        # self.__del_manual__()


def server_plot_all():
    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    for filename in os.listdir("../data/cloudphysics"):
        print(filename)
        if filename.endswith('.vscsitrace'):
            hm = heatmap()
            mem_size = mem_sizes[int(filename.split('_')[0][1:]) - 1] * 16
            reader = vscsiReader("../data/cloudphysics/" + filename)
            if os.path.exists("0601/" + filename + "LRU_Optimal_" + str(mem_size) + "_r.png"):
                continue

            # hm.run('r', 1000000000, "hit_rate_start_time_cache_size", reader, num_of_threads=48, cache_size=mem_size,
            #        save=True, bin_size=1000, text="size = " + str(mem_size), fixed_range=True,
            #        figname='0601/' + filename + '_hit_rate_start_time_cache_size_r.png')
            # # ,change_label='False'     ,   10000000
            #
            #
            # hm.run('r', 1000000000, "avg_rd_start_time_end_time", reader, num_of_threads=48,
            #        # LRU=False, cache='optimal',
            #        calculated=True, cache_size=mem_size,
            #        figname='0601/' + filename + "_avg_rd_start_time_end_time_r.png")
            #
            #
            # hm.run('r', 1000000000, "cold_miss_count_start_time_end_time", reader, num_of_threads=48,
            #        calculated=True, cache_size=mem_size,
            #        figname='0601/' + filename + "_cold_miss_count_start_time_end_time_r.png")
            #


            del hm
            del reader



def localtest():
    CACHE_SIZE = 2000
    TIME_INTERVAL = 10000000
    reader = vscsiReader("../data/trace.vscsi")
    hm = heatmap()
    from mimircache.cache.FIFO import FIFO

    # LRU
    # hm.heatmap(reader, 'r', TIME_INTERVAL, "hit_rate_start_time_end_time", num_of_threads=1, cache_size=CACHE_SIZE,
    #            save=True, text="size = " + str(CACHE_SIZE), fixed_range=True,
    #            figname='test_hit_rate_start_time_end_time_r.png')
    #
    # hm.heatmap(reader, 'r', TIME_INTERVAL, "hit_rate_start_time_cache_size", num_of_threads=8, cache_size=CACHE_SIZE,
    #            save=True, bin_size=1000, fixed_range=True,
    #            figname='test_hit_rate_start_time_cache_size_r.png')
    #
    # hm.heatmap(reader, 'r', TIME_INTERVAL, "avg_rd_start_time_end_time", num_of_threads=8,
    #            calculated=True, cache_size=CACHE_SIZE,
    #            figname="test_avg_rd_start_time_end_time_r.png")
    #
    # hm.heatmap(reader, 'r', TIME_INTERVAL, "cold_miss_count_start_time_end_time", num_of_threads=8,
    #            calculated=True, cache_size=CACHE_SIZE,
    #            figname="test_cold_miss_count_start_time_end_time_r.png")

    # Non-LRU
    hm.heatmap(reader, 'r', TIME_INTERVAL, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=CACHE_SIZE,
               save=True, text="size = " + str(CACHE_SIZE), fixed_range=True, algorithm=FIFO,  # "FIFO",
               figname='test_hit_rate_start_time_end_time_r.png')



if __name__ == "__main__":
    import time

    t1 = time.time()
    localtest()
    # server_plot_all()
    # cold_miss_2d('v', 100)
    t2 = time.time()
    print(t2 - t1)

