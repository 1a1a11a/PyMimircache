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


class heatmap:
    def __init__(self, cache_class=LRU):
        self.cache_class = cache_class
        if not os.path.exists('temp/'):
            os.mkdir('temp')
        self.other_plot_kwargs = {}

    def heatmap(self):
        pass

    def set_plot_params(self, axis, axis_type, **kwargs):
        log_num = 1
        label = ''
        tick = None
        xydict = None

        if 'xydict' in kwargs:
            xydict = kwargs['xydict']

        if 'fixed_range' in kwargs and kwargs['fixed_range']:
            # vmin, vmax = kwargs['fixed_range']
            self.other_plot_kwargs['vmin'] = 0
            self.other_plot_kwargs['vmax'] = 1

        if 'other_kwargs' in kwargs:
            self.other_plot_kwargs.update(kwargs['other_kwargs'])

        # pyplot.yscale('log')

        if 'text' in kwargs:
            ax = plt.gca()
            ax.text(kwargs['text'][0], kwargs['text'][1], kwargs['text'][2])  # , fontsize=20)  , color='blue')

        if axis == 'x':
            assert 'xydict' in kwargs, "you didn't provide xydict"
            array = xydict
        elif axis == 'y':
            array = xydict[0]
        elif axis == 'cb':  # color bar
            if axis_type == 'count':
                self.other_plot_kwargs['norm'] = matplotlib.colors.LogNorm()
        else:
            print("unsupported axis: " + str(axis))

        if axis_type == 'virtual_time':
            if 'label' in kwargs:
                label = kwargs['label']
            else:
                label = 'virtual time'
            # tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(bin_size * x))
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / (len(array) - 1)))
        elif axis_type == 'real_time':
            if 'label' in kwargs:
                label = kwargs['label']
            else:
                label = 'real time'
            # tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(bin_size * x))
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / (len(array) - 1)))

        elif axis_type == 'cache_size':
            label = 'cache size'
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(log_num ** x))

        elif axis_type == 'reuse_dist':
            label = 'reuse distance'
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(log_num ** x))

        if axis == 'x':
            plt.xlabel(label)
            plt.gca().xaxis.set_major_formatter(tick)

        elif axis == 'y':
            plt.ylabel(label)
            plt.gca().yaxis.set_major_formatter(tick)

        else:
            print("unsupported axis: " + str(axis))

    def run(self, mode, interval, plot_type, reader, **kwargs):
        """

        :param plot_type:
        :param mode:
        :param interval:
        :param reader:
        :param kwargs: include num_of_process, figname
        :return:
        """
        if 'cache_size' in kwargs:
            self.cache_size = kwargs['cache_size']
        self.interval = interval
        self.mode = mode
        reader.reset()
        if 'num_of_process' in kwargs:
            self.num_of_process = kwargs['num_of_process']
        else:
            self.num_of_process = 4

        if mode == 'r' or mode == 'v':
            xydict = c_heatmap.heatmap(reader.cReader, self.cache_size, 'LRU', 'r', 10000000,
                                       "hit_rate_start_time_end_time", 8)
        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        self.draw_heatmap(xydict, plot_type, interval=interval)
        reader.reset()
        # self.draw_test(xydict, **kwargs_dict)

        # self.__del_manual__()

    @staticmethod
    def draw_heatmap(xydict, **kwargs):
        if 'figname' in kwargs:
            filename = kwargs['figname']
        else:
            filename = 'heatmap.png'

        masked_array = np.ma.array(xydict, mask=np.tri(len(xydict), dtype=int))

        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        try:
            # ax.pcolor(masked_array.T, cmap=cmap)
            img = plt.imshow(masked_array.T, interpolation='nearest', origin='lower', aspect='auto')

            cb = plt.colorbar(img)

            # plt.show()
            plt.savefig(filename, dpi=600)
            colorfulPrint("red", "plot is saved at the same directory")
            plt.clf()
        except Exception as e:
            logging.warning(str(e))
            plt.pcolormesh(masked_array.T, cmap=cmap)
            plt.savefig(filename)


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
            reader = vscsiCacheReader("../data/cloudphysics/" + filename)
            if os.path.exists("0619/" + filename + "LRU_Optimal_" + str(mem_size) + "_r.png"):
                continue

            xydict = c_heatmap.differential_heatmap_with_Optimal(reader.cReader, mem_size, 'LRU', 'r', 1000000000,
                                                                 "hit_rate_start_time_end_time", num_of_threads=48)
            heatmap.draw_heatmap(xydict, "hit_rate_start_time_end_time",
                                 figname="0619/" + filename + "LRU_Optimal_" + str(
                                     mem_size) + "_r.png")  # , fixed_range=True, figname='LRU.png')
            reader.close()

            # hm.run_diff_optimal('r', 1000000000, "hit_rate_start_time_end_time", reader, num_of_process=48,
            #                     cache_size=mem_size, save=False, change_label=True,
            #                     figname='0601/'+filename+"LRU_Optimal_" + str(mem_size) + "_r.png")

            del hm
            del reader



def checker(xydict):
    for i in range(len(xydict)):
        for j in range(len(xydict[0])):
            if xydict[i][j]>1 or xydict[i][j]<0:
                print("{},{}: {}".format(i, j, xydict[i][j]))

if __name__ == "__main__":
    import time
    t1 = time.time()
    # server_plot_all()

    CACHESIZE = 327680
    BINSIZE = 1
    TIME_INTERVAL = 1000000000
    CACHE_TYPE = "Optimal"
    TIME_TYPE = 'r'
    PLOT_TYPE = "hit_rate_start_time_end_time"
    NUM_OF_THREADS = 48
    import mimircache.c_LRUProfiler as c_LRUProfiler
    from mimircache.cacheReader.plainReader import plainCacheReader
    from mimircache.profiler.generalProfiler import generalProfiler

    hm = heatmap()
    # reader = vscsiCacheReader('../data/trace.vscsi')
    reader = vscsiCacheReader('/home/cloudphysics/traces/w84_vscsi1.vscsitrace')

    # xydict = c_heatmap.heatmap(reader.cReader, CACHESIZE, 'LRU', 'r', 1000000000, "hit_rate_start_time_end_time", num_of_threads=NUM_OF_THREADS)
    # hm.draw_heatmap(xydict, figname='LRU.png')  # "hit_rate_start_time_end_time") #, fixed_range=True, figname='LRU.png')
    # print("Time: " + str(time.time() - t1))
    # t1 = time.time()
    # print("0,2: {}".format(xydict[0][2]))
    # checker(xydict)
    #
    # xydict = c_heatmap.heatmap(reader.cReader, CACHESIZE, 'Optimal', 'r', 1000000000, "hit_rate_start_time_end_time", num_of_threads=NUM_OF_THREADS)
    # hm.draw_heatmap(xydict, figname='Optimal.png')  # "hit_rate_start_time_end_time") #, fixed_range=True, figname='LRU.png')
    # print("Time: " + str(time.time() - t1))
    # t1 = time.time()
    # print("0,2: {}".format(xydict[0][2]))
    # checker(xydict)
    #
    # xydict = c_heatmap.differential_heatmap_with_Optimal(reader.cReader, CACHESIZE, CACHE_TYPE, TIME_TYPE,
    #                                                      TIME_INTERVAL, PLOT_TYPE, num_of_threads=NUM_OF_THREADS)
    # text = "   cache size: {},\n   cache type: {},\n   time type: {},\n   time interval: {},\n   plot type: \n{}".format(
    #     CACHESIZE, CACHE_TYPE, TIME_TYPE, TIME_INTERVAL, PLOT_TYPE)
    # x1, y1 = xydict.shape
    # x1 /= 3
    # y1 /= 8
    # hm.set_plot_params('x', 'real_time', xydict=xydict, label='start time (real)', text=(x1, y1, text))
    # hm.set_plot_params('y', 'real_time', xydict=xydict, label='end time (real)')
    # hm.draw_heatmap(xydict, figname='Optimal_LRU.png')  # "hit_rate_start_time_end_time") #, fixed_range=True, figname='LRU.png')
    # print("Time: " + str(time.time() - t1))
    # t1 = time.time()
    # print("0,2: {}".format(xydict[0][2]))
    # checker(xydict)

    xydict = c_heatmap.heatmap_rd_distribution(reader.cReader, TIME_TYPE, TIME_INTERVAL)
    hm.draw_heatmap(xydict, figname='rd_dist.png')



    # print("Reuse dist")
    # rd = c_LRUProfiler.get_reuse_dist_seq(reader.cReader, begin=23, end=43)
    # print(rd)
    #
    # print("hit count")
    # hc = c_LRUProfiler.get_hit_count_seq(reader.cReader, CACHESIZE, begin=23, end=43)
    # print(hc)
    #
    # print("Hit rate_LRU")
    # hr_LRU = c_LRUProfiler.get_hit_rate_seq(reader.cReader, CACHESIZE, begin=23, end=43)
    # print(hr_LRU[:-2])
    #
    # print("next access")
    # na = c_heatmap.get_next_access_dist(reader.cReader, 23, 43)
    # print(na)
    #

    # print("Hit rate Optimal")
    # hr = c_generalProfiler.get_hit_rate(reader.cReader, CACHESIZE, "Optimal", BINSIZE, 8, begin=23, end=43)
    # print(hr)

    # reader = plainCacheReader('../data/test.dat')
    # from mimircache.cache.Optimal import optimal
    # p = generalProfiler(optimal, (CACHESIZE, reader), BINSIZE, reader, 2)
    import time
    # p.run()
    # hr_Optimal_py = p.HRC
    # print(hr_Optimal_py)
