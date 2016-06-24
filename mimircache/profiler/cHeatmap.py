import logging
import os
import pickle
from collections import deque
import time
import matplotlib
import matplotlib.ticker as ticker
import mimircache.c_heatmap as c_heatmap
import numpy as np
from matplotlib import pyplot as plt

from mimircache import const
from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.oldModule.pardaProfiler import pardaProfiler
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.heatmap_subprocess import *
from mimircache.profiler.heatmap_subprocess2d import *
from mimircache.utils.printing import *


class cHeatmap:
    def __init__(self):
        # if not os.path.exists('temp/'):
        #     os.mkdir('temp')
        self.other_plot_kwargs = {}
        self.cache_size = -1

    def set_plot_params(self, axis, axis_type, **kwargs):
        log_base = 1
        label = ''
        tick = None
        xydict = None

        if 'xydict' in kwargs:
            xydict = kwargs['xydict']

        if 'log_base' in kwargs:
            log_base = kwargs['log_base']

        if 'fixed_range' in kwargs and kwargs['fixed_range']:
            # if 'vmin' in kwargs:
            #     self.other_plot_kwargs['vmin'] = kwargs['vmin']
            # vmin, vmax = kwargs['fixed_range']
            # self.other_plot_kwargs['vmin'] = 0
            # self.other_plot_kwargs['vmax'] = 1
            self.other_plot_kwargs['fixed_range'] = kwargs['fixed_range']

        if 'other_kwargs' in kwargs:
            self.other_plot_kwargs.update(kwargs['other_kwargs'])

        if 'text' in kwargs:
            ax = plt.gca()
            ax.text(kwargs['text'][0], kwargs['text'][1], kwargs['text'][2])  # , fontsize=20)  , color='blue')

        if axis == 'x':
            assert 'xydict' in kwargs, "you didn't provide xydict"
            array = xydict[0]
        elif axis == 'y':
            assert 'xydict' in kwargs, "you didn't provide xydict"
            array = xydict
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
            assert log_base != 1, "please provide log_base"
            label = 'cache size'
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(log_base ** x))

        elif axis_type == 'reuse_dist':
            assert log_base != 1, "please provide log_base"
            label = 'reuse distance'
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(log_base ** x))

        if axis == 'x':
            plt.xlabel(label)
            plt.gca().xaxis.set_major_formatter(tick)

        elif axis == 'y':
            plt.ylabel(label)
            plt.gca().yaxis.set_major_formatter(tick)

        elif axis == 'cb':
            pass

        else:
            print("unsupported axis: " + str(axis))

    def heatmap(self, reader, mode, time_interval, plot_type, **kwargs):
        """

        :param plot_type:
        :param mode:
        :param interval:
        :param reader:
        :param kwargs: include num_of_process, figname
        :return:
        """

        algorithm = "LRU"
        figname = None
        self.time_interval = time_interval
        self.mode = mode
        reader.reset()

        if 'cache_size' in kwargs:
            self.cache_size = kwargs['cache_size']

        if 'num_of_threads' in kwargs:
            self.num_of_threads = kwargs['num_of_threads']
        else:
            self.num_of_threads = 4

        if 'algorithm' in kwargs:
            algorithm = kwargs['algorithm']

        if 'figname' in kwargs:
            figname = kwargs['figname']

        if mode == 'r' or mode == 'v':
            if plot_type == "hit_rate_start_time_end_time":
                # assert algorithm!=None, "please specify your cache replacement algorithm in heatmap plotting"
                assert self.cache_size != -1, "please provide cache_size parameter for plotting hit_rate_start_time_end_time"
                print("going to plot heatmap of hit_rate_start_time_end_time using {} "
                      "of cache size: {}".format(algorithm, self.cache_size))

                if algorithm.lower() in const.c_available_cache:
                    xydict = c_heatmap.heatmap(reader.cReader, self.cache_size, algorithm, mode, time_interval,
                                               plot_type, num_of_threads=self.num_of_threads)
                else:
                    # xydict =
                    raise RuntimeError("haven't provide support yet")
                    pass

                text = "      cache size: {},\n      cache type: {},\n      time type: {},\n      time interval: " \
                       "{},\n      plot type: \n{}".format(self.cache_size, algorithm, mode, time_interval, plot_type)

                x1, y1 = xydict.shape
                x1 = int(x1 / 2.8)
                y1 /= 8
                if mode == 'r':
                    self.set_plot_params('x', 'real_time', xydict=xydict, label='start time (real)',
                                         text=(x1, y1, text))
                    self.set_plot_params('y', 'real_time', xydict=xydict, label='end time (real)', fixed_range=(0, 1))
                else:
                    self.set_plot_params('x', 'virtual_time', xydict=xydict, label='start time (virtual)',
                                         text=(x1, y1, text))
                    self.set_plot_params('y', 'virtual_time', xydict=xydict, label='end time (virtual)',
                                         fixed_range=(0, 1))

                if not figname:
                    figname = '_'.join([algorithm, str(self.cache_size), plot_type]) + '.png'
                self.draw_heatmap(xydict, figname=figname)



            elif plot_type == "hit_rate_start_time_cache_size":
                pass


            elif plot_type == "avg_rd_start_time_end_time":
                pass


            elif plot_type == "cold_miss_count_start_time_end_time":
                print("this plot is discontinued")


            elif plot_type == "???":
                pass




            elif plot_type == "rd_distribution":
                if not figname:
                    figname = 'rd_distribution.png'

                xydict, log_base = c_heatmap.heatmap_rd_distribution(reader.cReader, mode, time_interval,
                                                                     num_of_threads=self.num_of_threads)
                self.set_plot_params('x', 'real_time', xydict=xydict)
                self.set_plot_params('y', 'reuse_dist', xydict=xydict, log_base=log_base)
                self.set_plot_params('cb', 'count')
                self.draw_heatmap(xydict, figname=figname, not_mask=True)

            elif plot_type == "rd_distribution_CDF":
                if not figname:
                    figname = 'rd_distribution_CDF.png'

                xydict, log_base = c_heatmap.heatmap_rd_distribution(reader.cReader, mode, time_interval,
                                                                     num_of_threads=self.num_of_threads, CDF=1)
                self.set_plot_params('x', 'real_time', xydict=xydict)
                self.set_plot_params('y', 'reuse_dist', xydict=xydict, log_base=log_base)
                # self.set_plot_params('cb', 'count')
                self.draw_heatmap(xydict, figname=figname, not_mask=True)


            elif plot_type == "future_rd_distribution":
                if not figname:
                    figname = 'future_rd_distribution.png'

                xydict, log_base = c_heatmap.heatmap_future_rd_distribution(reader.cReader, mode, time_interval,
                                                                            num_of_threads=self.num_of_threads)
                self.set_plot_params('x', 'real_time', xydict=xydict)
                self.set_plot_params('y', 'reuse_dist', xydict=xydict, log_base=log_base)
                self.set_plot_params('cb', 'count')
                self.draw_heatmap(xydict, figname=figname, not_mask=True)


            else:
                raise RuntimeError(
                    "Cannot recognize this plot type, please check documentation, yoru input is %s" % mode)

        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        reader.reset()

    def differential_heatmap(self, reader, mode, time_interval, plot_type, alg1, alg2="Optimal", **kwargs):
        """

        :param plot_type:
        :param mode:
        :param interval:
        :param reader:
        :param kwargs: include num_of_process, figname
        :return:
        """

        self.time_interval = time_interval
        self.mode = mode
        figname = '_'.join([alg2 + '-' + alg1, str(self.cache_size), plot_type]) + '.png'
        reader.reset()

        if 'cache_size' in kwargs:
            self.cache_size = kwargs['cache_size']

        if 'num_of_threads' in kwargs:
            self.num_of_threads = kwargs['num_of_threads']
        else:
            self.num_of_threads = 4

        if 'figname' in kwargs:
            figname = kwargs['figname']

        if mode == 'r' or mode == 'v':
            if plot_type == "hit_rate_start_time_end_time":
                assert self.cache_size != -1, "please provide cache_size for plotting hit_rate_start_time_end_time"
                print("going to plot differential heatmap of hit_rate_start_time_end_time using {} - {} "
                      "of cache size: {}".format(alg2, alg1, self.cache_size))

                xydict = c_heatmap.differential_heatmap(reader.cReader, self.cache_size, alg1, alg2, mode,
                                                        time_interval,
                                                        plot_type, num_of_threads=self.num_of_threads)
                # xydict = c_heatmap.differential_heatmap_with_Optimal(reader.cReader, self.cache_size, alg1, mode,
                #                                         time_interval,
                #                                         plot_type, num_of_threads=self.num_of_threads)

                text = "      differential heatmap\n      cache size: {},\n      cache type: {}-{},\n      time type: {},\n      time interval: " \
                       "{},\n      plot type: \n{}".format(self.cache_size, alg2, alg1, mode, time_interval, plot_type)

                x1, y1 = xydict.shape
                x1 = int(x1 / 2.8)
                y1 /= 8
                if mode == 'r':
                    self.set_plot_params('x', 'real_time', xydict=xydict, label='start time (real)',
                                         text=(x1, y1, text))
                    self.set_plot_params('y', 'real_time', xydict=xydict, label='end time (real)')
                else:
                    self.set_plot_params('x', 'virtual_time', xydict=xydict, label='start time (virtual)',
                                         text=(x1, y1, text))
                    self.set_plot_params('y', 'virtual_time', xydict=xydict, label='end time (virtual)')

                self.draw_heatmap(xydict, figname=figname)



            elif plot_type == "hit_rate_start_time_cache_size":
                pass


            elif plot_type == "avg_rd_start_time_end_time":
                pass


            elif plot_type == "cold_miss_count_start_time_end_time":
                print("this plot is discontinued")


            elif plot_type == "???":
                pass



            else:
                raise RuntimeError(
                    "Cannot recognize this plot type, please check documentation, yoru input is %s" % mode)

        else:
            raise RuntimeError("Cannot recognize this mode, it can only be either real time(r) or virtual time(v), "
                               "but you input %s" % mode)

        reader.reset()

    def draw_heatmap(self, xydict, **kwargs):
        if 'figname' in kwargs:
            filename = kwargs['figname']
        else:
            filename = 'heatmap.png'

        if 'not_mask' in kwargs and kwargs['not_mask']:
            plot_array = xydict
        else:
            plot_array = np.ma.array(xydict, mask=np.tri(len(xydict), dtype=int).T)

        print(plot_array)

        cmap = plt.cm.jet
        cmap.set_bad('w', 1.)

        # plt.yscale('log')

        plt.title("Heatmap")

        try:
            # ax.pcolor(masked_array.T, cmap=cmap)
            # print(str(**self.other_plot_kwargs))
            if 'fixed_range' in self.other_plot_kwargs and self.other_plot_kwargs['fixed_range']:
                vmin, vmax = self.other_plot_kwargs['fixed_range']
                del self.other_plot_kwargs['fixed_range']
                img = plt.imshow(plot_array, vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower',
                                 aspect='auto', **self.other_plot_kwargs)
            else:
                img = plt.imshow(plot_array, interpolation='nearest', origin='lower', aspect='auto',
                                 **self.other_plot_kwargs)

            # , vmin=0, vmax=1

            cb = plt.colorbar(img)

            plt.savefig(filename, dpi=600)
            plt.show()
            colorfulPrint("red", "plot is saved at the same directory")
            plt.clf()
            self.other_plot_kwargs.clear()
        except Exception as e:
            logging.warning(str(e))
            plt.pcolormesh(plot_array.T, cmap=cmap)
            plt.savefig(filename)


def server_plot_all(path="../data/cloudphysics/", num_of_threads=48):
    TIME_INTERVAL = 1000000000
    PLOT_TYPE = "hit_rate_start_time_end_time"
    MODE = 'r'
    mem_sizes = []
    with open('memSize', 'r') as ifile:
        for line in ifile:
            mem_sizes.append(int(line.strip()))

    for filename in os.listdir(path):
        print(filename)
        if filename.endswith('.vscsitrace'):
            hm = cHeatmap()
            mem_size = mem_sizes[int(filename.split('_')[0][1:]) - 1] * 16
            reader = vscsiCacheReader(path + filename)
            # mem_sizes = LRUProfiler(reader).get_best_cache_sizes(12, 200, 20)
            #
            # if os.path.exists("0624/" + '_'.join([filename, "_LRU_Optimal_", str(mem_sizes[-1]), MODE]) + "_.png"):
            #     continue

            t1 = time.time()
            hm.heatmap(reader, MODE, TIME_INTERVAL, "rd_distribution_CDF", num_of_threads=num_of_threads, figname=
            "0624/" + '_'.join([filename, "_rd_distribution_CDF", MODE]) + "_.png")
            print("time: " + str(time.time() - t1))

            # t1 = time.time()
            # hm.heatmap(reader, MODE, TIME_INTERVAL, "future_rd_distribution", num_of_threads=num_of_threads,
            #            figname="0620/" + '_'.join([filename, "_future_rd_distribution_", MODE]) + "_.png")
            # print("time: " + str(time.time() - t1))


            # for mem_size in mem_sizes:
            #     t1 = time.time()
            #     hm.differential_heatmap(reader, MODE, TIME_INTERVAL, PLOT_TYPE, "LRU", cache_size=mem_size,
            #                             num_of_threads=num_of_threads, figname=
            #                             "0624/" + '_'.join([filename, "_LRU_Optimal_", str(mem_size), MODE]) + "_.png")
            #     print("cache size: " + str(mem_size) + ", time: " + str(time.time() - t1))

            # xydict = c_heatmap.differential_heatmap_with_Optimal(reader.cReader, mem_size, 'LRU', 'r', ,
            #                                                      "hit_rate_start_time_end_time", num_of_threads=48)
            # heatmap.draw_heatmap(xydict, "hit_rate_start_time_end_time",
            #                      figname="0619/" + filename + "LRU_Optimal_" + str(
            #                          mem_size) + "_r.png")  # , fixed_range=True, figname='LRU.png')


            reader.close()

            del hm
            del reader


def localtest():
    TIME_INTERVAL = 10000000
    PLOT_TYPE = "hit_rate_start_time_end_time"
    MODE = 'r'
    NUM_OF_THREADS = 8
    CACHE_SIZE = 2000

    reader = vscsiCacheReader('../data/trace.vscsi')

    hm = cHeatmap()

    print('a')
    hm.heatmap(reader, MODE, TIME_INTERVAL, "rd_distribution_CDF", num_of_threads=NUM_OF_THREADS,
               figname="rd_dist_CDF_relative.png")

    print('b')
    t1 = time.time()
    # hm.differential_heatmap(reader, MODE, TIME_INTERVAL, PLOT_TYPE, "FIFO", cache_size=CACHE_SIZE,
    #                         num_of_threads=NUM_OF_THREADS, figname="heatmap.png")
    print(time.time() - t1)


def checker(xydict):
    for i in range(len(xydict)):
        for j in range(len(xydict[0])):
            if xydict[i][j] > 1 or xydict[i][j] < 0:
                print("{},{}: {}".format(i, j, xydict[i][j]))


if __name__ == "__main__":
    import time

    t1 = time.time()
    server_plot_all(path="/run/shm/traces/")
    # server_plot_all(path="../../../../disk/traces/", num_of_threads=16)
    localtest()

    CACHESIZE = 327680
    BINSIZE = 1
    TIME_INTERVAL = 10000000
    CACHE_TYPE = "Optimal"
    TIME_TYPE = 'r'
    PLOT_TYPE = "hit_rate_start_time_end_time"
    NUM_OF_THREADS = 8
    import mimircache.c_LRUProfiler as c_LRUProfiler
    from mimircache.cacheReader.plainReader import plainCacheReader
    from mimircache.profiler.generalProfiler import generalProfiler

    # hm = heatmap()
    reader = vscsiCacheReader('../data/trace.vscsi')
    # reader = vscsiCacheReader('/home/cloudphysics/traces/w84_vscsi1.vscsitrace')




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
