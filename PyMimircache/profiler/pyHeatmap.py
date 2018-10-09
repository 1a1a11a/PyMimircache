# coding=utf-8
"""
this module provides the pyheatmap ploting engine, it is in Python, so it is very slow
it supports both virtual time (fixed number of trace requests) and
real time
it support using multiprocessing to speed up computation
currently, it only support LRU
the mechanism it works is it gets reuse distance from LRUProfiler
currently supported plot type:

        # 1. hit_ratio_start_time_end_time
        # 2. hit_ratio_start_time_cache_size
        # 3. avg_rd_start_time_end_time
        # 4. cold_miss_count_start_time_end_time
        # 5. rd_distribution

This module hasn't been fully optimized and might not get optimized soon
If you want to use this module instead of cHeatmap and
need optimizations, you can do it yourself or contact the author

Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""

import os
from concurrent.futures import as_completed, ProcessPoolExecutor

# this should be replaced by pure python module
from PyMimircache.const import DEF_NUM_BIN_PROF, DEF_EMA_HISTORY_WEIGHT

from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE
if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.Heatmap as c_heatmap
    from PyMimircache.profiler.cLRUProfiler import CLRUProfiler as LRUProfiler
else:
    from PyMimircache.profiler.pyLRUProfiler import PyLRUProfiler as LRUProfiler
from PyMimircache.profiler.profilerUtils import get_breakpoints, draw_heatmap
from PyMimircache.profiler.pyHeatmapAux import *
from PyMimircache.profiler.utils.dist import get_last_access_dist
from PyMimircache.utils.printing import *
from PyMimircache.const import cache_name_to_class
# from PyMimircache.const import *

import matplotlib.ticker as ticker
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt




class PyHeatmap:
    """
    heatmap class for plotting heatmaps in Python
    """

    all = ["heatmap", "diff_heatmap"]

    def __init__(self):
        """ nothing needs to be done, not using all static method due to consistency between profilers """
        pass


    def compute_heatmap(self,
                        reader, plot_type, time_mode, time_interval,
                        cache_size=-1,
                        num_of_pixel_of_time_dim=-1,
                        num_of_threads=os.cpu_count(), **kwargs):
        """
            calculate the data for plotting heatmap

        :param reader: reader for data
        :param plot_type: types of data, see heatmap (function) for details
        :param time_mode: real time (r) or virtual time (v)
        :param time_interval: the window size in computation
        :param cache_size: size of cache
        :param num_of_pixel_of_time_dim: as an alternative to time_interval, useful when you don't know the trace time span
        :param num_of_threads: number of threads/processes to use for computation, default: all
        :param kwargs: cache_params,
        :return:  a two-dimension list, the first dimension is x, the second dimension is y, the value is the heat value
        """


        bp = get_breakpoints(reader, time_mode, time_interval, num_of_pixel_of_time_dim)
        ppe = ProcessPoolExecutor(max_workers=num_of_threads)
        futures_dict = {}
        progress = 0
        xydict = np.zeros((len(bp)-1, len(bp)-1))


        if plot_type in [
            "avg_rd_st_et",
            "rd_distribution",
            "rd_distribution_CDF",
            "future_rd_distribution",
            "dist_distribution",
            "rt_distribution"
        ]:
            pass

        elif plot_type == "hr_st_et":
            ema_coef = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)
            enable_ihr = kwargs.get("interval_hit_ratio", False) or kwargs.get("enable_ihr", False)


            if kwargs.get("algorithm", "LRU").lower() == "lru":
                #TODO: replace CLRUProfiler with PyLRUProfiler
                rd = LRUProfiler(reader).get_reuse_distance()
                last_access_dist = get_last_access_dist(reader)

                for i in range(len(bp) - 1):
                    futures_dict[ppe.submit(cal_hr_list_LRU, rd, last_access_dist,
                                            cache_size, bp, i, enable_ihr=enable_ihr, ema_coef=ema_coef)] = i
            else:
                reader_params = reader.get_params()
                reader_params["open_c_reader"] = False
                cache_class = cache_name_to_class(kwargs.get("algorithm"))
                cache_params = kwargs.get("cache_params", {})

                for i in range(len(bp) - 1):
                    futures_dict[ppe.submit(cal_hr_list_general, reader.__class__, reader_params,
                                            cache_class, cache_size, bp, i, cache_params=cache_params)] = i

        elif plot_type == "hr_st_size":
            raise RuntimeError("Not Implemented")

        elif plot_type == "KL_st_et":
            rd = LRUProfiler(reader).get_reuse_distance()

            for i in range(len(bp) - 1):
                futures_dict[ppe.submit(cal_KL, rd, bp, i)] = i


        else:
            ppe.shutdown()
            raise RuntimeError("{} is not a valid heatmap type".format(plot_type))


        last_progress_print_time = time.time()
        for future in as_completed(futures_dict):
            result = future.result()
            xydict[-len(result):, futures_dict[future]] = np.array(result)
            # print("{} {}".format(xydict[futures_dict[future]], np.array(result)))
            progress += 1
            if time.time() - last_progress_print_time > 20:
                INFO("{:.2f}%".format(progress / len(futures_dict) * 100), end="\r")
                last_progress_print_time = time.time()

        ppe.shutdown()
        return xydict


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


    def heatmap(self, reader, time_mode, plot_type,
                algorithm="LRU",
                time_interval=-1,
                num_of_pixel_of_time_dim=-1,
                cache_params=None,
                **kwargs):
        """

        This functions provides different types of heatmap plotting

        :param reader: the reader instance for data input
        :param time_mode: either real time (r) or virtual time (v),
                    real time is wall clock time, it needs the reader containing real time info
                    virtual time is the reference number, aka. the number of requests
        :param plot_type: different types of heatmap, supported heatmaps are listed in the table below
        :param algorithm: cache replacement algorithm (default: LRU)
        :param time_interval: the time interval of each pixel
        :param num_of_pixel_of_time_dim: if don't want to specify time_interval, you can also specify how many pixels you want
        :param cache_params: params used in cache
        :param kwargs: include num_of_threads, figname, enable_ihr, ema_coef (default: 0.8), plot_kwargs, filter_rd, filter_count
        :return: a numpy two dimensional array


        ============================  ========================  ===========================================================================
        plot_type                       required parameters         descriptions
        ============================  ========================  ===========================================================================
        "hr_st_et"                      cache_size              hit ratio with regarding to start time (x) and end time (y)
        "hr_st_size"                    NOT IMPLEMENTED         hit ratio with reagarding to start time (x) and size (y)
        "avg_rd_st_et"                  NOT IMPLEMENTED         average reuse distance with regaarding to start time (x) and end time (y)
        "rd_distribution"               N/A                     reuse distance distribution (y) vs time (x)
        "rd_distribution_CDF"           N/A                     reuse distance distribution CDF (y) vs time (x)
        "future_rd_distribution"        N/A                     future reuse distance distribution (y) vs time (x)
        "dist_distribution"             N/A                     absolute distance distribution (y) vs time (x)
        "rt_distribution"               N/A                     reuse time distribution (y) vs time (x)
        ============================  ========================  ===========================================================================


        """

        reader.reset()

        cache_size = kwargs.get("cache_size", -1)
        figname = kwargs.get("figname", "PyHeatmap_{}.png".format(plot_type))
        num_of_threads = kwargs.get("num_of_threads", os.cpu_count())
        if cache_params is None: cache_params = {}
        plot_kwargs = kwargs.get("plot_kwargs", {"figname": figname})
        assert time_mode in ["r", "v"], "Cannot recognize this time_mode, "\
                                        "it can only be either real time(r) or virtual time(v), " \
                                        "but you give {}".format(time_mode)


        if plot_type == "hr_st_et":
            assert cache_size != -1, "please provide cache_size parameter for plotting hr_st_et"
            # this is used to specify the type of hit ratio for each pixel, when it is False, it means
            # the hit ratio is the overall hit ratio from the beginning of trace,
            # if True, then the hit ratio will be exponentially decayed moving average hit ratio from beginning plus
            # the hit ratio in current time interval, the ewma_coefficient specifies the weight of
            # history hit ratio in the calculation
            enable_ihr = kwargs.get("interval_hit_ratio", False) or kwargs.get("enable_ihr", False)
            ema_coef  = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)

            xydict = self.compute_heatmap(reader, plot_type, time_mode, time_interval,
                                          cache_size=cache_size,
                                          algorithm=algorithm,
                                          enable_ihr=enable_ihr,
                                          ema_coef=ema_coef,
                                          num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                          cache_params=cache_params,
                                          num_of_threads=num_of_threads)


            text = "cache size: {},\ncache type: {},\ntime mode:  {},\ntime interval: {},\n" \
                   "plot type:  {}".format(cache_size, algorithm, time_mode, time_interval, plot_type)

            # coordinate to put the text
            x1, y1 = xydict.shape
            x1, y1 = int(x1 / 2.4), y1/8
            ax = plt.gca()
            ax.text(x1, y1, text)  # , fontsize=20)  , color='blue')

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", 'Start Time ({})'.format(time_mode))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "End Time ({})".format(time_mode))
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[0]-1))))
            plot_kwargs["imshow_kwargs"] = {"vmin": 0, "vmax": 1}

            plot_data = np.ma.array(xydict, mask=np.tri(len(xydict), k=-1, dtype=int).T)


        elif plot_type == "hr_st_size":
            # hit ratio with start time and cache size
            assert cache_size != -1, "please provide cache_size parameter for plotting hr_interval_size"
            bin_size = kwargs.get("bin_size", cache_size // DEF_NUM_BIN_PROF + 1)
            enable_ihr = kwargs.get("interval_hit_ratio", False) or kwargs.get("enable_ihr", False)
            ema_coef  = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)

            xydict = self.compute_heatmap(reader, plot_type, time_mode, time_interval,
                                          cache_size=cache_size,
                                          algorithm=algorithm,
                                          enable_ihr=enable_ihr,
                                          ema_coef=ema_coef,
                                          bin_size = bin_size,
                                          num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                          cache_params=cache_params,
                                          num_of_threads=num_of_threads)

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", 'Time ({})'.format(time_mode))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Cache Size")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: int(x*bin_size)))
            plot_kwargs["imshow_kwargs"] = {"vmin": 0, "vmax": 1}

            plot_data = xydict


        elif plot_type == "KL_st_et":
            xydict = self.compute_heatmap(reader, plot_type, time_mode, time_interval,
                                          num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                          num_of_threads=num_of_threads)

            text = "time type:  {},\ntime interval: {},\nplot type:  {}".format(time_mode, time_interval, plot_type)

            # coordinate to put the text
            x1, y1 = xydict.shape
            x1, y1 = int(x1 / 2), y1/8
            ax = plt.gca()
            ax.text(x1, y1, text)  # , fontsize=20)  , color='blue')

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", 'Start Time ({})'.format(time_mode))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "End Time ({})".format(time_mode))
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[0]-1))))

            plot_data = np.ma.array(xydict, mask=np.tri(len(xydict), k=-1, dtype=int).T)


        elif plot_type == "avg_rd_start_time_end_time" or plot_type == "avg_rd_st_et":
            raise RuntimeError("NOT Implemented")


        elif plot_type == "cold_miss_count_start_time_end_time":
            raise RuntimeError("this plot is deprecated")


        elif plot_type == "rd_distribution":
            xydict, log_base = c_heatmap.\
                hm_rd_distribution(reader.c_reader, time_mode,
                                   time_interval=time_interval,
                                   num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                   num_of_threads=num_of_threads)

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", '{} Time'.format("Real" if time_mode=="r" else "Virtual"))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Reuse Distance")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1)))
            plot_kwargs["imshow_kwargs"] = {"norm": colors.LogNorm()}

            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)


        elif plot_type == "rd_distribution_CDF":
            xydict, log_base = c_heatmap.\
                hm_rd_distribution(reader.c_reader, time_mode,
                                   time_interval=time_interval,
                                   num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                   num_of_threads=num_of_threads, CDF=1)

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", '{} Time'.format("Real" if time_mode=="r" else "Virtual"))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Reuse Distance (CDF)")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1)))
            plot_kwargs["imshow_kwargs"] = {"norm": colors.LogNorm()}


            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)


        elif plot_type == "future_rd_distribution":
            xydict, log_base = c_heatmap.\
                hm_future_rd_distribution(reader.c_reader, time_mode,
                                          time_interval=time_interval,
                                          num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                          num_of_threads=num_of_threads)


            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", '{} Time'.format("Real" if time_mode=="r" else "Virtual"))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Future Reuse Distance")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1)))
            plot_kwargs["imshow_kwargs"] = {"norm": colors.LogNorm()}


            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)


        elif plot_type == "dist_distribution":
            xydict, log_base = c_heatmap.\
                hm_dist_distribution(reader.c_reader, time_mode,
                                     time_interval=time_interval,
                                     num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                     num_of_threads=num_of_threads)

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", '{} Time'.format("Real" if time_mode=="r" else "Virtual"))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Distance")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1)))
            plot_kwargs["imshow_kwargs"] = {"norm": colors.LogNorm()}


            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)


        elif plot_type == "rt_distribution":
            xydict, log_base = c_heatmap.\
                hm_reuse_time_distribution(reader.c_reader, time_mode,
                                           time_interval=time_interval,
                                           num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                           num_of_threads=num_of_threads)

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", '{} Time'.format("Real" if time_mode=="r" else "Virtual"))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Reuse Time")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1)))
            plot_kwargs["imshow_kwargs"] = {"norm": colors.LogNorm()}


            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)

            if 'filter_rt' in kwargs:
                assert kwargs['filter_rt'] > 0, "filter_rt must be positive"
                index_pos = int(np.log(kwargs['filter_rt'])/np.log(log_base))
                plot_data[:index_pos+1, :] = 0


        else:
            raise RuntimeError("PyHeatmap does not support {} "
                "please check documentation".format(plot_type))


        if 'filter_rd' in kwargs:
            # make reuse distance < filter_rd invisible, this can be useful
            # when the low-reuse-distance request dominates, which makes the heatmap hard to read
            assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
            index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
            plot_data[:index_pos+1, :] = 0

        if 'filter_count' in kwargs:
            assert kwargs['filter_count'] > 0, "filter_count must be positive"
            plot_data = np.ma.array(plot_data, mask=(plot_data < kwargs['filter_count']))


        draw_heatmap(plot_data, **plot_kwargs)
        reader.reset()
        return plot_data


if __name__ == "__main__":
    from PyMimircache import *

    reader = VscsiReader("../../data/trace.vscsi")
    ph = PyHeatmap()
    # ph.heatmap(reader, "r", "hr_st_et", time_interval=200 * 1000000, cache_size=2000)
    ph.heatmap(reader, "r", "KL_st_et", time_interval=20 * 1000000)
