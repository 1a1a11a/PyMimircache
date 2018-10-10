# coding=utf-8
"""
This module provides all the heatmap related plotting


TODO: refactoring

Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""


import os
import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from PyMimircache.const import ALLOW_C_MIMIRCACHE, DEF_NUM_BIN_PROF, DEF_EMA_HISTORY_WEIGHT, INSTALL_PHASE
if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.Heatmap as c_heatmap
else:
    if not INSTALL_PHASE:
        print("CMimircache is not successfully installed, modules beginning with C are all disabled")

from PyMimircache import const
from PyMimircache.utils.printing import *
from PyMimircache.profiler.profilerUtils import set_fig, draw_heatmap




class CHeatmap:
    """
    heatmap class for plotting heatmaps in C
    """

    all = ["heatmap", "diff_heatmap"]


    def __init__(self, **kwargs):
        """ nothing needs to be done, not using all static method due to consistency between profilers """
        pass


    @staticmethod
    def get_breakpoints(reader, time_mode,
                        time_interval=-1,
                        num_of_pixel_of_time_dim=-1,
                        **kwargs):
        """
        retrieve the breakpoints given time_mode and time_interval or num_of_pixel_of_time_dim,
        break point breaks the trace into chunks of given time_interval

        :param reader: reader for reading trace
        :param time_mode: either real time (r) or virtual time (v)
        :param time_interval: the intended time_interval of data chunk
        :param num_of_pixel_of_time_dim: the number of chunks, this is used when it is hard to estimate time_interval,
                                you only need specify one, either num_of_pixel_of_time_dim or time_interval
        :param kwargs: not used now
        :return: a numpy list of break points begin with 0, ends with total_num_requests
        """

        assert time_interval != -1 or num_of_pixel_of_time_dim != -1, \
            "please provide at least one parameter, time_interval or num_of_pixel_of_time_dim"
        return c_heatmap.get_breakpoints(reader.c_reader, time_mode=time_mode,
                                                    time_interval=time_interval,
                                                    num_of_pixel_of_time_dim=num_of_pixel_of_time_dim)


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
        :param kwargs: include num_of_threads, figname, enable_ihr, ema_coef(default: 0.8), info_on_fig
        :return:


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

        # assert time_interval != -1 or num_of_pixel_of_time_dim != -1, \
        #     "please provide at least one parameter, time_interval or num_of_pixel_of_time_dim"
        reader.reset()

        cache_size = kwargs.get("cache_size", -1)
        figname = kwargs.get("figname", "CHeatmap_{}.png".format(plot_type))
        num_of_threads = kwargs.get("num_of_threads", os.cpu_count())
        if cache_params is None: cache_params = {}
        plot_kwargs = kwargs.get("plot_kwargs", {"figname": figname})
        assert time_mode in ["r", "v"], "Cannot recognize this time_mode, "\
                                        "it can only be either real time(r) or virtual time(v), " \
                                        "but you give {}".format(time_mode)
        assert algorithm.lower() in const.C_AVAIL_CACHE, \
            "No support for given algorithm in CMimircache, please try PyHeatmap" + str(algorithm)


        if plot_type == "hit_ratio_start_time_end_time" or plot_type == "hr_st_et":
            assert cache_size != -1, "please provide cache_size parameter for plotting hr_st_et"
            # this is used to specify the type of hit ratio for each pixel, when it is False, it means
            # the hit ratio is the overall hit ratio from the beginning of trace,
            # if True, then the hit ratio will be exponentially decayed moving average hit ratio from beginning plus
            # the hit ratio in current time interval, the ewma_coefficient specifies the ratio of
            # history hit ratio in the calculation
            enable_ihr = False
            ema_coef  = DEF_EMA_HISTORY_WEIGHT
            if 'interval_hit_ratio'in kwargs or "enable_ihr" in kwargs:
                enable_ihr = kwargs.get("interval_hit_ratio", False) or kwargs.get("enable_ihr", False)
                ema_coef  = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)

            xydict = c_heatmap.heatmap(reader.c_reader, time_mode, plot_type,
                                                  cache_size, algorithm,
                                                  interval_hit_ratio=enable_ihr,
                                                  ewma_coefficient=ema_coef,
                                                  time_interval=time_interval,
                                                  num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                  cache_params=cache_params,
                                                  num_of_threads=num_of_threads)


            text = "cache size: {},\ncache type: {},\n" \
                   "time type:  {},\ntime interval: {},\n" \
                   "plot type: {}".format(cache_size, algorithm, time_mode, time_interval, plot_type)

            # coordinate to put the text
            x1, y1 = xydict.shape
            x1, y1 = int(x1 / 2), y1 / 8

            xlabel = 'Start Time ({})'.format(time_mode)
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "End Time ({})".format(time_mode)
            yticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            # yticks = ticker.FuncFormatter(lambda x, _: int(x*bin_size))
            ax = plt.gca()
            if kwargs.get("info_on_fig", True):
                ax.text(x1, y1, text)  # , fontsize=20)  , color='blue')

            plot_data = np.ma.array(xydict, mask=np.tri(len(xydict), dtype=int).T)
            self.draw_heatmap(plot_data, figname=figname, fixed_range=(0, 1),
                              xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks)


        elif plot_type == "hit_ratio_start_time_cache_size" or plot_type == "hr_st_size":
            assert cache_size != -1, "please provide cache_size parameter for plotting ihr_st_size"

            enable_ihr = False
            ema_coef  = DEF_EMA_HISTORY_WEIGHT
            bin_size = kwargs.get("bin_size", cache_size // DEF_NUM_BIN_PROF + 1)
            if 'interval_hit_ratio'in kwargs or "enable_ihr" in kwargs:
                enable_ihr = kwargs.get("interval_hit_ratio", False) or kwargs.get("enable_ihr", False)
                ema_coef  = kwargs.get("ema_coef", DEF_EMA_HISTORY_WEIGHT)

            # a temporary fix as hr_st_size is not implemented in C, only hr_st_size with enable_ihr

            if not enable_ihr:
                raise RuntimeError("NOT Implemented")

            else:
                plot_type = "hr_interval_size"


            xydict = c_heatmap.heatmap(reader.c_reader, time_mode, plot_type,
                                                  cache_size, algorithm,
                                                  ewma_coefficient=ema_coef,
                                                  bin_size = bin_size,
                                                  time_interval=time_interval,
                                                  num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                  cache_params=cache_params,
                                                  num_of_threads=num_of_threads)

            xlabel = 'Time ({})'.format(time_mode)
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Cache Size"
            yticks = ticker.FuncFormatter(lambda x, _: int(x*bin_size))

            plot_data = xydict
            self.draw_heatmap(plot_data, figname=figname, fixed_range=(0, 1),
                              xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks)



        elif plot_type == "effective_size":
            assert cache_size != -1, "please provide cache_size parameter for plotting effective_size"

            bin_size = kwargs.get("bin_size", cache_size // DEF_NUM_BIN_PROF + 1)
            use_percent = kwargs.get("use_percent", False)
            xydict = c_heatmap.heatmap(reader.c_reader, time_mode, plot_type,
                                                  cache_size, algorithm,
                                                  bin_size = bin_size,
                                                  time_interval=time_interval,
                                                  num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                  cache_params=None,
                                                  use_percent=use_percent,
                                                  num_of_threads=num_of_threads)

            plot_kwargs["xlabel"]  = plot_kwargs.get("xlabel", 'Time ({})'.format(time_mode))
            plot_kwargs["xticks"]  = plot_kwargs.get("xticks", ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1))))
            plot_kwargs["ylabel"]  = plot_kwargs.get("ylabel", "Cache Size")
            plot_kwargs["yticks"]  = plot_kwargs.get("yticks", ticker.FuncFormatter(lambda x, _: int(x*bin_size)))
            plot_kwargs["imshow_kwargs"] = {"cmap": "Oranges"}
            if use_percent:
                plot_kwargs["imshow_kwargs"].update({"vmin": 0, "vmax":1})

            plot_data = xydict
            draw_heatmap(plot_data, **plot_kwargs)



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



            title = "Reuse Distance Distribution Heatmap"
            xlabel = '{} Time'.format("Real" if time_mode=="r" else "Virtual")
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Reuse Distance"
            yticks = ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1))
            imshow_kwargs = {"norm": colors.LogNorm()}

            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)

            if 'filter_rd' in kwargs:
                # make reuse distance < filter_rd invisible,
                # this can be useful when the low-reuse-distance request dominates
                # thus makes the heatmap hard to read
                assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
                index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
                plot_data[:index_pos+1, :] = 0


            self.draw_heatmap(plot_data, figname=figname, xlabel=xlabel, ylabel=ylabel,
                              xticks=xticks, yticks=yticks, title=title,
                              imshow_kwargs=imshow_kwargs)


        elif plot_type == "rd_distribution_CDF":
            xydict, log_base = c_heatmap.\
                hm_rd_distribution(reader.c_reader, time_mode,
                                   time_interval=time_interval,
                                   num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                   num_of_threads=num_of_threads, CDF=1)

            title = "Reuse Distance Distribution CDF Heatmap"
            xlabel = '{} Time'.format("Real" if time_mode=="r" else "Virtual")
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Reuse Distance (CDF)"
            yticks = ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1))
            imshow_kwargs = {"norm": colors.LogNorm()}

            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)

            if 'filter_rd' in kwargs:
                # make reuse distance < filter_rd invisible,
                # this can be useful when the low-reuse-distance request dominates
                # thus makes the heatmap hard to read
                assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
                index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
                plot_data[:index_pos+1, :] = 0


            self.draw_heatmap(plot_data, figname=figname, xlabel=xlabel, ylabel=ylabel,
                              xticks=xticks, yticks=yticks, title=title,
                              imshow_kwargs=imshow_kwargs)


        elif plot_type == "future_rd_distribution":
            xydict, log_base = c_heatmap.\
                hm_future_rd_distribution(reader.c_reader, time_mode,
                                          time_interval=time_interval,
                                          num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                          num_of_threads=num_of_threads)

            title = "Future Reuse Distance Distribution Heatmap"
            xlabel = '{} Time'.format("Real" if time_mode=="r" else "Virtual")
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Future Reuse Distance"
            yticks = ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1))
            imshow_kwargs = {"norm": colors.LogNorm()}

            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)

            if 'filter_rd' in kwargs:
                # make reuse distance < filter_rd invisible,
                # this can be useful when the low-reuse-distance request dominates
                # thus makes the heatmap hard to read
                assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
                index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
                plot_data[:index_pos+1, :] = 0


            self.draw_heatmap(plot_data, figname=figname, xlabel=xlabel, ylabel=ylabel,
                              xticks=xticks, yticks=yticks, title=title,
                              imshow_kwargs=imshow_kwargs)

        elif plot_type == "dist_distribution":
            xydict, log_base = c_heatmap.\
                hm_dist_distribution(reader.c_reader, time_mode,
                                     time_interval=time_interval,
                                     num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                     num_of_threads=num_of_threads)

            title = "Distance Distribution Heatmap"
            xlabel = '{} Time'.format("Real" if time_mode=="r" else "Virtual")
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Distance"
            yticks = ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1))
            imshow_kwargs = {"norm": colors.LogNorm()}

            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)

            if 'filter_dist' in kwargs:
                # make reuse distance < filter_rd invisible,
                # this can be useful when the low-reuse-distance request dominates
                # thus makes the heatmap hard to read
                assert kwargs['filter_dist'] > 0, "filter_dist must be positive"
                index_pos = int(np.log(kwargs['filter_dist'])/np.log(log_base))
                plot_data[:index_pos+1, :] = 0


            self.draw_heatmap(plot_data, figname=figname, xlabel=xlabel, ylabel=ylabel,
                              xticks=xticks, yticks=yticks, title=title,
                              imshow_kwargs=imshow_kwargs)


        elif plot_type == "reuse_time_distribution" or plot_type == "rt_distribution":
            xydict, log_base = c_heatmap.\
                hm_reuse_time_distribution(reader.c_reader, time_mode,
                                           time_interval=time_interval,
                                           num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                           num_of_threads=num_of_threads)

            title = "Reuse Time Distribution Heatmap"
            xlabel = '{} Time'.format("Real" if time_mode=="r" else "Virtual")
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Reuse Time"
            yticks = ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(log_base ** x-1))
            imshow_kwargs = {"norm": colors.LogNorm()}

            plot_data = xydict
            np.ma.masked_where(plot_data==0, plot_data, copy=False)

            if 'filter_rt' in kwargs:
                # make reuse distance < filter_rd invisible,
                # this can be useful when the low-reuse-distance request dominates
                # thus makes the heatmap hard to read
                assert kwargs['filter_rt'] > 0, "filter_rt must be positive"
                index_pos = int(np.log(kwargs['filter_rt'])/np.log(log_base))
                plot_data[:index_pos+1, :] = 0


            self.draw_heatmap(plot_data, figname=figname, xlabel=xlabel, ylabel=ylabel,
                              xticks=xticks, yticks=yticks, title=title,
                              imshow_kwargs=imshow_kwargs)

        else:
            raise RuntimeError("Cannot recognize this plot type, "
                "please check documentation, yoru input is {}".format(plot_type))


        reader.reset()

    def diff_heatmap(self, reader, time_mode, plot_type, algorithm1,
                     time_interval=-1, num_of_pixel_of_time_dim=-1, algorithm2="Optimal",
                     cache_params1=None, cache_params2=None, **kwargs):
        """

        :param time_interval:
        :param num_of_pixel_of_time_dim:
        :param algorithm2:
        :param cache_params1:
        :param cache_params2:
        :param algorithm1:
        :param plot_type:
        :param time_mode:
        :param reader:
        :param kwargs: include num_of_process, figname
        :return:
        """

        reader.reset()

        cache_size = kwargs.get("cache_size", -1)
        figname = kwargs.get("figname", '_'.join([algorithm2 + '-' + algorithm1, str(cache_size), plot_type]) + '.png')
        num_of_threads = kwargs.get("num_of_threads", os.cpu_count())
        assert time_mode in ["r", "v"], "Cannot recognize time_mode {}, it can only be either " \
                                        "real time(r) or virtual time(v)".format(time_mode)

        if plot_type == "hit_ratio_start_time_end_time" or plot_type == "hr_st_et":
            assert cache_size != -1, "please provide cache_size for plotting hr_st_et"

            xydict = c_heatmap.diff_heatmap(reader.c_reader, time_mode,
                                                       "hr_st_et", cache_size,
                                                       algorithm1, algorithm2,
                                                       time_interval=time_interval,
                                                       num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                       cache_params1=cache_params1,
                                                       cache_params2=cache_params2,
                                                       num_of_threads=num_of_threads)

            text = "differential heatmap\ncache size: {},\ncache type: ({}-{})/{},\n" \
                   "time type: {},\ntime interval: {},\nplot type: {}".format(
                cache_size, algorithm2, algorithm1, algorithm1, time_mode, time_interval, plot_type)

            x1, y1 = xydict.shape
            x1, y1 = int(x1 / 2.4), y1/8

            title = "Differential Heatmap"
            xlabel = 'Start Time ({})'.format(time_mode)
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "End Time ({})".format(time_mode)
            yticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ax = plt.gca()
            ax.text(x1, y1, text)


            self.draw_heatmap(xydict, figname=figname, xlabel=xlabel, ylabel=ylabel,
                              xticks=xticks, yticks=yticks, title=title, fixed_range=(-1, 1))



        elif plot_type == "hit_ratio_start_time_cache_size":
            pass


        elif plot_type == "avg_rd_start_time_end_time":
            pass


        elif plot_type == "cold_miss_count_start_time_end_time":
            print("this plot is discontinued")


        elif plot_type == "???":
            pass



        else:
            raise RuntimeError(
                "Cannot recognize this plot type, please check documentation, yoru input is %s" % time_mode)

        reader.reset()


    def draw_heatmap(self, plot_array, **kwargs):
        filename = kwargs.get("figname", 'heatmap.png')

        if 'filter_count' in kwargs:
            assert kwargs['filter_count'] > 0, "filter_count must be positive"
            plot_array = np.ma.array(plot_array, mask=(plot_array<kwargs['filter_count']))

        imshow_kwargs = kwargs.get("imshow_kwargs", {})

        if "cmap" not in imshow_kwargs:
            imshow_kwargs["cmap"] = plt.cm.jet
        else:
            imshow_kwargs["cmap"] = plt.get_cmap(imshow_kwargs["cmap"])
        # cmap = plt.get_cmap("Oranges")
        imshow_kwargs["cmap"].set_bad(color='white', alpha=1.)

        # if 1:
        try:
            if kwargs.get('fixed_range', False):
                vmin, vmax = kwargs['fixed_range']
                img = plt.imshow(plot_array, vmin=vmin, vmax=vmax,
                                 interpolation='nearest', origin='lower',
                                 aspect='auto', **imshow_kwargs)
            else:
                img = plt.imshow(plot_array, interpolation='nearest',
                                 origin='lower', aspect='auto', **imshow_kwargs)

            cb = plt.colorbar(img)
            set_fig(no_legend=True, **kwargs)

            if not kwargs.get("no_save", False):
                plt.savefig(filename, dpi=600)
                INFO("plot is saved as {}".format(filename))
            if not kwargs.get("no_show", False):
                try: plt.show()
                except: pass
            if not kwargs.get("no_clear", False):
                try: plt.clf()
                except: pass

        except Exception as e:
            try:
                import time
                t = int(time.time())
                WARNING("plotting using imshow failed: {}, "
                        "now try to save the plotting data to /tmp/heatmap.{}.pickle".format(e, t))
                import pickle
                with open("/tmp/heatmap.{}.pickle".format(t), 'wb') as ofile:
                    pickle.dump(plot_array, ofile)
            except Exception as e:
                ERROR("failed to save plotting data")

            try:
                plt.pcolormesh(plot_array.T)
                plt.savefig(filename)
            except Exception as e:
                WARNING("further plotting using pcolormesh failed" + str(e))
