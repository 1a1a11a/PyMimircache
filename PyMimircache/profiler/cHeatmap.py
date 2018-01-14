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

from PyMimircache.const import ALLOW_C_MIMIRCACHE, DEF_NUM_BIN_PROF
if ALLOW_C_MIMIRCACHE:
    import PyMimircache.CMimircache.Heatmap as c_heatmap
from PyMimircache import const
from PyMimircache.utils.printing import *
from PyMimircache.profiler.utilProfiler import set_fig




class CHeatmap:
    """
    heatmap class for plotting heatmaps
    """

    all = ["heatmap", "diff_heatmap"]


    def __init__(self, **kwargs):
        if "reader" in kwargs:
            self.reader = kwargs["reader"]
            raise RuntimeWarning("reader set is not enabled")

        self.other_plot_kwargs = {}

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
        return c_heatmap.get_breakpoints(reader.cReader, time_mode=time_mode,
                                                    time_interval=time_interval,
                                                    num_of_pixel_of_time_dim=num_of_pixel_of_time_dim)


    def heatmap(self, reader, time_mode, plot_type,
                algorithm="LRU",
                time_interval=-1,
                num_of_pixel_of_time_dim=-1,
                cache_params=None,
                **kwargs):
        """

        :param cache_params:
        :param num_of_pixel_of_time_dim:
        :param time_interval:
        :param algorithm:
        :param plot_type:
        :param time_mode:
        :param reader:
        :param kwargs: include num_of_threads, figname, ewma_coefficient (default: 0.2)
        :return:
        """

        DEFAULT_EWMA_COEFFICIENT = 0.2
        # assert time_interval != -1 or num_of_pixel_of_time_dim != -1, \
        #     "please provide at least one parameter, time_interval or num_of_pixel_of_time_dim"
        reader.reset()

        cache_size = kwargs.get("cache_size", -1)
        figname = kwargs.get("figname", "heatmap.png")
        num_of_threads = kwargs.get("num_of_threads", os.cpu_count())
        assert time_mode in ["r", "v"], "Cannot recognize this time_mode, "\
                                        "it can only be either real time(r) or virtual time(v), " \
                                        "but you give {}".format(time_mode)


        if plot_type == "hit_ratio_start_time_end_time" or plot_type == "hr_st_et":
            assert cache_size != -1, "please provide cache_size parameter for plotting hr_st_et"
            # this is used to specify the type of hit ratio for each pixel, when it is False, it means
            # the hit ratio is the overall hit ratio from the beginning of trace,
            # if True, then the hit ratio will be exponentially decayed moving average hit ratio from beginning plus
            # the hit ratio in current time interval, the ewma_coefficient specifies the ratio of
            # history hit ratio in the calculation
            enable_ihr = False
            ewma_coefficient  = DEFAULT_EWMA_COEFFICIENT
            if 'interval_hit_ratio'in kwargs or "enable_ihr" in kwargs:
                enable_ihr = kwargs.get("interval_hit_ratio", False) or kwargs.get("enable_ihr", False)
                ewma_coefficient  = kwargs.get("ewma_coefficient", DEFAULT_EWMA_COEFFICIENT)

            if algorithm.lower() in const.C_AVAIL_CACHE:
                xydict = c_heatmap.heatmap(reader.cReader, time_mode, plot_type,
                                                      cache_size, algorithm,
                                                      interval_hit_ratio=enable_ihr,
                                                      ewma_coefficient=ewma_coefficient,
                                                      time_interval=time_interval,
                                                      num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                      cache_params=cache_params,
                                                      num_of_threads=num_of_threads)
            else:
                raise RuntimeError("haven't provided support given algorithm in C yet: " + str(algorithm))
                pass

            text = "      " \
                   "cache size: {},\n      cache type: {},\n      " \
                   "time type:  {},\n      time interval: {},\n      " \
                   "plot type: \n{}".format(cache_size,
                                            algorithm, time_mode, time_interval, plot_type)

            # coordinate to put the text
            x1, y1 = xydict.shape
            x1 = int(x1 / 2.8)
            y1 /= 8

            xlabel = 'Start Time ({})'.format(time_mode)
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "End Time ({})".format(time_mode)
            yticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            # yticks = ticker.FuncFormatter(lambda x, _: int(x*bin_size))
            ax = plt.gca()
            ax.text(x1, y1, text)  # , fontsize=20)  , color='blue')

            if not figname:
                figname = '_'.join([algorithm, str(cache_size), plot_type]) + '.png'

            plot_data = np.ma.array(xydict, mask=np.tri(len(xydict), dtype=int).T)
            self.draw_heatmap(plot_data, figname=figname, fixed_range=(0, 1),
                              xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks)


        elif plot_type == "hr_interval_size":
            assert cache_size != -1, "please provide cache_size parameter for plotting hr_st_et"
            ewma_coefficient = kwargs.get("ewma_coefficient", DEFAULT_EWMA_COEFFICIENT)
            bin_size = kwargs.get("bin_size", cache_size // DEF_NUM_BIN_PROF + 1)

            xydict = c_heatmap.heatmap(reader.cReader, time_mode, plot_type,
                                                  cache_size, algorithm,
                                                  ewma_coefficient=ewma_coefficient,
                                                  bin_size = bin_size,
                                                  time_interval=time_interval,
                                                  num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                  cache_params=cache_params,
                                                  num_of_threads=num_of_threads)

            xlabel = 'Time ({})'.format(time_mode)
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "Cache Size"
            yticks = ticker.FuncFormatter(lambda x, _: int(x*bin_size))
            title = "Interval Hit Ratio With Size (ewma {})".format(ewma_coefficient)

            plot_data = xydict

            if not figname:
                figname = '_'.join([algorithm, str(cache_size), plot_type]) + '.png'
            self.draw_heatmap(plot_data, figname=figname, fixed_range=(0, 1),
                              xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks, title=title)

        elif plot_type == "hit_ratio_start_time_cache_size" or plot_type == "hr_st_size":
            pass


        elif plot_type == "avg_rd_start_time_end_time" or plot_type == "avg_rd_st_et":
            pass


        elif plot_type == "cold_miss_count_start_time_end_time":
            raise RuntimeError("this plot is deprecated")


        elif plot_type == "???":
            pass




        elif plot_type == "rd_distribution":
            if not figname:
                figname = 'rd_distribution.png'

            xydict, log_base = c_heatmap.\
                hm_rd_distribution(reader.cReader, time_mode,
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
            if not figname:
                figname = 'rd_distribution_CDF.png'

            xydict, log_base = c_heatmap.\
                hm_rd_distribution(reader.cReader, time_mode,
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
            if not figname:
                figname = 'future_rd_distribution.png'

            xydict, log_base = c_heatmap.\
                hm_future_rd_distribution(reader.cReader, time_mode,
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
            if not figname:
                figname = 'dist_distribution.png'

            xydict, log_base = c_heatmap.\
                hm_dist_distribution(reader.cReader, time_mode,
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


        elif plot_type == "reuse_time_distribution":
            if not figname:
                figname = 'rt_distribution.png'

            xydict, log_base = c_heatmap.\
                hm_reuse_time_distribution(reader.cReader, time_mode,
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

            xydict = c_heatmap.diff_heatmap(reader.cReader, time_mode,
                                                       "hr_st_et", cache_size,
                                                       algorithm1, algorithm2,
                                                       time_interval=time_interval,
                                                       num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                       cache_params1=cache_params1,
                                                       cache_params2=cache_params2,
                                                       num_of_threads=num_of_threads)

            text = "      differential heatmap\n      cache size: {},\n      cache type: ({}-{})/{},\n" \
                   "      time type: {},\n      time interval: {},\n      plot type: \n{}".format(
                cache_size, algorithm2, algorithm1, algorithm1, time_mode, time_interval, plot_type)

            x1, y1 = xydict.shape
            x1 = int(x1 / 2.8)
            y1 /= 8

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
            plot_array = np.ma.array(xydict, mask=(plot_array<kwargs['filter_count']))

        imshow_kwargs = kwargs.get("imshow_kwargs", {})

        cmap = plt.cm.jet
        # cmap = plt.get_cmap("Oranges")
        cmap.set_bad(color='white', alpha=1.)

        # if 1:
        try:
            if kwargs.get('fixed_range', False):
                vmin, vmax = kwargs['fixed_range']
                img = plt.imshow(plot_array, vmin=vmin, vmax=vmax,
                                 interpolation='nearest', origin='lower',
                                 cmap=cmap, aspect='auto', **imshow_kwargs)
            else:
                img = plt.imshow(plot_array, interpolation='nearest',
                                 origin='lower', aspect='auto', cmap=cmap, **imshow_kwargs)

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
                with open("/tmp/heatmap.{}.pickle", 'wb') as ofile:
                    pickle.dump(plot_array, ofile)
            except Exception as e:
                WARNING("failed to save plotting data")

            try:
                plt.pcolormesh(plot_array.T, cmap=cmap)
                plt.savefig(filename)
            except Exception as e:
                WARNING("further plotting using pcolormesh failed" + str(e))
