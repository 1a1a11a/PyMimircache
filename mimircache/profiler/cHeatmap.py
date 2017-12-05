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

from mimircache.const import CExtensionMode, DEFAULT_BIN_NUM_PROFILER
if CExtensionMode:
    import mimircache.c_heatmap
from mimircache import const
from mimircache.utils.printing import *
from mimircache.profiler.utilProfiler import set_fig




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
    def get_breakpoints(reader, time_mode, time_interval=-1, num_of_pixel_of_time_dim=-1, **kwargs):
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
        return mimircache.c_heatmap.get_breakpoints(reader.cReader, time_mode=time_mode,
                                                    time_interval=time_interval,
                                                    num_of_pixel_of_time_dim=num_of_pixel_of_time_dim)

    def set_plot_params(self, axis, axis_type, **kwargs):
        """
        this function sets the label and ticks for x and y axis

        :param axis:
        :param axis_type:
        :param kwargs:
        :return:
        """
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
                self.other_plot_kwargs['norm'] = colors.LogNorm()
        else:
            print("unsupported axis: " + str(axis))

        if axis_type == 'virtual time':
            if 'label' in kwargs:
                label = kwargs['label']
            else:
                label = 'virtual time'
            # tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(bin_size * x))
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100 / (len(array) - 1)))
        elif axis_type == 'real time':
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

        elif axis_type == 'reuse_dist' or axis_type == "distance":
            assert log_base != 1, "please provide log_base"
            label = 'reuse distance'
            tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(log_base ** x-1))

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

    def heatmap(self, reader, time_mode, plot_type, algorithm="LRU", time_interval=-1, num_of_pixel_of_time_dim=-1,
                cache_params=None, **kwargs):
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

        if time_mode == 'r':
            mode_string = "real time"
        else:
            mode_string = "virtual time"

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

            if algorithm.lower() in const.c_available_cache:
                xydict = mimircache.c_heatmap.heatmap(reader.cReader, time_mode, plot_type,
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


            # if time_mode == 'r':
            #     self.set_plot_params('x', mode_string, xydict=xydict, label='start time (real)',
            #                          text=(x1, y1, text))
            #     self.set_plot_params('y', mode_string, xydict=xydict, label='end time (real)', fixed_range=(0, 1))
            # else:
            #     self.set_plot_params('x', mode_string, xydict=xydict, label='start time (virtual)',
            #                          text=(x1, y1, text))
            #     self.set_plot_params('y', mode_string, xydict=xydict, label='end time (virtual)',
            #                          fixed_range=(0, 1))

            if not figname:
                figname = '_'.join([algorithm, str(cache_size), plot_type]) + '.png'
            self.draw_heatmap(xydict, figname=figname, fixed_range=(0, 1),
                              xlabel=xlabel, ylabel=ylabel, xticks=xticks, yticks=yticks)


        elif plot_type == "hr_interval_size":
            assert cache_size != -1, "please provide cache_size parameter for plotting hr_st_et"
            ewma_coefficient = kwargs.get("ewma_coefficient", DEFAULT_EWMA_COEFFICIENT)
            bin_size = kwargs.get("bin_size", cache_size//DEFAULT_BIN_NUM_PROFILER + 1)

            xydict = mimircache.c_heatmap.heatmap(reader.cReader, time_mode, plot_type,
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

            print(xydict.shape)
            print_list(xydict[:8, 0])
            print_list(xydict[:8, 1])


            if not figname:
                figname = '_'.join([algorithm, str(cache_size), plot_type]) + '.png'
            self.draw_heatmap(xydict, no_mask=True, figname=figname, fixed_range=(0, 1),
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

            xydict, log_base = mimircache.c_heatmap.heatmap_rd_distribution(reader.cReader, time_mode,
                                                                            time_interval=time_interval,
                                                                            num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                                            num_of_threads=num_of_threads)

            if 'filter_rd' in kwargs:
                # make reuse distance < filter_rd unvisible,
                # this can be useful when the low-reuse-distance request dominates
                # thus makes the heatmap hard to read
                assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
                index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
                xydict[:index_pos+1, :] = 0


            self.set_plot_params('x', mode_string, xydict=xydict)
            self.set_plot_params('y', 'reuse_dist', xydict=xydict, log_base=log_base)
            self.set_plot_params('cb', 'count') #, fixed_range=(0.01, 1))
            self.draw_heatmap(xydict, figname=figname, no_mask=True)

        elif plot_type == "rd_distribution_CDF":
            if not figname:
                figname = 'rd_distribution_CDF.png'

            xydict, log_base = mimircache.c_heatmap.heatmap_rd_distribution(reader.cReader, time_mode,
                                                                            time_interval=time_interval,
                                                                            num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                                            num_of_threads=num_of_threads, CDF=1)
            self.set_plot_params('x', mode_string, xydict=xydict)
            self.set_plot_params('y', 'reuse_dist', xydict=xydict, log_base=log_base)
            self.set_plot_params('cb', 'count') #, fixed_range=(0.01, 1))
            self.draw_heatmap(xydict, figname=figname, no_mask=True)


        elif plot_type == "future_rd_distribution":
            if not figname:
                figname = 'future_rd_distribution.png'

            xydict, log_base = mimircache.c_heatmap.heatmap_future_rd_distribution(reader.cReader, time_mode,
                                                                                   time_interval=time_interval,
                                                                                   num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                                                   num_of_threads=num_of_threads)
            self.set_plot_params('x', mode_string, xydict=xydict)
            self.set_plot_params('y', 'reuse_dist', xydict=xydict, log_base=log_base)
            self.set_plot_params('cb', 'count') #, fixed_range=(0.01, 1))
            self.draw_heatmap(xydict, figname=figname, no_mask=True)

        elif plot_type == "dist_distribution":
            if not figname:
                figname = 'dist_distribution.png'

            xydict, log_base = mimircache.c_heatmap.heatmap_dist_distribution(reader.cReader, time_mode,
                                                                              time_interval=time_interval,
                                                                              num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                                              num_of_threads=num_of_threads)

            if 'filter_rd' in kwargs:
                assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
                index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
                xydict[:index_pos+1, :] = 0


            self.set_plot_params('x', mode_string, xydict=xydict)
            self.set_plot_params('y', 'distance', xydict=xydict, log_base=log_base)
            self.set_plot_params('cb', 'count') #, fixed_range=(0.01, 1))
            self.draw_heatmap(xydict, figname=figname, no_mask=True)

        elif plot_type == "reuse_time_distribution":
            if not figname:
                figname = 'rt_distribution.png'

            xydict, log_base = mimircache.c_heatmap.heatmap_reuse_time_distribution(reader.cReader, time_mode,
                                                                                    time_interval=time_interval,
                                                                                    num_of_pixel_of_time_dim=num_of_pixel_of_time_dim,
                                                                                    num_of_threads=num_of_threads)

            if 'filter_rd' in kwargs:
                assert kwargs['filter_rd'] > 0, "filter_rd must be positive"
                index_pos = int(np.log(kwargs['filter_rd'])/np.log(log_base))
                xydict[:index_pos+1, :] = 0


            self.set_plot_params('x', mode_string, xydict=xydict)
            self.set_plot_params('y', 'distance', xydict=xydict, log_base=log_base)
            plt.ylabel("reuse time")
            self.set_plot_params('cb', 'count') #, fixed_range=(0.01, 1))
            self.draw_heatmap(xydict, figname=figname, no_mask=True)



        else:
            raise RuntimeError("Cannot recognize this plot type, "
                "please check documentation, yoru input is {}".format(plot_type))


        reader.reset()

    def diffHeatmap(self, reader, time_mode, plot_type, algorithm1,
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

        # self.time_interval = time_interval
        # self.num_of_pixel_of_time_dim = num_of_pixel_of_time_dim
        # self.time_mode = time_mode

        reader.reset()

        cache_size = kwargs.get("cache_size", -1)
        figname = kwargs.get("figname", '_'.join([algorithm2 + '-' + algorithm1, str(cache_size), plot_type]) + '.png')
        num_of_threads = kwargs.get("num_of_threads", os.cpu_count())
        assert time_mode in ["r", "v"], "Cannot recognize time_mode {}, it can only be either " \
                                        "real time(r) or virtual time(v)".format(time_mode)

        if time_mode == 'r':
            mode_string = "real time"
        else:
            mode_string = "virtual time"

        if plot_type == "hit_ratio_start_time_end_time" or plot_type == "hr_st_et":
            assert cache_size != -1, "please provide cache_size for plotting hr_st_et"

            xydict = mimircache.c_heatmap.diff_heatmap(reader.cReader, time_mode,
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

            if time_mode == 'r':
                self.set_plot_params('x', mode_string, xydict=xydict, label='start time (real)',
                                     text=(x1, y1, text))
                self.set_plot_params('y', mode_string, xydict=xydict, label='end time (real)', fixed_range=(-1, 1))
            else:
                self.set_plot_params('x', mode_string, xydict=xydict, label='start time (virtual)',
                                     text=(x1, y1, text))
                self.set_plot_params('y', mode_string, xydict=xydict, label='end time (virtual)',
                                     fixed_range=(-1, 1))

            title = "Differential Heatmap"
            xlabel = 'Start Time ({})'.format(time_mode)
            xticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ylabel = "End Time ({})".format(time_mode)
            yticks = ticker.FuncFormatter(lambda x, _: '{:.0%}'.format(x / (xydict.shape[1]-1)))
            ax = plt.gca()
            ax.text(x1, y1, text)  # , fontsize=20)  , color='blue')


            self.draw_heatmap(xydict, figname=figname)



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


    def draw_heatmap(self, xydict, **kwargs):
        filename = kwargs.get("figname", 'heatmap.png')

        if 'no_mask' in kwargs and kwargs['no_mask']:
            plot_array = xydict
        else:
            plot_array = np.ma.array(xydict, mask=np.tri(len(xydict), dtype=int).T)

        if 'filter_count' in kwargs:
            assert kwargs['filter_count'] > 0, "filter_count must be positive"
            plot_array = np.ma.array(xydict, mask=(xydict<kwargs['filter_count']))

        cmap = plt.cm.jet
        # cmap = plt.get_cmap("Oranges")
        cmap.set_bad('w', 1.)

        if 1:
        # try:
            if kwargs.get('fixed_range', False):
                vmin, vmax = kwargs['fixed_range']
                img = plt.imshow(plot_array, vmin=vmin, vmax=vmax,
                                 interpolation='nearest', origin='lower',
                                 cmap=cmap, aspect='auto')
            else:
                img = plt.imshow(plot_array, interpolation='nearest',
                                 origin='lower', aspect='auto', cmap=cmap)

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

        # except Exception as e:
        #     try:
        #         import time
        #         t = int(time.time())
        #         WARNING("plotting using imshow failed: {}, "
        #                 "now try to save the plotting data to /tmp/heatmap.{}.pickle".format(e, t))
        #         import pickle
        #         with open("/tmp/heatmap.{}.pickle", 'wb') as ofile:
        #             pickle.dump(plot_array, ofile)
        #     except Exception as e:
        #         WARNING("failed to save plotting data")
        #
        #     try:
        #         plt.pcolormesh(plot_array.T, cmap=cmap)
        #         plt.savefig(filename)
        #     except Exception as e:
        #         WARNING("further plotting using pcolormesh failed" + str(e))
