# coding=utf-8

"""
this module offer the upper level API to user
"""
from matplotlib.ticker import FuncFormatter
import mimircache.c_heatmap as c_heatmap
from mimircache.profiler.evictionStat import *
from mimircache.profiler.twoDPlots import *
from mimircache.utils.prepPlotParams import *


class cachecow:
    all = ["open", "csv", "vscsi", "binary", "set_size", "num_of_req", "num_of_uniq_req",
           "heatmap", "diffHeatmap", "twoDPlot", "eviction_plot", "plotHRCs", "plotMRCs" "close"]
    def __init__(self, **kwargs):
        self.reader = None
        self.cache_size = 0
        self.n_req = -1
        self.n_uniq_req = -1
        self.cacheclass_mapping = {}

    def open(self, file_path, data_type='c'):
        """
        open a plain text file, which contains a label each line
        :param file_path:
        :param data_type: can be either 'c' for string or 'l' for number (like block IO)
        :return:
        """
        if self.reader:
            self.reader.close()
        self.reader = plainReader(file_path, data_type=data_type)
        return self.reader

    def csv(self, file_path, init_params, data_type='c', block_unit_size=0, disk_sector_size=0):
        """
        open a csv file
        :param file_path:
        :param init_params: params related to csv file, see csvReader for detail
        :param data_type: can be either 'c' for string or 'l' for number (like block IO)
        :param block_unit_size: the page size for a cache 
        :param disk_sector_size: the disk sector size of input file 
        :return:
        """
        if self.reader:
            self.reader.close()
        self.reader = csvReader(file_path, data_type=data_type,
                                block_unit_size=block_unit_size,
                                disk_sector_size=disk_sector_size,
                                init_params=init_params)
        return self.reader

    def binary(self, file_path, init_params, data_type='l', block_unit_size=0, disk_sector_size=0):
        """
        open a binary file
        :param file_path:
        :param init_params: params related to csv file, see csvReader for detail
        :param data_type: can be either 'c' for string or 'l' for number (like block IO)
        :param block_unit_size: the page size for a cache 
        :param disk_sector_size: the disk sector size of input file 
        :return:
        """
        if self.reader:
            self.reader.close()
        self.reader = binaryReader(file_path, data_type=data_type,
                                   block_unit_size=block_unit_size,
                                   disk_sector_size=disk_sector_size,
                                   init_params=init_params)
        return self.reader

    def vscsi(self, file_path, data_type='l', block_unit_size=0, disk_sector_size=512):
        """
        open vscsi trace file
        :param file_path:
        :param data_type: can be either 'c' for string or 'l' for number (like block IO)
        :param block_unit_size: the page size for a cache 
        :param disk_sector_size: the disk sector size of input file 
        :return:
        """
        if self.reader:
            self.reader.close()
        self.reader = vscsiReader(file_path, data_type=data_type,
                                  block_unit_size=block_unit_size,
                                  disk_sector_size=disk_sector_size)
        return self.reader

    def set_size(self, size):
        """
        set the size of cachecow 
        :param size: 
        :return: 
        """
        assert isinstance(size, int), "size can only be an integer"
        self.cache_size = size

    def num_of_req(self):
        """
        return the number of requests in the trace
        :return:
        """
        if self.n_req == -1:
            self.n_req = self.reader.get_num_total_req()
        return self.n_req

    def num_of_uniq_req(self):
        """
        return the number of unique requests in the trace
        :return:
        """
        if self.n_uniq_req == -1:
            self.n_uniq_req = self.reader.get_num_unique_req()
        return self.n_uniq_req

    def reset(self):
        """
        reset reader to the beginning of the trace
        :return:
        """
        assert self.reader is not None, "reader is None, cannot reset"
        self.reader.reset()

    def _profiler_pre_check(self, **kwargs):
        """
        check whether user has provided new cache size and data information
        :param kwargs:
        :return:
        """
        reader = None

        if 'num_of_threads' in kwargs:
            num_of_threads = kwargs['num_of_threads']
        elif 'num_of_threads' in kwargs:
            num_of_threads = kwargs['num_of_threads']
        else:
            num_of_threads = DEFAULT_NUM_OF_THREADS

        if 'data' in kwargs and 'dataType' in kwargs:
            if kwargs['dataType'] == 'plain':
                reader = plainReader(kwargs['data'])
            if kwargs['dataType'] == 'csv':
                assert 'column' in kwargs, "you didn't provide column number for csv reader"
                reader = csvReader(kwargs['data'], kwargs['column'])
            if kwargs['dataType'] == 'vscsi':
                reader = vscsiReader(kwargs['data'])
        elif 'reader' in kwargs:
            reader = kwargs['reader']
        else:
            reader = self.reader

        assert reader is not None, "you didn't provide a reader nor data (data file and data type)"
        self.reader = reader

        return reader, num_of_threads

    def heatmap(self, mode, plot_type, time_interval=-1, num_of_pixels=-1,
                algorithm="LRU", cache_params=None, cache_size=-1, **kwargs):
        """

        :param cache_size:
        :param cache_params:
        :param algorithm:
        :param num_of_pixels:
        :param time_interval:
        :param plot_type:
        :param mode:
        :param kwargs: algorithm:
        :return:
        """

        reader, num_of_threads = self._profiler_pre_check(**kwargs)
        assert cache_size <= self.num_of_req(), "you cannot specify cache size({}) " \
                                                    "larger than trace length({})".format(cache_size,
                                                                                          self.num_of_req())

        l = ["avg_rd_start_time_end_time", "hit_rate_start_time_cache_size"]

        if plot_type in l:
            hm = heatmap()
        else:
            if algorithm.lower() in c_available_cache:
                hm = cHeatmap()

            else:
                hm = heatmap()

        hm.heatmap(reader, mode, plot_type,
                   time_interval=time_interval,
                   num_of_pixels=num_of_pixels,
                   cache_size=cache_size,
                   algorithm=cache_alg_mapping[algorithm.lower()],
                   cache_params=cache_params,
                   **kwargs)

    def diffHeatmap(self, mode, plot_type, algorithm1, time_interval=-1, num_of_pixels=-1,
                    algorithm2="Optimal", cache_params1=None, cache_params2=None, cache_size=-1, **kwargs):
        """
        alg2 - alg1
        :param cache_size:
        :param cache_params2:
        :param cache_params1:
        :param algorithm2:
        :param num_of_pixels:
        :param time_interval:
        :param algorithm1:
        :param mode:
        :param plot_type:
        :param kwargs:
        :return:
        """
        figname = 'differential_heatmap.png'
        if 'figname' in kwargs:
            figname = kwargs['figname']

        assert cache_size != -1, "you didn't provide size for cache"
        assert cache_size <= self.num_of_req(), "you cannot specify cache size({}) " \
                                                    "larger than trace length({})".format(cache_size,
                                                                                          self.num_of_req())

        reader, num_of_threads = self._profiler_pre_check(**kwargs)

        if algorithm1.lower() in c_available_cache and algorithm2.lower() in c_available_cache:
            hm = cHeatmap()
            hm.diffHeatmap(reader, mode, plot_type,
                           cache_size=cache_size,
                           time_interval=time_interval,
                           num_of_pixels=num_of_pixels,
                           algorithm1=cache_alg_mapping[algorithm1.lower()],
                           algorithm2=cache_alg_mapping[algorithm2.lower()],
                           cache_params1=cache_params1,
                           cache_params2=cache_params2,
                           **kwargs)

        else:
            hm = heatmap()
            if algorithm1.lower() not in c_available_cache:
                xydict1 = hm.calculate_heatmap_dat(reader, mode, plot_type,
                                                   time_interval=time_interval,
                                                   cache_size=cache_size,
                                                   algorithm=algorithm1,
                                                   cache_params=cache_params1,
                                                   **kwargs)[0]
            else:
                xydict1 = c_heatmap.heatmap(reader.cReader, mode, plot_type,
                                            cache_size=cache_size,
                                            time_interval=time_interval,
                                            algorithm=algorithm1,
                                            cache_params=cache_params1,
                                            num_of_threads=num_of_threads)

            if algorithm2.lower() not in c_available_cache:
                xydict2 = hm.calculate_heatmap_dat(reader, mode, plot_type,
                                                   time_interval=time_interval,
                                                   cache_size=cache_size,
                                                   algorithm=algorithm2,
                                                   cache_params=cache_params2,
                                                   **kwargs)[0]
            else:
                xydict2 = c_heatmap.heatmap(reader.cReader, mode, plot_type,
                                            time_interval=time_interval,
                                            cache_size=cache_size,
                                            algorithm=algorithm2,
                                            cache_params=cache_params2,
                                            num_of_threads=num_of_threads)

            cHm = cHeatmap()
            text = "      differential heatmap\n      cache size: {},\n      cache type: ({}-{})/{},\n" \
                   "      time type: {},\n      time interval: {},\n      plot type: \n{}".format(
                cache_size, algorithm2, algorithm1, algorithm1, mode, time_interval, plot_type)

            x1, y1 = xydict1.shape
            x1 = int(x1 / 2.8)
            y1 /= 8
            if mode == 'r':
                cHm.setPlotParams('x', 'real_time', xydict=xydict1, label='start time (real)',
                                  text=(x1, y1, text))
                cHm.setPlotParams('y', 'real_time', xydict=xydict1, label='end time (real)', fixed_range=(-1, 1))
            else:
                cHm.setPlotParams('x', 'virtual_time', xydict=xydict1, label='start time (virtual)',
                                  text=(x1, y1, text))
                cHm.setPlotParams('y', 'virtual_time', xydict=xydict1, label='end time (virtual)',
                                  fixed_range=(-1, 1))

            np.seterr(divide='ignore', invalid='ignore')

            plot_dict = (xydict2 - xydict1) / xydict1
            cHm.draw_heatmap(plot_dict, figname=figname)

    def profiler(self, algorithm, cache_params=None, cache_size=-1, use_general_profiler=False, **kwargs):
        """
        profiler
        :param cache_size:
        :param cache_params:
        :param algorithm:
        :param use_general_profiler: for LRU only, if it is true, then return a cGeneralProfiler for LRU,
                                        otherwise, return a LRUProfiler for LRU 
                                        Note: LRUProfiler provides does not require cache_size/bin_size params, 
                                        does not sample and provide a smooth curve, however, it is O(logN) at each step, 
                                        in constrast, cGeneralProfiler samples the curve, but use O(1) at each step 
        :param kwargs:
        :return:
        """

        reader, num_of_threads = self._profiler_pre_check(**kwargs)

        profiler = None
        bin_size = -1

        if algorithm.lower() == "lru" and not use_general_profiler:
            profiler = LRUProfiler(reader, cache_size, cache_params)
        else:
            assert cache_size != -1, "you didn't provide size for cache"
            assert cache_size <= self.num_of_req(), "you cannot specify cache size({}) " \
                                                        "larger than trace length({})".format(cache_size,
                                                                                              self.num_of_req())
            if 'bin_size' in kwargs:
                bin_size = kwargs['bin_size']

            if isinstance(algorithm, str):
                if algorithm.lower() in c_available_cache:
                    profiler = cGeneralProfiler(reader, cache_alg_mapping[algorithm.lower()],
                                                cache_size, bin_size,
                                                cache_params, num_of_threads)
                else:
                    profiler = generalProfiler(reader, self.cacheclass_mapping[algorithm.lower()],
                                               cache_size, bin_size,
                                               cache_params, num_of_threads)
            else:
                profiler = generalProfiler(reader, algorithm, cache_size, bin_size,
                                           cache_params, num_of_threads)

        return profiler

    def twoDPlot(self, plot_type, **kwargs):
        """
        two dimensional plots
        :param plot_type:
        :param kwargs:
        :return:
        """
        if "figname" in kwargs:
            figname = kwargs['figname']
        else:
            figname = plot_type + ".png"

        if plot_type == 'cold_miss' or plot_type == "cold_miss_count":
            if plot_type == 'cold_miss':
                print("please use cold_miss_count, cold_miss is deprecated")
            assert "mode" in kwargs, "you need to provide mode(r/v) for plotting cold_miss2d"
            assert "time_interval" in kwargs, "you need to provide time_interval for plotting cold_miss2d"
            cold_miss_count_2d(self.reader, kwargs['mode'], kwargs['time_interval'], figname=figname)

        elif plot_type == 'cold_miss_ratio':
            assert "mode" in kwargs, "you need to provide mode(r/v) for plotting cold_miss2d"
            assert "time_interval" in kwargs, "you need to provide time_interval for plotting cold_miss2d"
            cold_miss_ratio_2d(self.reader, kwargs['mode'], kwargs['time_interval'], figname=figname)

        elif plot_type == 'request_num':
            assert "mode" in kwargs, "you need to provide mode(r/v) for plotting request_num2d"
            assert "time_interval" in kwargs, "you need to provide time_interval for plotting request_num2d"
            request_num_2d(self.reader, kwargs['mode'], kwargs['time_interval'], figname=figname)

        elif plot_type == 'mapping':
            nameMapping_2d(self.reader, kwargs.get('ratio', 0.1), figname=figname)

        else:
            print("currently don't support your specified plot_type: " + str(plot_type))

    def evictionPlot(self, mode, time_interval, plot_type, algorithm, cache_size, cache_params=None, **kwargs):
        """
        plot eviction stat vs time, currently support 
        reuse_dist, freq, accumulative_freq 
        :param mode: 
        :param time_interval: 
        :param plot_type: 
        :param algorithm: 
        :param cache_size: 
        :param cache_params: 
        :param kwargs: 
        :return: 
        """
        if plot_type == "reuse_dist":
            eviction_stat_reuse_dist_plot(self.reader, algorithm, cache_size, mode,
                                          time_interval, cache_params=cache_params, **kwargs)
        elif plot_type == "freq":
            eviction_stat_freq_plot(self.reader, algorithm, cache_size, mode, time_interval,
                                    accumulative=False, cache_params=cache_params, **kwargs)

        elif plot_type == "accumulative_freq":
            eviction_stat_freq_plot(self.reader, algorithm, cache_size, mode, time_interval,
                                    accumulative=True, cache_params=cache_params, **kwargs)
        else:
            print("the plot type you specified is not supported: {}, currently only support: {}".format(
                plot_type, "reuse_dist, freq, accumulative_freq"
            ))

    def plotHRCs(self, algorithm_list, cache_params=None, cache_size=-1, bin_size=-1, auto_size=True, **kwargs):
        """
        
        :param algorithm_list: 
        :param cache_params: 
        :param cache_size: 
        :param bin_size: 
        :param auto_size: 
        :param kwargs: block_unit_size, num_of_threads, label, autosize_threshold  
        :return: 
        """
        plot_dict = prepPlotParams("Hit Ratio Curve", "Cache Size/item.", "Hit Ratio", "HRC.png", **kwargs)
        num_of_threads = 4
        if 'num_of_threads' in kwargs:
            num_of_threads = kwargs['num_of_threads']

        use_general_profiler = False
        if 'use_general_profiler' in kwargs:
            use_general_profiler = True

        save_gradually = False
        if 'save_gradually' in kwargs and kwargs['save_gradually']:
            save_gradually = True

        profiling_with_size = False
        label = algorithm_list
        threshold = 0.98
        LRU_HR = None

        if 'label' in kwargs:
            label = kwargs['label']
        if 'autosize_threshold' in kwargs:
            threshold = kwargs['autosize_threshold']

        if auto_size:
            LRU_HR = LRUProfiler(self.reader).plotHRC(auto_resize=True, threshold=threshold, no_save=True)
            cache_size = len(LRU_HR)
        else:
            assert cache_size < self.num_of_req(), "you cannot specify cache size larger than trace length"

        if bin_size == -1:
            bin_size = cache_size // DEFAULT_BIN_NUM_PROFILER + 1

        # check whether profiling with size
        for i in range(len(algorithm_list)):
            if i < len(cache_params) and cache_params[i] and 'block_unit_size' in cache_params[i]:
                profiling_with_size = True
                break

        for i in range(len(algorithm_list)):
            alg = algorithm_list[i]
            if cache_params and i < len(cache_params):
                cache_param = cache_params[i]
                if profiling_with_size:
                    if cache_param is None or 'block_unit_size' not in cache_param:
                        ERROR("seems you want to profiling with size, but you didn't provide block_unit_size in "
                              "cache params {}".format(cache_param))

            else:
                cache_param = None
            profiler = self.profiler(alg, cache_param, cache_size, use_general_profiler,
                                     bin_size=bin_size, num_of_threads=num_of_threads)
            t1 = time.time()

            if alg == "LRU":
                if LRU_HR is None:  # no auto_resize
                    hr = profiler.get_hit_rate()
                    plt.plot(hr[:-2], label=label[i])
                else:
                    plt.plot(LRU_HR, label=label[i])
            else:
                hr = profiler.get_hit_rate()
                plt.plot([i * bin_size for i in range(len(hr))], hr, label=label[i])
            self.reader.reset()
            INFO("HRC plotting {} computation finished using time {} s".format(alg, time.time() - t1))
            if save_gradually:
                plt.savefig(plot_dict['figname'], dpi=600)

        plt.legend(loc="best")
        plt.xlabel(plot_dict['xlabel'])
        plt.ylabel(plot_dict['ylabel'])
        plt.title(plot_dict['title'], fontsize=18, color='black')

        if profiling_with_size:
            plt.xlabel("Cache Size (MB)")
            plt.gca().xaxis.set_major_formatter(
                FuncFormatter(lambda x, p: int(x * kwargs['block_unit_size'] // 1024 // 1024)))

        if not 'no_save' in kwargs or not kwargs['no_save']:
            plt.savefig(plot_dict['figname'], dpi=600)
        INFO("HRC plot is saved at the same directory")
        try:
            plt.show()
        except:
            pass
        plt.clf()

    def plotMRCs(self, algorithm_list, cache_params=None, cache_size=-1, bin_size=-1, auto_size=True, **kwargs):
        """
        plot MRCs, not updated, might be deprecated 
        :param algorithm_list: 
        :param cache_params: 
        :param cache_size: 
        :param bin_size: 
        :param auto_size: 
        :param kwargs: 
        :return: 
        """
        plot_dict = prepPlotParams("Miss Ratio Curve", "Cache Size/item.", "Miss Ratio", "MRC.png", **kwargs)
        num_of_threads = 4
        if 'num_of_threads' in kwargs:
            num_of_threads = kwargs['num_of_threads']
        if 'label' not in kwargs:
            label = algorithm_list
        else:
            label = kwargs['label']

        threshold = 0.98
        if 'autosize_threshold' in kwargs:
            threshold = kwargs['autosize_threshold']

        ymin = 1

        if auto_size:
            cache_size = LRUProfiler(self.reader).plotMRC(auto_resize=True, threshold=threshold, no_save=True)
        else:
            assert cache_size < self.num_of_req(), "you cannot specify cache size larger than trace length"

        if bin_size == -1:
            bin_size = cache_size // DEFAULT_BIN_NUM_PROFILER + 1
        for i in range(len(algorithm_list)):
            alg = algorithm_list[i]
            if cache_params and i < len(cache_params):
                cache_param = cache_params[i]
            else:
                cache_param = None
            profiler = self.profiler(alg, cache_param, cache_size,
                                     bin_size=bin_size, num_of_threads=num_of_threads)
            mr = profiler.get_miss_rate()
            ymin = min(ymin, max(min(mr) - 0.02, 0))
            self.reader.reset()
            # plt.xlim(0, cache_size)
            if alg != "LRU":
                plt.plot([i * bin_size for i in range(len(mr))], mr, label=label[i])
            else:
                plt.plot(mr[:-2], label=label[i])

        print("ymin = {}".format(ymin))
        if "ymin" in kwargs:
            ymin = kwargs['ymin']

        plt.ylim(ymin=ymin)
        plt.semilogy()
        plt.legend(loc="best")
        plt.xlabel(plot_dict['xlabel'])
        plt.ylabel(plot_dict['ylabel'])
        plt.title(plot_dict['title'], fontsize=18, color='black')
        if not 'no_save' in kwargs or not kwargs['no_save']:
            plt.savefig(plot_dict['figname'], dpi=600)
        INFO("plot is saved at the same directory")
        try:
            plt.show()
        except:
            pass
        plt.clf()

    def __len__(self):
        assert self.reader is not None, "you must open a trace to call len"
        return len(self.reader)

    def __iter__(self):
        assert self.reader, "you haven't provided a data file"
        return self.reader

    def next(self):
        return self.__next__()

    def __next__(self):  # Python 3
        return self.reader.next()

    def __del__(self):
        self.close()

    def close(self):
        """
        close the reader opened in cachecow, and clean up in the future 
        :return: 
        """
        if self.reader is not None:
            self.reader.close()
            self.reader = None
