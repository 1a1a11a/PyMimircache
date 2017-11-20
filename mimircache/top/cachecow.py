# coding=utf-8

"""
this module offer the upper level API to user, it currently supports four types of operations,

* **trace loading**
* **trace information retrieving**
* **trace profiling**
* **plotting**

Author: Jason Yang <peter.waynechina@gmail.com> 2017/08

"""
from matplotlib.ticker import FuncFormatter
from mimircache.const import CExtensionMode
if CExtensionMode:
    import mimircache.c_heatmap
    
from mimircache.profiler.evictionStat import *
from mimircache.profiler.twoDPlots import *
from mimircache.utils.prepPlotParams import *
from mimircache.cacheReader.traceStat import traceStat
from multiprocessing import cpu_count


class cachecow:
    __all__ = ["open",
           "csv",
           "vscsi",
           "binary",
           "stat",
           "num_of_req",
           "num_of_uniq_req",
           "get_reuse_distance",
           "get_hit_ratio_dict",
           "heatmap",
           "diff_heatmap",
           "twoDPlot",
           "eviction_plot",
           "plotHRCs",
           "plotMRCs",
           "characterize",
           "close"]

    def __init__(self, **kwargs):
        self.reader = None
        self.cache_size = 0
        self.n_req = -1
        self.n_uniq_req = -1
        self.cacheclass_mapping = {}

    def open(self, file_path, trace_type="p", data_type="c", **kwargs):
        """

        The default operation of this function opens a plain text trace,
        the format of a plain text trace is such a file that each line contains a label.

        By changing trace type, it can be used for opening other types of trace,
        supported trace type includes

        ==============  ==========  ===================
        trace_type      file type   require init_params
        ==============  ==========  ===================
            "p"         plain text          No
            "c"            csv             Yes
            "b"           binary           Yes
            "v"           vscsi             No
        ==============  ==========  ===================

        the effect of this is the save as calling corresponding functions (csv, binary, vscsi)

        :param file_path: the path to the data
        :param trace_type: type of trace, "p" for plainText, "c" for csv, "v" for vscsi, "b" for binary
        :param data_type: the type of request label, \
                    can be either "c" for string or "l" for number (for example block IO LBA)
        :param kwargs: parameters for opening the trace
        :return: reader object
        """

        if self.reader:
            self.reader.close()
        if trace_type == "p":
            self.reader = plainReader(file_path, data_type=data_type)

        elif trace_type == "c":
            assert "init_params" in kwargs, "please provide init_params for csv trace"
            init_params = kwargs["init_params"]
            kwargs_new = {}
            kwargs_new.update(kwargs)
            del kwargs_new["init_params"]
            self.csv(file_path, init_params, data_type=data_type, **kwargs_new)

        elif trace_type == 'b':
            assert "init_params" in kwargs, "please provide init_params for csv trace"
            init_params = kwargs["init_params"]
            kwargs_new = {}
            kwargs_new.update(kwargs)
            del kwargs_new["init_params"]
            self.binary(file_path, init_params, data_type=data_type, **kwargs_new)

        elif trace_type == 'v':
            self.vscsi(file_path, **kwargs)

        else:
            raise RuntimeError("unknown trace type {}".format(trace_type))

        return self.reader

    def csv(self, file_path, init_params, data_type="c",
            block_unit_size=0, disk_sector_size=0, **kwargs):
        """
        open a csv trace, init_params is a dictionary specifying the specs of the csv file,
        the possible keys are listed in the table below.
        The column/field number begins from 1, so the first column(field) is 1, the second is 2, etc.

        :param file_path: the path to the data
        :param init_params: params related to csv file, see above or csvReader for details
        :param data_type: the type of request label, \
                    can be either "c" for string or "l" for number (for example block IO LBA)
        :param block_unit_size: the block size for a cache, currently storage system only
        :param disk_sector_size: the disk sector size of input file, storage system only
        :return: reader object

        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | Keyword Argument | file type   | Value Type   | Default Value       | Description                                       |
        +==================+=============+==============+=====================+===================================================+
        | label            | csv/ binary | int          | this is required    | the column of the label of the request            |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | fmt              | binary      | string       | this is required    | fmt string of binary data, same as python struct  |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | header           | csv         | True/False   |      False          | whether csv data has header                       |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | delimiter        | csv         | char         |        ","          | the delimiter separating fields in the csv file   |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | real_time        | csv/ binary | int          |        NA           | the column of real time                           |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | op               | csv/ binary | int          |        NA           | the column of operation (read/write)              |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        | size             | csv/ binary | int          |        NA           | the column of block/request size                  |
        +------------------+-------------+--------------+---------------------+---------------------------------------------------+
        """

        if self.reader:
            self.reader.close()
        self.reader = csvReader(file_path, data_type=data_type,
                                block_unit_size=block_unit_size,
                                disk_sector_size=disk_sector_size,
                                init_params=init_params, **kwargs)
        return self.reader

    def binary(self, file_path, init_params, data_type='l',
               block_unit_size=0, disk_sector_size=0, **kwargs):
        """
        open a binary trace file, init_params see function csv

        :param file_path: the path to the data
        :param init_params: params related to the spec of data, see above csv for details
        :param data_type: the type of request label, \
                    can be either "c" for string or "l" for number (for example block IO LBA)
        :param block_unit_size: the block size for a cache, currently storage system only
        :param disk_sector_size: the disk sector size of input file, storage system only
        :return: reader object
        """

        if self.reader:
            self.reader.close()
        self.reader = binaryReader(file_path, data_type=data_type,
                                   block_unit_size=block_unit_size,
                                   disk_sector_size=disk_sector_size,
                                   init_params=init_params, **kwargs)
        return self.reader

    def vscsi(self, file_path, block_unit_size=0, disk_sector_size=512, **kwargs):
        """
        open vscsi trace file

        :param file_path: the path to the data
        :param block_unit_size: the block size for a cache, currently storage system only
        :param disk_sector_size: the disk sector size of input file, storage system only
        :return: reader object
        """

        if self.reader:
            self.reader.close()
        if "data_type" in kwargs:
            del kwargs["data_type"]
        self.reader = vscsiReader(file_path, data_type="l",
                                  block_unit_size=block_unit_size,
                                  disk_sector_size=disk_sector_size, **kwargs)
        return self.reader


    def reset(self):
        """
        reset cachecow to initial state, including
            reset reader to the beginning of the trace

        """
        assert self.reader is not None, "reader is None, cannot reset"
        self.reader.reset()


    def close(self):
        """
        close the reader opened in cachecow, and clean up in the future
        """

        if self.reader is not None:
            self.reader.close()
            self.reader = None


    def stat(self):
        """
        obtain the statistical information about the trace, including

            * number of requests
            * number of uniq items
            * cold miss ratio
            * a list of top 10 popular in form of (obj, num of requests):
            * number of obj/block accessed only once
            * frequency mean
            * time span

        :return: a string of the information above
        """
        assert self.reader, "you haven't provided a data file"
        return traceStat(self.reader).get_stat()


    def num_of_req(self):
        """

        :return: the number of requests in the trace
        """
        if self.n_req == -1:
            self.n_req = self.reader.get_num_of_req()
        return self.n_req

    def num_of_uniq_req(self):
        """

        :return: the number of unique requests in the trace
        """
        if self.n_uniq_req == -1:
            self.n_uniq_req = self.reader.get_num_of_uniq_req()
        return self.n_uniq_req

    def get_reuse_distance(self):
        """

        :return: an array of reuse distance
        """
        return LRUProfiler(self.reader).get_reuse_distance()

    def get_hit_ratio_dict(self, algorithm, cache_size=-1, cache_params=None, bin_size=-1,
                      use_general_profiler=False, **kwargs):
        """

        :param algorithm: cache replacement algorithms
        :param cache_size: size of cache
        :param cache_params: parameters passed to cache, some of the cache replacement algorithms require parameters,
                for example LRU-K, SLRU
        :param bin_size: if algorithm is not LRU, then the hit ratio will be calculated by simulating cache at
                cache size [0, bin_size, bin_size*2 ... cache_size], this is not required for LRU
        :param use_general_profiler: if algorithm is LRU and you don't want to use LRUProfiler, then set this to True,
                possible reason for not using a LRUProfiler: 1. LRUProfiler is too slow for your large trace
                because the algorithm is O(NlogN) and it uses single thread; 2. LRUProfiler has a bug (let me know if you found a bug).
        :param kwargs: other parameters including num_of_threads
        :return: an dict of hit ratio of given algorithms, mapping from cache_size -> hit ratio
        """

        hit_ratio_dict = {}
        p = self.profiler(algorithm,
                          cache_params=cache_params,
                          cache_size=cache_size,
                          bin_size=bin_size,
                          use_general_profiler=use_general_profiler, **kwargs)
        hr = p.get_hit_ratio(cache_size=cache_size)
        if isinstance(p, LRUProfiler):
            for i in range(len(hr)-2):
                hit_ratio_dict[i] = hr[i]
        elif isinstance(p, cGeneralProfiler) or isinstance(p, generalProfiler):
            for i in range(len(hr)):
                hit_ratio_dict[i * p.bin_size] = hr[i]
        return hit_ratio_dict


    def profiler(self, algorithm, cache_params=None, cache_size=-1, bin_size=-1,
                 use_general_profiler=False, **kwargs):
        """
        get a profiler instance, this should not be used by most users

        :param use_general_profiler: this option is for LRU only, if it is True,
                                        then return a cGeneralProfiler for LRU,
                                        otherwise, return a LRUProfiler for LRU.

                                        Note: LRUProfiler does not require cache_size/bin_size params,
                                        it does not sample thus provides a smooth curve, however, it is O(logN) at each step,
                                        in constrast, cGeneralProfiler samples the curve, but use O(1) at each step
        :param kwargs: num_of_threads
        :return: a profiler instance
        """

        num_of_threads = kwargs.get("num_of_threads", DEFAULT_NUM_OF_THREADS)
        assert self.reader is not None, "you haven't opened a trace yet"

        if algorithm.lower() == "lru" and not use_general_profiler:
            profiler = LRUProfiler(self.reader, cache_size, cache_params)
        else:
            assert cache_size != -1, "you didn't provide size for cache"
            assert cache_size <= self.num_of_req(), "you cannot specify cache size({}) " \
                                                        "larger than trace length({})".format(cache_size,
                                                                                              self.num_of_req())
            if isinstance(algorithm, str):
                if algorithm.lower() in c_available_cache:
                    profiler = cGeneralProfiler(self.reader, cache_alg_mapping[algorithm.lower()],
                                                cache_size, bin_size,
                                                cache_params, num_of_threads)
                else:
                    profiler = generalProfiler(self.reader, self.cacheclass_mapping[algorithm.lower()],
                                               cache_size, bin_size,
                                               cache_params, num_of_threads)
            else:
                profiler = generalProfiler(self.reader, algorithm, cache_size, bin_size,
                                           cache_params, num_of_threads)

        return profiler


    def heatmap(self, time_mode, plot_type, time_interval=-1, num_of_pixels=-1,
                algorithm="LRU", cache_params=None, cache_size=-1, **kwargs):
        """
        plot heatmaps, currently supports the following heatmaps

        * hit_ratio_start_time_end_time

        * hit_ratio_start_time_cache_size (python only)
        * avg_rd_start_time_end_time (python only)
        * cold_miss_count_start_time_end_time (python only)

        * rd_distribution
        * rd_distribution_CDF
        * future_rd_distribution
        * dist_distribution
        * reuse_time_distribution

        :param time_mode: the type of time, can be "v" for virtual time, or "r" for real time
        :param plot_type: the name of plot types, see above for plot types
        :param time_interval: the time interval of one pixel
        :param num_of_pixels: if you don't to use time_interval,
                    you can also specify how many pixels you want in one dimension,
                    note this feature is not well tested
        :param algorithm: what algorithm to use for plotting heatmap,
                this is not required for distance related heatmap like rd_distribution
        :param cache_params: parameters passed to cache, some of the cache replacement algorithms require parameters,
                for example LRU-K, SLRU
        :param cache_size: The size of cache, this is required only for *hit_ratio_start_time_end_time*
        :param kwargs: other parameters for computation and plotting such as num_of_threads, figname
        """

        assert self.reader is not None, "you haven't opened a trace yet"
        assert cache_size <= self.num_of_req(), \
                    "you cannot specify cache size({}) larger than " \
                    "trace length({})".format(cache_size, self.num_of_req())

        l = ["avg_rd_start_time_end_time", "hit_ratio_start_time_cache_size"]

        if plot_type in l:
            hm = heatmap()
        else:
            if algorithm.lower() in c_available_cache:
                hm = cHeatmap()

            else:
                hm = heatmap()

        hm.heatmap(self.reader, time_mode, plot_type,
                   time_interval=time_interval,
                   num_of_pixels=num_of_pixels,
                   cache_size=cache_size,
                   algorithm=cache_alg_mapping[algorithm.lower()],
                   cache_params=cache_params,
                   **kwargs)


    def diff_heatmap(self, time_mode, plot_type, algorithm1, time_interval=-1, num_of_pixels=-1,
                     algorithm2="Optimal", cache_params1=None, cache_params2=None, cache_size=-1, **kwargs):
        """
        Plot the differential heatmap between two algorithms by alg2 - alg1

        :param time_mode: time time_mode "v" for virutal time, "r" for real time
        :param plot_type: same as the name in heatmap function
        :param algorithm1:  name of the first alg
        :param time_interval: same as in heatmap
        :param num_of_pixels: same as in heatmap
        :param algorithm2: name of the second algorithm
        :param cache_params1: parameters of the first algorithm
        :param cache_params2: parameters of the second algorithm
        :param kwargs: include num_of_threads
        """

        figname = kwargs.get("figname", 'differential_heatmap.png')
        num_of_threads = kwargs.get("num_of_threads", DEFAULT_NUM_OF_THREADS)
        assert self.reader is not None, "you haven't opened a trace yet"
        assert cache_size != -1, "you didn't provide size for cache"
        assert cache_size <= self.num_of_req(), \
                    "you cannot specify cache size({}) larger than " \
                    "trace length({})".format(cache_size, self.num_of_req())


        if algorithm1.lower() in c_available_cache and algorithm2.lower() in c_available_cache:
            hm = cHeatmap()
            hm.diffHeatmap(self.reader, time_mode, plot_type,
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
                xydict1 = hm.calculate_heatmap_dat(self.reader, time_mode, plot_type,
                                                   time_interval=time_interval,
                                                   cache_size=cache_size,
                                                   algorithm=algorithm1,
                                                   cache_params=cache_params1,
                                                   **kwargs)[0]
            else:
                xydict1 = mimircache.c_heatmap.heatmap(self.reader.cReader, time_mode, plot_type,
                                                       cache_size=cache_size,
                                                       time_interval=time_interval,
                                                       algorithm=algorithm1,
                                                       cache_params=cache_params1,
                                                       num_of_threads=num_of_threads)

            if algorithm2.lower() not in c_available_cache:
                xydict2 = hm.calculate_heatmap_dat(self.reader, time_mode, plot_type,
                                                   time_interval=time_interval,
                                                   cache_size=cache_size,
                                                   algorithm=algorithm2,
                                                   cache_params=cache_params2,
                                                   **kwargs)[0]
            else:
                xydict2 = mimircache.c_heatmap.heatmap(self.reader.cReader, time_mode, plot_type,
                                                       time_interval=time_interval,
                                                       cache_size=cache_size,
                                                       algorithm=algorithm2,
                                                       cache_params=cache_params2,
                                                       num_of_threads=num_of_threads)

            cHm = cHeatmap()
            text = "      differential heatmap\n      cache size: {},\n      cache type: ({}-{})/{},\n" \
                   "      time type: {},\n      time interval: {},\n      plot type: \n{}".format(
                cache_size, algorithm2, algorithm1, algorithm1, time_mode, time_interval, plot_type)

            x1, y1 = xydict1.shape
            x1 = int(x1 / 2.8)
            y1 /= 8
            if time_mode == 'r':
                time_mode_string = "Real"
            elif time_mode == "v":
                time_mode_string = "Virtual"
            else:
                raise RuntimeError("unknown time time_mode {}".format(time_mode))

            cHm.setPlotParams('x', '{} Time'.format(time_mode_string), xydict=xydict1,
                              label='Start Time ({})'.format(time_mode_string),
                              text=(x1, y1, text))
            cHm.setPlotParams('y', '{} Time'.format(time_mode_string), xydict=xydict1,
                              label='End Time ({})'.format(time_mode_string),
                              fixed_range=(-1, 1))
            np.seterr(divide='ignore', invalid='ignore')

            plot_dict = (xydict2 - xydict1) / xydict1
            cHm.draw_heatmap(plot_dict, figname=figname)



    def twoDPlot(self, plot_type, **kwargs):
        """
        an aggregate function for all two dimenional plots printing except hit ratio curve


        ========================  ============================  =================================================
                plot type               required parameters         Description
        ========================  ============================  =================================================
            cold_miss_count         time_mode, time_interval     cold miss count VS time
            cold_miss_ratio         time_mode, time_interval     coid miss ratio VS time
            request_rate            time_mode, time_interval     num of requests VS time
            popularity              NA                           Percentage of obj VS frequency
            rd_popularity           NA                           Num of req VS reuse distance
            rt_popularity           NA                           Num of req VS reuse time
            mapping                 NA                           mapping from original objID to sequential number
          interval_hit_ratio        cache_size                   hit ratio of interval VS time
        ========================  ============================  =================================================


        :param plot_type: type of the plot, see above
        :param kwargs: paramters related to plots, see twoDPlots module for detailed control over plots
        """

        kwargs["figname"] = kwargs.get("figname", "{}.png".format(plot_type))

        if plot_type == 'cold_miss' or plot_type == "cold_miss_count":
            if plot_type == 'cold_miss':
                print("please use cold_miss_count, cold_miss is deprecated")
            assert "mode" in kwargs or "time_mode" in kwargs, \
                "you need to provide time_mode (r/v) for plotting cold_miss2d"
            assert "time_interval" in kwargs, \
                "you need to provide time_interval for plotting cold_miss2d"
            return cold_miss_count_2d(self.reader, **kwargs)

        elif plot_type == 'cold_miss_ratio':
            assert "mode" in kwargs or "time_mode" in kwargs, \
                "you need to provide time_mode (r/v) for plotting cold_miss2d"
            assert "time_interval" in kwargs, \
                "you need to provide time_interval for plotting cold_miss2d"
            return cold_miss_ratio_2d(self.reader, **kwargs)

        elif plot_type == "request_rate":
            assert "mode" in kwargs or "time_mode" in kwargs, \
                "you need to provide time_mode (r/v) for plotting request_rate2d"
            assert "time_interval" in kwargs, \
                "you need to provide time_interval for plotting request_num2d"
            return request_rate_2d(self.reader, **kwargs)

        elif plot_type == "popularity":
            return popularity_2d(self.reader, **kwargs)

        elif plot_type == "rd_popularity":
            return rd_popularity_2d(self.reader, **kwargs)

        elif plot_type == "rt_popularity":
            return rt_popularity_2d(self.reader, **kwargs)

        elif plot_type == 'mapping':
            namemapping_2d(self.reader, **kwargs)

        elif plot_type == "interval_hit_ratio":
            assert "cache_size" in kwargs, "please provide cache size for interval hit ratio curve plotting"
            return interval_hit_ratio_2d(self.reader, **kwargs)

        else:
            WARNING("currently don't support your specified plot_type: " + str(plot_type))


    def evictionPlot(self, mode, time_interval, plot_type, algorithm, cache_size, cache_params=None, **kwargs):
        """
        plot eviction stat vs time, currently support reuse_dist, freq, accumulative_freq

        This function is going to be deprecated
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


    def plotHRCs(self, algorithm_list, cache_params=(),
                 cache_size=-1, bin_size=-1,
                 auto_resize=True, figname="HRC.png", **kwargs):
        """
        this function provides hit ratio curve plotting

        :param algorithm_list: a list of algorithm(s)
        :param cache_params: the corresponding cache params for the algorithms,
                                use None for algorithms that don't require cache params,
                                if none of the alg requires cache params, you don't need to set this
        :param cache_size:  maximal size of cache, use -1 for max possible size
        :param bin_size:    bin size for non-LRU profiling
        :param auto_resize:   when using max possible size or specified cache size too large,
                                you will get a huge plateau at the end of hit ratio curve,
                                set auto_resize to True to cutoff most of the big plateau
        :param kwargs: options: block_unit_size, num_of_threads,
                        auto_resize_threshold, xlimit, ylimit, cache_unit_size

                        save_gradually - save a figure everytime computation for one algorithm finishes,

                        label - instead of using algorithm list as label, specify user-defined label
        """

        plot_dict = prepPlotParams("Hit Ratio Curve", "Cache Size (Items)", "Hit Ratio", figname, **kwargs)
        hit_ratio_dict = {}

        num_of_threads          =       kwargs.get("num_of_threads",        os.cpu_count())
        cache_unit_size         =       kwargs.get("cache_unit_size",       0)
        use_general_profiler    =       kwargs.get("use_general_profiler",  False)
        save_gradually          =       kwargs.get("save_gradually",        False)
        threshold               =       kwargs.get('auto_resize_threshold', 0.98)
        label                   =       kwargs.get("label",                 algorithm_list)

        profiling_with_size = False
        LRU_HR = None

        if cache_size == -1 and auto_resize:
            LRU_HR = LRUProfiler(self.reader).plotHRC(auto_resize=True, threshold=threshold, no_save=True)
            cache_size = len(LRU_HR)
        else:
            assert cache_size < self.num_of_req(), "you cannot specify cache size larger than trace length"

        if bin_size == -1:
            bin_size = cache_size // DEFAULT_BIN_NUM_PROFILER + 1

        # check whether profiling with size
        block_unit_size = 0
        for i in range(len(algorithm_list)):
            if i < len(cache_params) and cache_params[i]:
                block_unit_size = cache_params[i].get("block_unit_size", 0)
                if block_unit_size != 0:
                    profiling_with_size = True
                    break
        if profiling_with_size and cache_unit_size != 0 and block_unit_size != cache_unit_size:
            raise RuntimeError("cache_unit_size and block_unit_size is not equal {} {}".\
                                format(cache_unit_size, block_unit_size))


        for i in range(len(algorithm_list)):
            alg = algorithm_list[i]
            if cache_params and i < len(cache_params):
                cache_param = cache_params[i]
                if profiling_with_size:
                    if cache_param is None or 'block_unit_size' not in cache_param:
                        ERROR("it seems you want to profiling with size, "
                              "but you didn't provide block_unit_size in "
                              "cache params {}".format(cache_param))
                    elif cache_param["block_unit_size"] != block_unit_size:
                        ERROR("only same block unit size for single plot is allowed")

            else:
                cache_param = None
            profiler = self.profiler(alg, cache_param, cache_size, bin_size=bin_size,
                                     use_general_profiler=use_general_profiler,
                                     num_of_threads=num_of_threads)
            t1 = time.time()

            if alg == "LRU":
                if LRU_HR is None:  # no auto_resize
                    hr = profiler.get_hit_ratio()
                    if use_general_profiler:
                        # save the computed hit ratio
                        hit_ratio_dict["LRU"] = {}
                        for j in range(len(hr)):
                            hit_ratio_dict["LRU"][j * bin_size] = hr[j]
                        plt.plot([j * bin_size for j in range(len(hr))], hr, label=label[i])
                    else:
                        # save the computed hit ratio
                        hit_ratio_dict["LRU"] = {}
                        for j in range(len(hr)-2):
                            hit_ratio_dict["LRU"][j] = hr[j]
                        plt.plot(hr[:-2], label=label[i])
                else:
                    # save the computed hit ratio
                    hit_ratio_dict["LRU"] = {}
                    for j in range(len(LRU_HR)):
                        hit_ratio_dict["LRU"][j] = LRU_HR[j]
                    plt.plot(LRU_HR, label=label[i])
            else:
                hr = profiler.get_hit_ratio()
                # save the computed hit ratio
                hit_ratio_dict[alg] = {}
                for j in range(len(hr)):
                    hit_ratio_dict[alg][j * bin_size] = hr[j]
                plt.plot([j * bin_size for j in range(len(hr))], hr, label=label[i])
            self.reader.reset()
            INFO("HRC plotting {} computation finished using time {} s".format(alg, time.time() - t1))
            if save_gradually:
                plt.savefig(plot_dict['figname'], dpi=600)

        plt.legend(loc="best")
        plt.xlabel(plot_dict['xlabel'])
        plt.ylabel(plot_dict['ylabel'])
        plt.title(plot_dict['title'], fontsize=18, color='black')

        if "xlimit" in kwargs:
            plt.xlim(kwargs["xlimit"])
        if "ylimit" in kwargs:
            plt.ylim(kwargs["ylimit"])

        if cache_unit_size != 0:
            plt.xlabel("Cache Size (MB)")
            plt.gca().xaxis.set_major_formatter(
                FuncFormatter(lambda x, p: int(x * cache_unit_size // 1024 // 1024)))

        if not 'no_save' in kwargs or not kwargs['no_save']:
            plt.savefig(plot_dict['figname'], dpi=600)
        INFO("HRC plot is saved")
        try:
            plt.show()
        except:
            pass
        plt.clf()
        return hit_ratio_dict


    def characterize(self, type, cache_size=-1, **kwargs):
        """
        use this function to obtain a series of plots about your trace, the type includes

        * short - short run time, fewer plots with less accuracy
        * medium
        * long
        * all - most of the available plots with high accuracy, notice it can take **LONG** time on big trace

        :param type: see above, options: short, medium, long, all
        :param cache_size: estimated cache size for the trace, if -1, mimircache will estimate the cache size
        :param kwargs: print_stat
        :return: trace stat string
        """

        # TODO: jason: allow one single function call to obtain the most useful information
        # and would be better to give time estimation while running

        supported_types = ["short", "medium", "long", "all"]
        if type not in supported_types:
            WARNING("unknown type {}, supported types: {}".format(type, supported_types))
            return

        trace_stat = traceStat(self.reader)
        if kwargs.get("print_stat", True):
            INFO("trace information ")
            print(trace_stat)

        if cache_size == -1:
            cache_size = trace_stat.num_of_uniq_obj//100

        if type == "short":
            # short should support [basic stat, HRC of LRU, OPT, cold miss ratio, popularity]
            INFO("now begin to plot cold miss ratio curve")
            self.twoDPlot("cold_miss_ratio", mode="v", time_interval=trace_stat.num_of_requests//100)

            INFO("now begin to plot popularity curve")
            self.twoDPlot("popularity")

            INFO("now begin to plot hit ratio curves")
            self.plotHRCs(["LRU", "Optimal"], cache_size=cache_size, bin_size=cache_size//cpu_count()+1,
                          num_of_threads=cpu_count(),
                          use_general_profiler=True, save_gradually=True)

        elif type == "medium":
            if trace_stat.time_span != 0:
                INFO("now begin to plot request rate curve")
                self.twoDPlot("request_rate", mode="r", time_interval=trace_stat.time_span//100)

            INFO("now begin to plot cold miss ratio curve")
            self.twoDPlot("cold_miss_ratio", mode="v", time_interval=trace_stat.num_of_requests//100)

            INFO("now begin to plot popularity curve")
            self.twoDPlot("popularity")

            INFO("now begin to plot mapping plot")
            self.twoDPlot("mapping")

            INFO("now begin to plot hit ratio curves")
            self.plotHRCs(["LRU", "Optimal", "LFU"], cache_size=cache_size,
                          bin_size=cache_size//cpu_count()//4+1,
                          num_of_threads=cpu_count(),
                          use_general_profiler=True, save_gradually=True)


        elif type == "long":
            if trace_stat.time_span != 0:
                INFO("now begin to plot request rate curve")
                self.twoDPlot("request_rate", mode="r", time_interval=trace_stat.time_span//100)

            INFO("now begin to plot cold miss ratio curve")
            self.twoDPlot("cold_miss_ratio", mode="v", time_interval=trace_stat.num_of_requests//100)

            INFO("now begin to plot popularity curve")
            self.twoDPlot("popularity")

            INFO("now begin to plot rd distribution popularity")
            self.twoDPlot("rd_distribution")

            INFO("now begin to plot mapping plot")
            self.twoDPlot("mapping")

            INFO("now begin to plot rd distribution heatmap")
            self.heatmap("v", "rd_distribution", time_interval=trace_stat.num_of_requests//100)

            INFO("now begin to plot hit ratio curves")
            self.plotHRCs(["LRU", "Optimal", "LFU", "ARC"], cache_size=cache_size,
                          bin_size=cache_size//cpu_count()//16+1,
                          num_of_threads=cpu_count(),
                          save_gradually=True)

            if kwargs.get("print_stat", True):
                INFO("now begin to plot hit_ratio_start_time_end_time heatmap")
            self.heatmap("v", "hit_ratio_start_time_end_time",
                         time_interval=trace_stat.num_of_requests//100,
                         cache_size=cache_size)


        elif type == "all":
            if trace_stat.time_span != 0:
                INFO("now begin to plot request rate curve")
                self.twoDPlot("request_rate", mode="r", time_interval=trace_stat.time_span//200)

            INFO("now begin to plot cold miss ratio curve")
            self.twoDPlot("cold_miss_ratio", mode="v", time_interval=trace_stat.num_of_requests//200)

            INFO("now begin to plot popularity curve")
            self.twoDPlot("popularity")

            INFO("now begin to plot rd distribution popularity")
            self.twoDPlot("rd_distribution")

            INFO("now begin to plot mapping plot")
            self.twoDPlot("mapping")

            INFO("now begin to plot rd distribution heatmap")
            self.heatmap("v", "rd_distribution", time_interval=trace_stat.num_of_requests//200)


            INFO("now begin to plot hit ratio curves")
            self.plotHRCs(["LRU", "Optimal", "LFU", "ARC"], cache_size=cache_size,
                          bin_size=cache_size//cpu_count()//60+1,
                          num_of_threads=cpu_count(),
                          save_gradually=True)

            INFO("now begin to plot hit_ratio_start_time_end_time heatmap")
            self.heatmap("v", "hit_ratio_start_time_end_time",
                         time_interval=trace_stat.num_of_requests//200,
                         cache_size=cache_size)

        return str(trace_stat)


    def __len__(self):
        assert self.reader, "you haven't provided a data file"
        return len(self.reader)

    def __iter__(self):
        assert self.reader, "you haven't provided a data file"
        return self.reader

    def __next__(self):  # Python 3
        return self.reader.next()

    def __del__(self):
        self.close()
