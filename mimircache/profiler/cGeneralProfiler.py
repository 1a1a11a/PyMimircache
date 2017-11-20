# coding=utf-8
""" this module is used for non-LRU cache replacement algorithms (of course, LRU also works)
it uses sampling, basically it simulates a cache at cache size [0, bin_size, bin_size*2 ...]
(of course, there is no need for cache size 0,
so the hit count or hit ratio for cache size 0 is always 0).
The time complexity of this simulation is O(mN) where m is the number of bins (cache_size//bin_size),
N is the trace length.

For LRU, you can use module, but LRUProfiler will provide a better accuracy
as it does not have sampling over cache size, you will always get a smooth curve,
but the cost is an O(nlogn) algorithm.

Author: Jason Yang <peter.waynechina@gmail.com> 2016/07

"""
# -*- coding: utf-8 -*-


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mimircache.cacheReader.abstractReader import cacheReaderAbstract
from mimircache.utils.printing import *
from mimircache.const import CExtensionMode
if CExtensionMode:
    import mimircache.c_generalProfiler
from mimircache.const import *


class cGeneralProfiler:
    all = ("get_hit_count",
           "get_hit_ratio",
           "get_miss_ratio",
           "plotMRC",
           "plotHRC")

    def __init__(self, reader,
                 cache_name, cache_size, bin_size=-1, cache_params=None,
                 num_of_threads=DEFAULT_NUM_OF_THREADS):
        """
        initialization of a cGeneralProfiler
        :param reader:
        :param cache_name:
        :param cache_size:
        :param bin_size: the sample granularity, the smaller the better, but also much longer run time
        :param cache_params: parameters about the given cache replacement algorithm
        :param num_of_threads:
        """

        # make sure reader is valid
        assert isinstance(reader, cacheReaderAbstract), \
            "you provided an invalid cacheReader: {}".format(reader)

        assert cache_name.lower() in cache_alg_mapping, "please check your cache replacement algorithm: " + cache_name
        assert cache_name.lower() in c_available_cache, \
            "cGeneralProfiler currently only available on the following caches: {}\n, " \
            "please use generalProfiler".format(pformat(c_available_cache))
        assert cache_size != 0, "you cannot provide size 0"

        self.reader = reader
        self.cache_size = cache_size
        self.cache_name = cache_alg_mapping[cache_name.lower()]
        self.bin_size = bin_size
        if self.bin_size == -1:
            self.bin_size = int(self.cache_size / DEFAULT_BIN_NUM_PROFILER)

        if self.bin_size == 0:
            self.bin_size = 1

        self.cache_params = cache_params
        if self.cache_params is None:
            self.cache_params = {}
        self.num_of_threads = num_of_threads

        # check whether user want to profling with size
        self.block_unit_size = self.cache_params.get("block_unit_size", 0)
        block_unit_size_names = {"unit_size", "block_size", "chunk_size"}
        for name in block_unit_size_names:
            if name in self.cache_params:
                self.block_unit_size = cache_params[name]
                break

        # if the given file is not embedded reader, needs conversion for C backend
        need_convert = True
        for instance in c_available_cacheReader:
            if isinstance(reader, instance):
                need_convert = False
                break
        if need_convert:
            self._prepare_file()

        # this is for deprecated functions, as old version use hit rate instead of hit ratio
        self.get_hit_rate = self.get_hit_ratio
        self.get_miss_rate = self.get_miss_ratio


    def _prepare_file(self):
        """
        this is used when user passed in a customized reader,
        but customized reader is not supported in C backend
        so we convert it to plainText
        TODO: this is not the best approach due to information loss in the conversion
        :return:
        """
        self.num_of_lines = 0
        with open('.temp.dat', 'w') as ofile:
            i = self.reader.read_one_element()
            while i is not None:
                self.num_of_lines += 1
                ofile.write(str(i) + '\n')
                i = self.reader.read_one_element()
        self.reader = plainReader('.temp.dat')


    def get_hit_count(self, **kwargs):
        """
        obtain hit count at cache size [0, bin_size, bin_size*2 ...]
        :return: a numpy array, with hit count corresponding to size [0, bin_size, bin_size*2 ...]
        """
        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)

        # this is going to be deprecated
        if 'begin' in kwargs:
            sanity_kwargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            sanity_kwargs['end'] = kwargs['end']

        if self.block_unit_size != 0:
            print("not supported yet")
        else:
            return mimircache.c_generalProfiler.get_hit_count(self.reader.cReader, self.cache_name, cache_size,
                                                   self.bin_size, cache_params=self.cache_params, **sanity_kwargs)

    def get_hit_ratio(self, **kwargs):
        """
        obtain hit ratio at cache size [0, bin_size, bin_size*2 ...]
        :return: a numpy array, with hit rate corresponding to size [0, bin_size, bin_size*2 ...]
        """
        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)
        bin_size = kwargs.get("bin_size", self.bin_size)

        # this is going to be deprecated
        if 'begin' in kwargs:
            sanity_kwargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            sanity_kwargs['end'] = kwargs['end']

        # handles both withsize and no size, but currently only storage system trace are supported with size
        return mimircache.c_generalProfiler.get_hit_ratio(self.reader.cReader, self.cache_name, cache_size,
                                                bin_size, cache_params=self.cache_params, **sanity_kwargs)



    def get_miss_ratio(self, **kwargs):
        """
        obtain miss ratio at cache size [0, bin_size, bin_size*2 ...]
        :return: a numpy array, with miss rate corresponding to size [0, bin_size, bin_size*2 ...]
        """
        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)
        bin_size = kwargs.get("bin_size", self.bin_size)

        # this is going to be deprecated
        if 'begin' in kwargs:
            sanity_kwargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            sanity_kwargs['end'] = kwargs['end']

        if self.block_unit_size != 0:
            print("not supported yet")
        else:
            return mimircache.c_generalProfiler.get_miss_ratio(self.reader.cReader, self.cache_name, cache_size,
                                                    bin_size, cache_params=self.cache_params, **sanity_kwargs)

    def plotMRC(self, figname="MRC.png", **kwargs):
        """
        this function is deprecated now and should not be used
        :param figname:
        :param kwargs:
        :return:
        """
        raise RuntimeWarning("function not updated")
        MRC = self.get_miss_ratio(**kwargs)
        try:
            # tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(x * self.bin_size))
            # plt.gca().xaxis.set_major_formatter(tick)
            plt.xlim(0, self.cache_size)
            plt.plot(range(0, self.cache_size + 1, self.bin_size), MRC)
            plt.xlabel("Cache Size (items)")
            plt.ylabel("Miss Rate")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=600)
            INFO("plot is saved")
            plt.show()
            plt.clf()
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function reports error, maybe this is a headless server? \nERROR: {}".format(e))
        return MRC


    def plotHRC(self, figname="HRC.png", **kwargs):
        """
        plot hit ratio curve of the given trace under given algorithm
        :param figname:
        :param kwargs:
        :return:
        """
        HRC = self.get_hit_ratio(**kwargs)
        try:
            plt.xlim(0, self.cache_size)
            plt.plot(range(0, self.cache_size + 1, self.bin_size), HRC)
            xlabel = "Cache Size (items)"

            cache_unit_size = kwargs.get("cache_unit_size", self.block_unit_size)
            if cache_unit_size != 0:
                xlabel = "Cache Size (MB)"
                plt.gca().xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: int(x * cache_unit_size // 1024 // 1024)))

            plt.xlabel(xlabel)
            plt.ylabel("Hit Ratio")
            plt.title('Hit Ratio Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=600)
            INFO("plot is saved")
            try: plt.show()
            except: pass
            plt.clf()
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function reports error, maybe this is a headless server? \nERROR: {}".format(e))
        return HRC