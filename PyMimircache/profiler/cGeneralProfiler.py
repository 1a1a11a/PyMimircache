# coding=utf-8
"""
    this module is used for non-LRU cache replacement algorithms (of course, LRU also works)
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


import os
import math
from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.utils.printing import *
from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE
if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.GeneralProfiler as c_generalProfiler
from PyMimircache.const import *
# from PyMimircache.const import CACHE_NAME_CONVRETER, INSTALL_PHASE, ALLOW_C_MIMIRCACHE, DEF_NUM_BIN_PROF, DEF_NUM_THREADS, DEF_EMA_HISTORY_WEIGHT
from PyMimircache.profiler.profilerUtils import util_plotHRC


class CGeneralProfiler:
    """
    generalProfiler of C lang version
    """
    all = ["get_hit_count",
           "get_hit_ratio",
           "plotHRC"]

    def __init__(self, reader, cache_alg, cache_size,
                 bin_size=-1, num_of_bins=-1, cache_params=None, **kwargs):

        """
        initialization of a cGeneralProfiler
        :param reader:
        :param cache_alg:
        :param cache_size:
        :param bin_size: the sample granularity, the smaller the better, but also much longer run time
        :param cache_params: parameters about the given cache replacement algorithm
        :param kwargs: num_of_threads
        """

        # make sure reader is valid
        self.reader = reader
        self.cache_size = cache_size
        self.cache_name = CACHE_NAME_CONVRETER[cache_alg.lower()]
        self.bin_size = bin_size
        self.num_of_bins = num_of_bins
        self.hit_count = None
        self.hit_ratio = None

        assert isinstance(reader, AbstractReader), \
            "you provided an invalid cacheReader: {}".format(reader)

        assert cache_alg.lower() in CACHE_NAME_CONVRETER, \
            "please check your cache replacement algorithm: " + cache_alg
        assert cache_alg.lower() in C_AVAIL_CACHE, \
            "cGeneralProfiler currently only available on the following caches: {}\n, " \
            "please use generalProfiler".format(pformat(C_AVAIL_CACHE))

        assert self.bin_size == -1 or self.num_of_bins == -1, \
            "please don't specify bin_size ({}) and num_of_bins ({}) at the same time".format(self.bin_size, self.num_of_bins)
        assert isinstance(self.cache_size, int) and self.cache_size > 0, \
            "cache size {} is not valid for {}".format(cache_size, self.get_classname())


        if self.bin_size == -1:
            if self.num_of_bins == -1:
                self.num_of_bins = DEF_NUM_BIN_PROF
            self.bin_size = int(math.ceil(self.cache_size / self.num_of_bins)) # this guarantees bin_size >= 1
        else:
            self.num_of_bins = int(math.ceil(self.cache_size / self.bin_size))

        self.cache_params = cache_params
        if self.cache_params is None:
            self.cache_params = {}
        self.num_of_threads = kwargs.get("num_of_threads", DEF_NUM_THREADS)

        # check whether user want to profling with size
        self.block_unit_size = self.cache_params.get("block_unit_size", 0)
        block_unit_size_names = {"unit_size", "block_size", "chunk_size"}
        for name in block_unit_size_names:
            if name in self.cache_params:
                self.block_unit_size = cache_params[name]
                break

        # if the given file is not embedded reader, needs conversion for C backend
        need_convert = True
        for instance in C_AVAIL_CACHEREADER:
            if isinstance(reader, instance):
                need_convert = False
                break
        if need_convert:
            self._prepare_file()

        # this is for deprecated functions, as old version use hit rate instead of hit ratio
        self.get_hit_rate = self.get_hit_ratio


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
            i = self.reader.read_one_req()
            while i is not None:
                self.num_of_lines += 1
                ofile.write(str(i) + '\n')
                i = self.reader.read_one_req()
        self.reader = PlainReader('.temp.dat')


    @classmethod
    def get_classname(cls):
        """
        return the name of class
        :return: a string of classname
        """
        return cls.__name__


    def get_eviction_age(self, **kwargs):
        """
        obtain the eviction age of each req under specified cache replacement algorithm and cache size
        :param kwargs: Not used
        :return: a two-dimensional numpy array with shape num_of_bin * num_of_req,
        each row is an array of eviction age corresponding to cache_size = bin_size * row_id (beginning from 0),
        for example, the second row is the eviction age of cache_size = bin_size * (2-1),
        the nth element of the row corresponds to the eviction age of nth request,
        if the nth request is reused before evicted or not evicted (like ending of trace), then the
        eviction age is -1.
        .. NOTE: the first row will always be an array of zero because it corresponds to size zero.
        """

        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)
        bin_size = kwargs.get("bin_size", self.bin_size)
        cache_params = kwargs.get("cache_params", self.cache_params)

        return c_generalProfiler.get_eviction_age(self.reader.c_reader,
                                                              self.cache_name,
                                                              cache_size,
                                                              bin_size,
                                                              cache_params=cache_params,
                                                              **sanity_kwargs)


    def get_hit_result(self, **kwargs):
        """
        obtain the hit result (whether it is a hit or miss) under specified cache replacement algorithm and cache size
        :param kwargs: Not used
        :return: a two-dimensional numpy array with shape num_of_bin * num_of_req,
        each row is an array of True/False (hit_result) corresponding to cache_size = bin_size * row_id (beginning from 0),
        for example, the second row is hit result of cache_size = bin_size * (2-1),
        the nth element of the row corresponds to the whether it is hit or miss of nth request,
        if the nth request is a hit, then it is True, otherwise False.
        .. NOTE: the first row will always be an array of False because it corresponds to size zero.
        """

        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)
        bin_size = kwargs.get("bin_size", self.bin_size)
        cache_params = kwargs.get("cache_params", self.cache_params)

        return c_generalProfiler.get_hit_result(self.reader.c_reader,
                                                  self.cache_name,
                                                  cache_size,
                                                  bin_size,
                                                  cache_params=cache_params,
                                                  **sanity_kwargs)


    def get_hit_count(self, **kwargs):
        """
        obtain hit count at cache size [0, bin_size, bin_size*2 ...]
        .. NOTICE: the hit count array is not a CDF, while hit ratio array is CDF

        :return: a numpy array, with hit count corresponding to size [0, bin_size, bin_size*2 ...]
        """

        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)
        bin_size = kwargs.get("bin_size", self.bin_size)
        cache_params = kwargs.get("cache_params", self.cache_params)

        if self.block_unit_size != 0:
            print("not supported yet")
        else:
            self.hit_count = c_generalProfiler.get_hit_count(self.reader.c_reader,
                                                              self.cache_name,
                                                              cache_size,
                                                              bin_size,
                                                              cache_params=cache_params,
                                                              **sanity_kwargs)
        return self.hit_count

    def get_hit_ratio(self, **kwargs):
        """
        obtain hit ratio at cache size [0, bin_size, bin_size*2 ...]

        :return: a numpy array, with hit rate corresponding to size [0, bin_size, bin_size*2 ...]
        """

        sanity_kwargs = {"num_of_threads": kwargs.get("num_of_threads", self.num_of_threads)}
        cache_size = kwargs.get("cache_size", self.cache_size)
        bin_size = kwargs.get("bin_size", self.bin_size)
        cache_params = kwargs.get("cache_params", self.cache_params)

        # handles both withsize and no size, but currently only storage system trace are supported with size
        self.hit_ratio = c_generalProfiler.get_hit_ratio(self.reader.c_reader,
                                                          self.cache_name,
                                                          cache_size,
                                                          bin_size,
                                                          cache_params=cache_params,
                                                          **sanity_kwargs)
        return self.hit_ratio


    def plotHRC(self, **kwargs):
        """
        plot hit ratio curve of the given trace under given algorithm
        :param kwargs: figname, cache_unit_size (unit: Byte), no_clear, no_save
        :return:
        """

        dat_name = os.path.basename(self.reader.file_loc)
        kwargs["figname"] = kwargs.get("figname", "HRC_{}.png".format(dat_name))
        kwargs["label"] = kwargs.get("label", self.cache_name)

        self.get_hit_ratio(**kwargs)
        hit_ratio_size_list = [self.bin_size * i for i in range(self.num_of_bins + 1)]
        util_plotHRC(hit_ratio_size_list, self.hit_ratio, **kwargs)


        return self.hit_ratio