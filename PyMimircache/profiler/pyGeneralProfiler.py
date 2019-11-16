# coding=utf-8
""" this module is used for all cache replacement algorithms in python
    this module can be very slow, but you can easily plug in new algorithm.

    This module has just been re-written meaning it might contain bugs.

    TODO: use numda and other JIT technique to improve run time

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/07

"""
# -*- coding: utf-8 -*-


import os
import math
import numpy as np

from concurrent.futures import as_completed, ProcessPoolExecutor

from PyMimircache.const import *
from PyMimircache.const import cache_name_to_class
from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.utils.printing import *
from PyMimircache.profiler.profilerUtils import util_plotHRC, util_plotMRC

__all__ = ["PyGeneralProfiler"]


def _cal_hit_count_subprocess(cache_class,
                              cache_size,
                              reader_class,
                              reader_params,
                              cache_params=None):
    """
    subprocess for simulating a cache, this will be used as init func for simulating a cache,
    it reads data from reader and calculates the number of hits and misses

    :param cache_class: the __class__ attribute of cache, this will be used to create cache instance
    :param cache_size:  size of cache
    :param reader_class: the __class__ attribute of reader, this will be used to create local reader instance
    :param reader_params:   parameters for reader, used in creating local reader instance
    :param cache_params:    parameters for cache, used in creating cache
    :return: a tuple of number of hits and number of misses
    """

    # local reader and cache
    process_reader = reader_class(**reader_params)

    if cache_size == 0:
        return 0, process_reader.get_num_of_req()

    if cache_params is None:
        cache_params = {}
    if cache_class.__name__ == "Optimal":
        cache_params["reader"] = process_reader
    process_cache = cache_class(cache_size, **cache_params)
    n_hits = 0
    n_misses = 0

    req = process_reader.read_as_req_item()
    while req:
        hit = process_cache.access(req, )
        if hit:
            n_hits += 1
        else:
            n_misses += 1
        req = process_reader.read_as_req_item()

    process_reader.close()
    return n_hits, n_misses


class PyGeneralProfiler:
    """
    Python version of generalProfiler
    """

    all = ["get_hit_count", "get_hit_ratio", "plotMRC"]

    def __init__(self, reader, cache_class, cache_size=-1,
                 bin_size=-1, cache_params=None, **kwargs):

        self.cache_class = cache_class
        if isinstance(self.cache_class, str):
            self.cache_class = cache_name_to_class(self.cache_class)

        self.cache_params = cache_params
        self.reader = reader
        self.cache_size, self.bin_size = cache_size, bin_size
        self.cache_size_list = kwargs.get("cache_size_list", [])
        self.num_of_threads = kwargs.get("num_of_threads", DEF_NUM_THREADS)
        self.num_of_req = 0

        assert isinstance(reader, AbstractReader), \
            "you provided an invalid cacheReader: {}".format(reader)

        self.get_hit_rate = self.get_hit_ratio

        if len(self.cache_size_list) == 0:
            if self.bin_size == -1:
                assert self.cache_size != -1
                self.bin_size = self.cache_size
                self.cache_size_list.append(self.cache_size)
            else:
                for i in range(self.cache_size // self.bin_size + 1):
                    self.cache_size_list.append(i * self.bin_size)

        self.hit_count = np.zeros((len(self.cache_size_list),), dtype=np.longlong)
        self.hit_ratio = np.zeros((len(self.cache_size_list),), dtype=np.double)
        self.miss_ratio = np.zeros((len(self.cache_size_list),), dtype=np.double)

        self.has_ran = False

    @classmethod
    def get_classname(cls):
        """
        return class name
        :return:
        """
        return cls.__name__

    def _run(self):
        """
        simulate self.num_of_bins cache and calculate hit count and hit ratio for each cache size
        :return: True if succeed, else False
        """

        reader_params = self.reader.get_params()
        reader_params["open_c_reader"] = False

        self.num_of_req = self.reader.get_num_of_req()
        count = 0
        with ProcessPoolExecutor(max_workers=self.num_of_threads) as ppe:
            future_to_size_ind = {ppe.submit(_cal_hit_count_subprocess,
                                             self.cache_class, self.cache_size_list[idx],
                                             self.reader.__class__, reader_params,
                                             self.cache_params): idx \
                                  for idx in range(len(self.cache_size_list)-1, -1, -1)}
            last_print_ts = time.time()
            for future in as_completed(future_to_size_ind):
                result = future.result()
                idx = future_to_size_ind[future]
                if result:
                    assert sum(result) == self.num_of_req, \
                        "hit/miss {}/{}, trace length {}".format(result[0], result[1], self.num_of_req)
                    self.hit_count[idx] = result[0]
                    count += 1
                    if time.time() - last_print_ts > 20:
                        print("{}/{}".format(count, len(self.cache_size_list)), end="\r")
                        last_print_ts = time.time()
                else:
                    raise RuntimeError("failed to fetch results from cache of size {}".format(
                        self.cache_size_list[idx]
                    ))
        for i in range(len(self.hit_count)):
            self.hit_ratio[i] = self.hit_count[i] / self.num_of_req
            self.miss_ratio[i] = 1 - self.hit_ratio[i]

        # right now self.hit_count is a CDF array like hit_ratio, now we transform it into non-CDF
        for i in range(len(self.cache_size_list)-1, 0, -1):
            self.hit_count[i] = self.hit_count[i] - self.hit_count[i - 1]

        self.has_ran = True
        return True

    def get_hit_count(self, **kwargs):
        """
        obtain hit count at cache size [0, bin_size, bin_size*2 ...]
        .. NOTICE: the hit count array is not a CDF, while hit ratio array is CDF

        :param kwargs: not used now
        :return: a numpy array, with hit count corresponding to size [0, bin_size, bin_size*2 ...]
        """

        if not self.has_ran:
            self._run()
        return self.hit_count

    def get_hit_ratio(self, **kwargs):
        """
        obtain hit ratio at cache size [0, bin_size, bin_size*2 ...]

        :param kwargs: not used now
        :return: a numpy array, with hit ratio corresponding to size [0, bin_size, bin_size*2 ...]
        """

        if not self.has_ran:
            self._run()
        return self.hit_ratio

    def get_miss_ratio(self, **kwargs):
        """
        obtain hit ratio at cache size [0, bin_size, bin_size*2 ...]

        :param kwargs: not used now
        :return: a numpy array, with hit ratio corresponding to size [0, bin_size, bin_size*2 ...]
        """

        if not self.has_ran:
            self._run()
        return self.miss_ratio

    def plotHRC(self, **kwargs):
        """
        plot hit ratio curve of the given trace under given algorithm

        :param kwargs: figname, cache_unit_size (unit: Byte), no_clear, no_save
        :return:
        """

        if not self.has_ran:
            self._run()

        dat_name = os.path.basename(self.reader.file_loc)
        kwargs["figname"] = kwargs.get("figname", "HRC_{}.png".format(dat_name))
        kwargs["label"] = kwargs.get("label", self.cache_class.__name__)

        util_plotHRC(self.cache_size_list, self.hit_ratio, **kwargs)

        return self.hit_ratio

    def plotMRC(self, **kwargs):
        """
        plot miss ratio curve of the given trace under given algorithm

        :param kwargs: figname, cache_unit_size (unit: Byte), no_clear, no_save
        :return:
        """

        if not self.has_ran:
            self._run()

        dat_name = os.path.basename(self.reader.file_loc)
        kwargs["figname"] = kwargs.get("figname", "MRC_{}.png".format(dat_name))
        kwargs["label"] = kwargs.get("label", self.cache_class.__name__)
        util_plotMRC(self.cache_size_list, self.miss_ratio, **kwargs)

        return self.miss_ratio
