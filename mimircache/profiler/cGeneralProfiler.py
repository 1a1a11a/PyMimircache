""" this module is used for all other cache replacement algorithms excluding LRU(LRU also works, but slow compared to
    using pardaProfiler),
"""
# -*- coding: utf-8 -*-


import math
import os

# deal with headless situation
import traceback

import matplotlib
# matplotlib.use('Agg')

from multiprocessing import Process, Pipe, Array
from os import path

from mimircache.cacheReader.vscsiReader import vscsiCacheReader

from mimircache.cache.ARC import ARC
from mimircache.cache.clock import clock
from mimircache.cache.FIFO import FIFO
from mimircache.cache.LFU_LRU__NEED_OPTIMIZATION import LFU_LRU
from mimircache.cache.LFU_MRU import LFU_MRU
from mimircache.cache.LFU_RR import LFU_RR
from mimircache.cache.LRU import LRU
from mimircache.cache.MRU import MRU
from mimircache.cache.Random import Random
from mimircache.cache.SLRU import SLRU
from mimircache.cache.S4LRU import S4LRU
from mimircache.cache.Optimal import optimal

from mimircache.cacheReader.plainReader import plainCacheReader

import mimircache.c_generalProfiler as c_generalProfiler

import matplotlib.pyplot as plt

from mimircache.const import c_available_cache, cache_alg_mapping
from mimircache.profiler.abstract.abstractProfiler import profilerAbstract
from mimircache.const import *


class cGeneralProfiler:
    def __init__(self, reader, cache_name, cache_size, bin_size=-1, cache_params=None,
                 num_of_threads=DEFAULT_NUM_OF_PROCESS):
        assert cache_name.lower() in cache_alg_mapping, "please check your cache replacement algorithm: " + cache_name
        assert cache_name.lower() in c_available_cache, \
            "cGeneralProfiler currently only available on the following caches: {}\n, " \
            "please use generalProfiler".format(c_available_cache)

        self.reader = reader
        self.cache_size = cache_size
        self.cache_name = cache_alg_mapping[cache_name.lower()]
        if bin_size == -1:
            self.bin_size = int(self.cache_size / DEFAULT_BIN_NUM_PROFILER)
        else:
            self.bin_size = bin_size
        self.cache_params = cache_params
        self.num_of_threads = num_of_threads

        # if the given file is not basic reader, needs conversion
        need_convert = True
        for instance in c_available_cacheReader:
            if isinstance(reader, instance):
                need_convert = False
                break
        if need_convert:
            self.prepare_file()

    all = ["get_hit_count", "get_hit_rate", "get_miss_rate", "plotMRC", "plotHRC"]

    def prepare_file(self):
        self.num_of_lines = 0
        with open('temp.dat', 'w') as ofile:
            i = self.reader.read_one_element()
            while i is not None:
                self.num_of_lines += 1
                ofile.write(str(i) + '\n')
                i = self.reader.read_one_element()
        self.reader = plainCacheReader('temp.dat')


    def get_hit_count(self, **kwargs):
        """

        :return: a numpy array, with hit count corresponding to size [0, bin_size, bin_size*2 ...]
        """

        return c_generalProfiler.get_hit_count(self.reader.cReader, self.cache_name, self.cache_size, \
                                               self.bin_size, num_of_threads=self.num_of_threads, **kwargs)

    def get_hit_rate(self, **kwargs):
        """

        :return: a numpy array, with hit rate corresponding to size [0, bin_size, bin_size*2 ...]
        """

        return c_generalProfiler.get_hit_rate(self.reader.cReader, self.cache_name, self.cache_size, \
                                              self.bin_size, num_of_threads=self.num_of_threads, **kwargs)

    def get_miss_rate(self, **kwargs):
        """

        :return: a numpy array, with miss rate corresponding to size [0, bin_size, bin_size*2 ...]
        """

        return c_generalProfiler.get_miss_rate(self.reader.cReader, self.cache_name, self.cache_size, \
                                               self.bin_size, num_of_threads=self.num_of_threads, **kwargs)

    def plotMRC(self, figname="MRC.png", **kwargs):
        MRC = self.get_miss_rate(**kwargs)
        try:
            num_of_blocks = len(MRC)

            plt.plot(MRC)
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=300)
            plt.show()
            plt.clf()
        except Exception as e:
            plt.savefig(figname)
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def plotHRC(self, figname="HRC.png", **kwargs):
        HRC = self.get_hit_rate(**kwargs)
        try:
            num_of_blocks = len(HRC)

            plt.plot(HRC)
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=600)
            plt.show()
            plt.clf()
        except Exception as e:
            plt.savefig(figname)
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def __del__(self):
        if (os.path.exists('temp.dat')):
            os.remove('temp.dat')


if __name__ == "__main__":
    import time

    # t1 = time.time()
    # r = plainCacheReader('../../data/test')
    # r = plainCacheReader('../data/parda.trace')
    r = vscsiCacheReader('../data/trace.vscsi')
    cg = cGeneralProfiler(r, 'Optimal', 2000, 200)

    t1 = time.time()

    print(cg.get_hit_rate())
    print(cg.get_hit_count())
    print(cg.get_miss_rate())

    t2 = time.time()
    print("TIME: %f" % (t2 - t1))

    print(cg.plotMRC())
    print(cg.get_hit_rate(begin=1, end=20))
