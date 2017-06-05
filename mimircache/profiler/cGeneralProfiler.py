# coding=utf-8
""" this module is used for all other cache replacement algorithms excluding LRU(LRU also works, but slow compared to
    using pardaProfiler),
"""
# -*- coding: utf-8 -*-


import math
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mimircache.utils.printing import *
import mimircache.c_generalProfiler as c_generalProfiler
from mimircache.const import *


class cGeneralProfiler:
    def __init__(self, reader, cache_name, cache_size, bin_size=-1, cache_params=None,
                 num_of_threads=DEFAULT_NUM_OF_THREADS):
        assert cache_name.lower() in cache_alg_mapping, "please check your cache replacement algorithm: " + cache_name
        assert cache_name.lower() in c_available_cache, \
            "cGeneralProfiler currently only available on the following caches: {}\n, " \
            "please use generalProfiler".format(c_available_cache)

        self.reader = reader
        assert cache_size != 0, "you cannot provide size 0"
        self.cache_size = cache_size
        self.cache_name = cache_alg_mapping[cache_name.lower()]
        if bin_size == -1:
            self.bin_size = int(self.cache_size / DEFAULT_BIN_NUM_PROFILER)
        else:
            self.bin_size = bin_size

        if self.bin_size == 0:
            self.bin_size = 1

        self.cache_params = cache_params
        self.num_of_threads = num_of_threads

        if cache_params is not None and 'block_unit_size' in cache_params:
            self.with_size = True
        else:
            self.with_size = False

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
        self.reader = plainReader('temp.dat')


    def get_hit_count(self, **kwargs):
        """

        :return: a numpy array, with hit count corresponding to size [0, bin_size, bin_size*2 ...]
        """
        sanity_kwargs = {}
        if 'num_of_threads' not in kwargs:
            sanity_kwargs['num_of_threads'] = self.num_of_threads
        else:
            sanity_kwargs['num_of_threads'] = kwargs['num_of_threads']
        if 'cache_size' in kwargs:
            cache_size = kwargs['cache_size']
        else:
            cache_size = self.cache_size
        if 'begin' in kwargs:
            sanity_kwargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            sanity_kwargs['end'] = kwargs['end']
        if self.with_size:
            print("not supported yet")
        else:
            return c_generalProfiler.get_hit_count(self.reader.cReader, self.cache_name, cache_size,
                                                   self.bin_size, cache_params=self.cache_params, **sanity_kwargs)

    def get_hit_rate(self, **kwargs):
        """

        :return: a numpy array, with hit rate corresponding to size [0, bin_size, bin_size*2 ...]
        """
        sanity_kwargs = {}
        if 'num_of_threads' not in kwargs:
            sanity_kwargs['num_of_threads'] = self.num_of_threads
        else:
            sanity_kwargs['num_of_threads'] = kwargs['num_of_threads']
        if 'cache_size' in kwargs:
            cache_size = kwargs['cache_size']
        else:
            cache_size = self.cache_size
        if 'begin' in kwargs:
            sanity_kwargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            sanity_kwargs['end'] = kwargs['end']
        # handles both withsize and no size
        return c_generalProfiler.get_hit_rate(self.reader.cReader, self.cache_name, cache_size,
                                                        self.bin_size, cache_params=self.cache_params, **sanity_kwargs)



    def get_miss_rate(self, **kwargs):
        """

        :return: a numpy array, with miss rate corresponding to size [0, bin_size, bin_size*2 ...]
        """

        sanity_kwargs = {}
        if 'num_of_threads' not in kwargs:
            sanity_kwargs['num_of_threads'] = self.num_of_threads
        if 'cache_size' in kwargs:
            cache_size = kwargs['cache_size']
        else:
            cache_size = self.cache_size
        if 'begin' in kwargs:
            sanity_kwargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            sanity_kwargs['end'] = kwargs['end']
        if self.with_size:
            print("not supported yet")
        else:
            return c_generalProfiler.get_miss_rate(self.reader.cReader, self.cache_name, cache_size,
                                                   self.bin_size, cache_params=self.cache_params, **sanity_kwargs)

    def plotMRC(self, figname="MRC.png", **kwargs):
        print("function not updated")
        MRC = self.get_miss_rate(**kwargs)
        try:
            # tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(x * self.bin_size))
            # plt.gca().xaxis.set_major_formatter(tick)
            plt.xlim(0, self.cache_size)
            plt.plot(range(0, self.cache_size + 1, self.bin_size), MRC)
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=600)
            INFO("plot is saved at the same directory")
            plt.show()
            plt.clf()
            del MRC
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function is not wrong, is this a headless server? {}".format(e))

    def plotHRC(self, figname="HRC.png", **kwargs):
        HRC = self.get_hit_rate(**kwargs)
        try:
            plt.xlim(0, self.cache_size)
            plt.plot(range(0, self.cache_size + 1, self.bin_size), HRC)
            if 'block_unit_size' in self.cache_params:
                plt.xlabel("Cache Size (MB)")
                plt.gca().xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: int(x * self.cache_params['block_unit_size'] // 1024 // 1024)))
            else:
                plt.xlabel("Cache Size")
            plt.ylabel("Hit Ratio")
            plt.title('Hit Ratio Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=600)
            INFO("plot is saved at the same directory")
            # plt.show()
            plt.clf()
            del HRC
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function is not wrong, is this a headless server? {}".format(e))




    # def __del__(self):
    #     import os
    #     if (os.path.exists('temp.dat')):
    #         os.remove('temp.dat')


