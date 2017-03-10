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

from mimircache.utils.printing import colorfulPrint
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
        self.cache_size = cache_size
        self.cache_name = cache_alg_mapping[cache_name.lower()]
        if bin_size == -1:
            self.bin_size = int(self.cache_size / DEFAULT_BIN_NUM_PROFILER)
            if self.bin_size == 0:
                self.bin_size =1
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
        if "prefetch" in kwargs and kwargs['prefetch']:
            return c_generalProfiler.get_hit_rate_with_prefetch(self.reader.cReader, self.cache_name, cache_size,
                                              self.bin_size, cache_params=self.cache_params, **sanity_kwargs)
        else:
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
        return c_generalProfiler.get_miss_rate(self.reader.cReader, self.cache_name, cache_size,
                                               self.bin_size, cache_params=self.cache_params, **sanity_kwargs)

    def plotMRC(self, figname="MRC.png", **kwargs):
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
            colorfulPrint("red", "plot is saved at the same directory")
            plt.show()
            plt.clf()
            del MRC
        except Exception as e:
            plt.savefig(figname)
            print("the plotting function is not wrong, is this a headless server?\n{}".format(e), file=sys.stderr)

    def plotHRC(self, figname="HRC.png", **kwargs):
        HRC = self.get_hit_rate(**kwargs)
        try:
            # tick = ticker.FuncFormatter(lambda x, pos: '{:2.0f}'.format(x * self.bin_size))
            # plt.gca().xaxis.set_major_formatter(tick)

            plt.xlim(0, self.cache_size)
            plt.plot(range(0, self.cache_size + 1, self.bin_size), HRC)
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.savefig(figname, dpi=600)
            colorfulPrint("red", "plot is saved at the same directory")
            plt.show()
            plt.clf()
            del HRC
        except Exception as e:
            plt.savefig(figname)
            print("the plotting function is not wrong, is this a headless server?")
            print(e)


    # def __del__(self):
    #     import os
    #     if (os.path.exists('temp.dat')):
    #         os.remove('temp.dat')


def server_plot_all(path='/run/shm/traces/', threads=48):
    from mimircache.profiler.LRUProfiler import LRUProfiler
    folder = '0628_HRC'
    for filename in os.listdir(path):
        print(filename)
        if filename.endswith('.vscsitrace'):
            reader = vscsiReader(path + filename)
            p1 = LRUProfiler(reader)
            size = p1.plotHRC(figname=folder + "/" + filename + '_LRU_HRC.png')

            if os.path.exists(folder + "/" + filename + '_LRU4_HRC_' + str(size) + '_.png'):
                reader.close()
                continue

            p2 = cGeneralProfiler(reader, 'Optimal', cache_size=size, bin_size=int(size / 500), num_of_threads=threads)
            p2.plotHRC(figname=folder + "/" + filename + '_Optimal_HRC_' + str(size) + '_.png')
            p3 = cGeneralProfiler(reader, "LRU_K", cache_size=size, cache_params={"K": 2}, bin_size=int(size / 500))
            p3.plotHRC(figname=folder + "/" + filename + '_LRU2_HRC_' + str(size) + '_.png', num_of_threads=threads)
            # p4 = cGeneralProfiler(reader, "LRU_K", cache_size=size, cache_params={"K": 3}, bin_size=int(size / 500))
            # p4.plotHRC(figname=folder + "/" + filename + '_LRU3_HRC_' + str(size) + '_.png', num_of_threads=threads)
            # p5 = cGeneralProfiler(reader, "LRU_K", cache_size=size, cache_params={"K": 4}, bin_size=int(size / 500))
            # p5.plotHRC(figname=folder + "/" + filename + '_LRU4_HRC_' + str(size) + '_.png', num_of_threads=threads)
            reader.close()






def run():
    from mimircache import cachecow
    MAX_SUPPORT = 8
    MIN_SUPPORT = 1
    CONFIDENCE = 0
    ITEM_SET_SIZE = 50       # 10 for w49~w106
    TRAINING_PERIOD = 1000000
    PREFETCH_LIST_SIZE = 3

    NUM_OF_THREADS = 48



    # for f in os.listdir("/scratch/jason/traces/"):
    if True:
        f = sys.argv[1][sys.argv[1].rfind('/')+1:]
        print(f)
        c = cachecow()
        if not f.endswith('vscsitrace'):
            sys.exit(1)
        if 'w01' in f:
            sys.exit(1)
        # c.vscsi("/scratch/jason/traces/{}".format(f))
        c.vscsi("/home/cloudphysics/traces/{}".format(f))
        print(sys.argv[1])
        n = c.num_of_request()
        CACHE_SIZE = n//100
        TRAINING_PERIOD = n//10
        figname = "mimir/HRC_{}_{}_{}_{}_{}_{}.png".format(f, MAX_SUPPORT, MIN_SUPPORT, CONFIDENCE, ITEM_SET_SIZE, TRAINING_PERIOD)

        if os.path.exists(figname):
            sys.exit(1)

        c.plotHRCs(["LRU", "mimir", "test1"],
               cache_params=[None, {"max_support": MAX_SUPPORT, "min_support": MIN_SUPPORT, "confidence": CONFIDENCE,
                                    "item_set_size": ITEM_SET_SIZE, "training_period": TRAINING_PERIOD,
                                    "prefetch_list_size": PREFETCH_LIST_SIZE}],
               cache_size=CACHE_SIZE, bin_size=int(CACHE_SIZE/95), auto_size=False, num_of_threads=NUM_OF_THREADS,
               figname=figname)


if __name__ == "__main__":
    import time
    DAT = "hm_0.csv"



    NUM_OF_THREADS = 8
    CACHE_SIZE = 8000
    BIN_SIZE = int(CACHE_SIZE/ NUM_OF_THREADS / 4) + 1



    from mimircache import *

    run_type = 1

    c = cachecow()
    # c.vscsi("../data/traces/{}_vscsi1.vscsitrace".format(DAT))
    if run_type ==1:
        c.vscsi("../data/trace.vscsi")
        # c.open("/home/jason/ALL_DATA/cloudphysics_txt_64K/{}.txt".format(DAT), data_type='l')         # 99 104 105
        # c.open("/disk/cloudphysics_txt_64K/{}.txt".format(DAT), data_type='l')         # 99 104 105
        #                                                                            # no 102 103 106

        # c.vscsi("../data/traces/w38_vscsi1.vscsitrace")
        # c.open("../data/trace.txt", data_type='l')

    if run_type == 2:
        # c.csv("/run/shm/wiki.1000000", init_params={"label_column": 3})
        c.open("/home/jason/ALL_DATA/redis/86400-sec-redis-ephemeral-cmds.anonymized.log.extract.get.clean")
        # c.open("/home/jason/setget.all.2")

    n = c.num_of_request()
    nu = c.reader.get_num_of_unique_requests()
    print("total " + str(n) + ", unique " + str(nu))
    CACHE_SIZE = nu // 200
    # CACHE_SIZE = 201
    BIN_SIZE = CACHE_SIZE//NUM_OF_THREADS+10
    # BIN_SIZE = CACHE_SIZE-1
    # TRAINING_PERIOD = n // 80
    TRAINING_PERIOD = 6000
    figname = "HRC_wiki.png"
    c.reader.reset()
    t1 = time.time()

    if run_type == 1:
        c.plotHRCs(["LRU", "AMP", "mimir", "PG", "Optimal"], # , "FIFO", "mimir"], #, "Optimal"], # ""test1"],  #, "Optimal"],
                   cache_params=[None, {"K":3, "pthreshold":256},
                                  {
                                    "max_support": 12,
                                    "min_support": 3,
                                    "confidence": 0,
                                    "item_set_size": 20,
                                    "prefetch_list_size": 2,
                                    "cache_type": "AMP",
                                    "sequential_type":2,
                                    "max_metadata_size": 0.2,
                                    "block_size":64*1024,
                                    "sequential_K":2,
                                    "cycle_time":2,
                                    "AMP_pthreshold":256,
                                   },
                                 {
                                     "lookahead": 2,
                                     "cache_type": "LRU",
                                     "max_metadata_size": 0.1,
                                     "prefetch_threshold": 0.2,
                                     "block_size": 64*1024,
                                 }, None
                                 ],
                   cache_size=CACHE_SIZE, bin_size=BIN_SIZE, auto_size=False, num_of_threads=NUM_OF_THREADS,
                   figname=figname)
    elif run_type == 2:
        c.plotHRCs(["LRU", "mimir", "FIFO", "mimir", "Optimal"], # ""test1"],  #, "Optimal"],
                   cache_params=[None, # {"APT":4, "read_size":1},
                                 {
                                     "max_support": 12,
                                     "min_support":4,
                                     "confidence": 0,
                                     "item_set_size": 2000,
                                     "prefetch_list_size": 2,
                                     "cache_type": "LRU",
                                     "sequential_type": 0,
                                     "max_metadata_size": 0.1,
                                     "block_size": 64 * 1024,
                                     "sequential_K": 0,
                                     "cycle_time": 2,
                                     "AMP_pthreshold":256
                                 },
                                 None,
                                 {
                                     "max_support": 12,
                                     "min_support": 4,
                                     "confidence": 0,
                                     "item_set_size": 2000,
                                     "prefetch_list_size": 2,
                                     "cache_type": "FIFO",
                                     "sequential_type": 0,
                                     "max_metadata_size": 0.2,
                                     "block_size": 64 * 1024,
                                     "sequential_K": 0,
                                     "cycle_time": 2,
                                     "AMP_pthreshold": 256
                                 },
                                    None
                                 ],
                   cache_size=CACHE_SIZE, bin_size=BIN_SIZE, auto_size=False, num_of_threads=NUM_OF_THREADS,
                   figname=figname)
    print("{} s".format(time.time() - t1))

'''
    # plt.xlim(xmin=800)
    c.plotMRCs(["LRU", "AMP", "MS2", "mimir"], # , "FIFO", "mimir"], #, "Optimal"], # ""test1"],  #, "Optimal"],
               cache_params=[None, {"APT":4, "read_size":1},
                             {"max_support": MAX_SUPPORT,
                              "min_support": MIN_SUPPORT,
                              "confidence": CONFIDENCE,
                              "item_set_size": ITEM_SET_SIZE,
                              "mining_period": TRAINING_PERIOD,
                              "prefetch_list_size": PREFETCH_LIST_SIZE,
                              "cache_type": "LRU",
                              "mining_period_type": 'v',
                              "sequential_K": 2,
                              "prefetch_table_size":200000,
                                "sequential_type": 1
                              },
                              {
                                "max_support": 20,
                               "min_support": 2,
                               "confidence": 1,
                               "item_set_size": 20,
                               "mining_period": TRAINING_PERIOD,
                               "prefetch_list_size": 2,
                               "mining_period_type": 'v',
                               "cache_type": "AMP",
                                  "sequential_type":2,
                               # "prefetch_table_size": 20000,
                                  "max_metadata_size": 0.2,
                                  "block_size":64*1024,
                                  "sequential_K":1,
                                "cycle_time":2,
                               }],
               cache_size=CACHE_SIZE, bin_size=BIN_SIZE, auto_size=False, num_of_threads=NUM_OF_THREADS,
               figname=figname)
'''

    # t1 = time.time()
    #
    # DAT = "w90"
    # # r = vscsiReader("../1a1a11a/prefetch_input_data/{}/xab".format(DAT))
    # r = csvReader("MSR/xab", init_params={"real_time_column": 1, "label_column": 5})
    # # r = vscsiReader("/scratch/jason/traces/w38_vscsi1.vscsitrace")
    # cg = cGeneralProfiler(r, 'LRU', CACHE_SIZE, BIN_SIZE, num_of_threads=NUM_OF_THREADS)
    # cg2 = cGeneralProfiler(r, 'test1', CACHE_SIZE, BIN_SIZE, num_of_threads=NUM_OF_THREADS)
    #
    # prefetch = False
    # figname = "{}_xab_non_prefetch.png".format(DAT)
    # hr1 = cg.get_hit_rate(prefetch=prefetch)
    # print("TIME: %f" % (time.time() - t1))
    # t1 = time.time()
    #
    #
    # prefetch = True
    # figname = "{}_xab_prefetch_all.png".format(DAT)
    # hr2 = cg.get_hit_rate(prefetch=prefetch)
    # print("TIME: %f" % (time.time() - t1))
    # t1 = time.time()
    #
    # prefetch = False
    # hr3 = cg2.get_hit_rate(prefetch=prefetch)
    # print("TIME: %f" % (time.time() - t1))
    # t1 = time.time()
    #
    # n  = r. get_num_of_total_requests()
    # nu = r.get_num_of_unique_requests()
    #
    # plt.xlim(0, CACHE_SIZE)
    # plt.plot((0, CACHE_SIZE), (1-nu/n, 1-nu/n), '-', label="cold miss")
    # plt.plot(range(0, CACHE_SIZE + 1, BIN_SIZE), hr1, label="no_prefetch")
    # plt.plot(range(0, CACHE_SIZE + 1, BIN_SIZE), hr2, label="prefetch")
    # plt.plot(range(0, CACHE_SIZE + 1, BIN_SIZE), hr3, label="test1")
    #
    #
    # plt.legend(loc="lower right")
    # plt.xlabel("cache Size")
    # plt.ylabel("Hit Rate")
    # plt.title('Hit Rate Curve', fontsize=18, color='black')
    # plt.savefig(figname, dpi=600)
    # colorfulPrint("red", "plot is saved at the same directory")




