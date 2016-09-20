

import time, os, sys
from mimircache import *




NUM_OF_THREADS = 1

MAX_SUPPORT = 4
MIN_SUPPORT = 2
CONFIDENCE = 1
ITEM_SET_SIZE = 80
PREFETCH_LIST_SIZE = 2


def f1(dat, real_cache_size):

    c = cachecow()
    c.vscsi("../mimircache/data/traces/{}_vscsi1.vscsitrace".format(dat))
        # c.open("../1a1a11a/mix_dat.txt")
    # c.vscsi("../data/trace.vscsi")
    # c.csv("MSR/"+DAT, init_params={"real_time_column": 0, "label_column": 4})
    #     # c.csv("../data/trace.csv", init_params={"real_time_column": 1, "label_column": 4})
    n = c.num_of_request()
    nu = c.reader.get_num_of_unique_requests()
    ave_size = c.reader.get_average_size()
    print("total " + str(n) + ", unique " + str(nu) + ", average size "+ str(ave_size))

    all_cache_size = []

    for r_size in real_cache_size:
        CACHE_SIZE = int(r_size * 1024 * 1024 *1024 / ave_size)
        all_cache_size.append(CACHE_SIZE)
        BIN_SIZE = CACHE_SIZE
        MINING_PERIOD = n // 3
        CACHE_TYPE = "LRU"
        MINING_PERIOD_TYPE = 'v'


        c.reset()
        p = cGeneralProfiler(c.reader, "mimir",
                             cache_params={"max_support": MAX_SUPPORT,
                                           "min_support": MIN_SUPPORT,
                                           "confidence": CONFIDENCE,
                                          "item_set_size": ITEM_SET_SIZE,
                                           "mining_period": MINING_PERIOD,
                                          "prefetch_list_size": PREFETCH_LIST_SIZE,
                                           "cache_type": CACHE_TYPE,
                                           "mining_period_type": MINING_PERIOD_TYPE},
                             cache_size=CACHE_SIZE,
                             bin_size=BIN_SIZE,
                             num_of_threads=NUM_OF_THREADS)
        print("size: {}GB, miss rate: {}".format(r_size, p.get_miss_rate()[1]))
        # p.plotHRC()

    c.reader.reset()
    p2 = LRUProfiler(c.reader)
    hr = p2.get_hit_rate()
    for size in all_cache_size:
        print("size {}, hit rate {}".format(size, hr[size]))


if __name__ == "__main__":
    # REAL_CACHE_SIZE_w15 = [0.125, 0.5, 2, 8, 32] # , 8, 32]      # GB
    REAL_CACHE_SIZE_w102 = [0.0675, 0.125, 0.25, 0.5, 1, 2] # , 8, 32]      # GB

    REAL_CACHE_SIZE = REAL_CACHE_SIZE_w102
    f1("w102", REAL_CACHE_SIZE)