

import os, time
from mimircache import *





MINING_PERIOD1       = 5000
PREFETCH_TABLE1      = 30000
PREFETCH_LIST_SIZE1  = 2


MINING_PERIOD2       = 5000
PREFETCH_TABLE2      = 80000
PREFETCH_LIST_SIZE2  = 2





def get_r1(dat):
    result = []


    # reader = plainReader("/scratch/jason/msr_parda_64KB_aligned_rw/txt/{}".format(dat), data_type='l')
    reader = plainReader("/home/jason/ALL_DATA/cloudphysics_txt_16KB/w{}.txt".format(dat), data_type='l')

    # r = vscsiReader(dat, data_type='l')
    n = reader.get_num_of_total_requests()
    nu = reader.get_num_of_unique_requests()
    print("{}: total {}, uniq {}".format(dat, n, nu))
    CACHE_SIZE = nu//20
    BIN_SIZE = CACHE_SIZE
    COMPENSATE1 = 14*4        # 225
    COMPENSATE2 = 16
    COMPENSATE2 = int( ((8 + 8 + 8 * 3) * MINING_PERIOD2 + (8 + 8 * 2) * (0.02*nu)) / (16*1024))


    lp = LRUProfiler(reader)
    mr_LRU = lp.get_miss_rate()
    result.append(mr_LRU[CACHE_SIZE])

    cg_AMP = cGeneralProfiler(reader, "AMP",
                              CACHE_SIZE,
                              BIN_SIZE,
                              cache_params={"APT":4, "read_size":1},
                              num_of_threads=1)
    mr_AMP = cg_AMP.get_miss_rate()
    result.append(mr_AMP[1])

    cg_Mithril1 = cGeneralProfiler(reader, "mimir",
                                   CACHE_SIZE-COMPENSATE1,
                                   BIN_SIZE-COMPENSATE1,
                                   cache_params={"max_support": 12,
                                                 "min_support": 2,
                                                 "confidence": 0,
                                                 "item_set_size": 20,
                                                 "mining_period": MINING_PERIOD1,
                                                 "prefetch_list_size": PREFETCH_LIST_SIZE1,
                                                 "cache_type": "LRU",
                                                 "mining_period_type": 'v',
                                                 "sequential_type":1,
                                                 "sequential_K": 2,
                                                 "prefetch_table_size": PREFETCH_TABLE1
                                                 },
                                   num_of_threads=1)
    mr_M1 = cg_Mithril1.get_miss_rate()
    result.append(mr_M1[1])

    cg_Mithril2 = cGeneralProfiler(reader, "mimir",
                                   CACHE_SIZE-COMPENSATE2,
                                   BIN_SIZE-COMPENSATE2,
                                   cache_params={"max_support": 20,
                                                 "min_support": 1,
                                                 "confidence": 2,
                                                 "item_set_size": 20,
                                                 "mining_period": MINING_PERIOD2,
                                                 "prefetch_list_size": PREFETCH_LIST_SIZE2,
                                                 "cache_type": "AMP",
                                                 "mining_period_type": 'v',
                                                 "sequential_type":2,
                                                 "sequential_K": 1,
                                                 "prefetch_table_size": int(0.01 * nu)
                                                 },
                                   num_of_threads=1)
    mr_M2 = cg_Mithril2.get_miss_rate()
    result.append(mr_M2[1])

    return result


# def run_for_traces(reader):
#     n = reader.num_
#     p = cGeneralProfiler(reader, 'AMP', CACHE_SIZE, BIN_SIZE, num_of_threads=NUM_OF_THREADS)



if __name__ == "__main__":
    # reader = vscsiReader("../data/trace.vscsi")
    # f = '../mimircache/data/trace.vscsi'
    # for f in os.listdir("/scratch/jason/msr_parda_64KB_aligned_rw/txt"):
    import time
    # time.sleep(3600)
    for f in range(106, 0, -1):
        # print(f)
        r = get_r1(f)
        print("w{}: {}".format(f, r))