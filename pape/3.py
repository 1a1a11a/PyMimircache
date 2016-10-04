

import copy
from mimircache import *

NUM_OF_THREADS = 8
DAT = "w81"  # 83


MAX_SUPPORT = 20
MIN_SUPPORT = 2
CONFIDENCE = 0
ITEM_SET_SIZE = 20
PREFETCH_LIST_SIZE = 2
MINING_PERIOD = 2000
SEQUENTIAL_K = 1


c = cachecow()
c.open("/home/jason/ALL_DATA/cloudphysics_txt_64K/{}.txt".format(DAT), data_type='l')  # 99 104 105

n = c.num_of_request()
nu = c.reader.get_num_of_unique_requests()
print("total " + str(n) + ", unique " + str(nu))

CACHE_SIZE = n // 200
BIN_SIZE = CACHE_SIZE // NUM_OF_THREADS //32 + 10
TRAINING_PERIOD = 5000

figname = "varying_max_support.png".format(DAT, MAX_SUPPORT, MIN_SUPPORT, CONFIDENCE, ITEM_SET_SIZE,
                                                      TRAINING_PERIOD, PREFETCH_LIST_SIZE)
c.reader.reset()

mimir_params = {"max_support": MAX_SUPPORT, "min_support": MIN_SUPPORT, "confidence": CONFIDENCE,
                "item_set_size": ITEM_SET_SIZE, "mining_period": MINING_PERIOD,
                "prefetch_list_size": PREFETCH_LIST_SIZE, "cache_type": "LRU", "mining_period_type": 'v',
                "sequential_K": SEQUENTIAL_K, "sequential_type": 1, "prefetch_table_size": 20000 }

cache_params = [None, {"APT": 4, "read_size": 1}]

# for i in [12]:
#     p = copy.deepcopy(mimir_params)
#     p["max_support"] = i
#     cache_params.append(p)




# c.plotMRCs(["LRU", "AMP", "mimir", "mimir", "mimir", "mimir", "mimir"], cache_params=cache_params,
#            cache_size=CACHE_SIZE, bin_size=BIN_SIZE, auto_size=False, num_of_threads=NUM_OF_THREADS,
#            figname=figname, ymin=0.05,
#            legend=["LRU", "AMP", "mimir-2", "mimir-4", "mimir-6", "mimir-8", "mimir-12", "mimir-16"])


p = copy.deepcopy(mimir_params)
p["min_support"] = 1
p["sequential_K"] = 1
cache_params.append(p)

p = copy.deepcopy(mimir_params)
p["sequential_K"] = 2
p["min_support"] = 2
cache_params.append(p)


c.plotMRCs(["LRU", "AMP", "mimir", "mimir"], cache_params=cache_params,
           cache_size=CACHE_SIZE, bin_size=BIN_SIZE, auto_size=False, num_of_threads=NUM_OF_THREADS,
           figname=figname, ymin=0.06, xlabel="cache size/64K block",
           legend=["LRU", "AMP", "MS1", "MS2"])