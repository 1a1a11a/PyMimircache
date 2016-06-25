import os
import configparser
from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader

# global c_available_cache
c_available_cache = []
c_available_cacheReader = [plainCacheReader, vscsiCacheReader, csvCacheReader]
cache_alg_mapping = {}
BASE_DIR = os.path.dirname(__file__)

DEFAULT_BIN_NUM_PROFILER = 100
DEFAULT_NUM_OF_PROCESS = 4


def init():
    _init_C_available_modules()
    _init_cache_alg_mapping()


def _init_C_available_modules():
    config = configparser.ConfigParser()
    # print(BASE_DIR + '/conf')
    config.read(BASE_DIR + '/conf.py')
    if 'C_available_cache' in config.sections():
        c_available_cache.extend(config['C_available_cache'])
    else:
        raise RuntimeWarning("cannot find any cache module in C")

        # if 'C_available_reader' in config.sections():
        #     c_available_cacheReader.extend(config['C_available_reader'])
        # else:
        #     raise RuntimeWarning("cannot find any cacheReader module in C")





def _init_cache_alg_mapping():
    """
    match all possible cache replacement algorithm names(lower case) to available cache replacement algorithms
    :return:
    """

    cache_alg_mapping['optimal'] = 'Optimal'
    cache_alg_mapping['rr'] = "Random"
    cache_alg_mapping['lru'] = "LRU"
    cache_alg_mapping['fifo'] = "FIFO"
    cache_alg_mapping['arc'] = "ARC"
    cache_alg_mapping['clock'] = "clock"
    cache_alg_mapping['mru'] = "MRU"
    cache_alg_mapping['slru'] = "SLRU"
    cache_alg_mapping['s4lru'] = "S4LRU"
    cache_alg_mapping['lfu_rr'] = "LFU_RR"
    cache_alg_mapping['lfu_mru'] = "LFU_MRU"
