# coding=utf-8

"""
    This module defines all the const and related func

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""


import os
import sys

# const
INSTALL_PHASE = False
ALLOW_C_MIMIRCACHE = True
INTERNAL_USE = True
DEF_NUM_BIN_PROF = 100
DEF_NUM_THREADS = os.cpu_count()
DEF_EMA_HISTORY_WEIGHT = 0.80

# try to import cMimircache
if not INSTALL_PHASE and ALLOW_C_MIMIRCACHE:
    failed_components = []
    failed_reason = set()
    try:
        import PyMimircache.CMimircache.CacheReader
    except Exception as e:
        failed_reason.add(e)
        failed_components.append("CMimircache.CacheReader")
    try:
        import PyMimircache.CMimircache.LRUProfiler
    except Exception as e:
        failed_reason.add(e)
        failed_components.append("CMimircache.LRUProfiler")
    try:
        import PyMimircache.CMimircache.GeneralProfiler
    except Exception as e:
        failed_reason.add(e)
        failed_components.append("CMimircache.GeneralProfiler")
    try:
        import PyMimircache.CMimircache.Heatmap
    except Exception as e:
        failed_reason.add(e)
        failed_components.append("CMimircache.Heatmap")

    if len(failed_components):
        ALLOW_C_MIMIRCACHE = False
        print("CMimircache components {} import failed, performance will degrade, "
              "reason: {}, ignore this warning if this is installation".
              format(", ".join(failed_components),
                     ", ".join(["{!s}".format(i) for i in failed_reason])),
              file=sys.stderr)


from PyMimircache.cache.arc import ARC
from PyMimircache.cache.fifo import FIFO
from PyMimircache.cache.lru import LRU
from PyMimircache.cache.lfu import LFU
from PyMimircache.cache.lfu2 import LFU
from PyMimircache.cache.mru import MRU
from PyMimircache.cache.optimal import Optimal
from PyMimircache.cache.random import Random
from PyMimircache.cache.s4lru import S4LRU
from PyMimircache.cache.slru import SLRU
from PyMimircache.cache.clock import Clock
from PyMimircache.cache.linuxclock import LinuxClock
from PyMimircache.cache.tear import Tear
from PyMimircache.cache.secondChance import SecondChance


from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader

# global C_AVAIL_CACHE (lower case)
C_AVAIL_CACHE = ["lru", "fifo", "optimal", "arc", "random",
                     "lfufast", "lfu", "mru",
                     "slru", "lru_k", "lru_2",
                     "mimir", "mithril", "amp", "pg ",
                 ]

C_AVAIL_CACHEREADER = [PlainReader, VscsiReader, CsvReader, BinaryReader]


CACHE_NAME_TO_CLASS_DICT = {"LRU":LRU, "LFU": LFU, "MRU":MRU, "ARC":ARC, "Optimal":Optimal,
                            "FIFO":FIFO, "Clock":Clock, "LinuxClock":LinuxClock,  "TEAR":Tear, "Random":Random, "SecondChance": SecondChance,
                            "SLRU":SLRU, "S4LRU":S4LRU
                            }

# used to mapping user provided cache name to a unified cache replacement alg name (Appearntly this is not a good idea)
CACHE_NAME_CONVRETER = {alg_name.lower(): alg_name for alg_name in CACHE_NAME_TO_CLASS_DICT.keys()}

CACHE_NAME_CONVRETER.update({
    "lru_k": "lru_k", "lru_2": "lru_2",
    "mithril": "Mithril", "amp": "AMP", "pg": "PG",
})

CACHE_NAME_CONVRETER.update( {"opt": "Optimal",
                        "rr": "Random",
                        })


def cache_name_to_class(cache_name):
    """
    used in PyMimircache
    convert cache replacement algorithm name to corresponding python cache class

    :param cache_name: name of cache
    :return: the class of given cache replacement algorithm
    """

    cache_standard_name = CACHE_NAME_CONVRETER.get(cache_name.lower(), cache_name)
    cache_class = CACHE_NAME_TO_CLASS_DICT.get(cache_standard_name, None)

    if cache_class:
        return cache_class
    else:
        raise RuntimeError("cannot recognize given cache replacement algorithm {}, "
                           "supported algorithms {}".format(cache_name, CACHE_NAME_CONVRETER.values()))


def add_new_cache_alg(name, cls):
    CACHE_NAME_CONVRETER[name.lower()] = name
    CACHE_NAME_TO_CLASS_DICT[name] = cls
    # print("add new cache replacement algorithm {}".format(name))


__all__ = ["ALLOW_C_MIMIRCACHE", "INTERNAL_USE", "DEF_NUM_BIN_PROF", "DEF_NUM_THREADS", "DEF_EMA_HISTORY_WEIGHT",
           "C_AVAIL_CACHE", "C_AVAIL_CACHEREADER", "CACHE_NAME_CONVRETER", "CACHE_NAME_TO_CLASS_DICT",
           "cache_name_to_class", "INSTALL_PHASE"]
