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
from PyMimircache.cache.mru import MRU
from PyMimircache.cache.optimal import Optimal
from PyMimircache.cache.random import Random
from PyMimircache.cache.s4lru import S4LRU
from PyMimircache.cache.slru import SLRU
from PyMimircache.cache.clock import Clock
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


CACHE_NAME_TO_CLASS_DICT = {"LRU":LRU, "MRU":MRU, "ARC":ARC, "Optimal":Optimal,
                            "FIFO":FIFO, "Clock":Clock, "Random":Random, "SecondChance": SecondChance,
                            "SLRU":SLRU, "S4LRU":S4LRU,
                            }

# used to mapping user provided cache name to a unified cache replacement alg name (Appearntly this is not a good idea)
CACHE_NAME_CONVRETER = {}
for alg_name in CACHE_NAME_TO_CLASS_DICT.keys():
    CACHE_NAME_CONVRETER[alg_name.lower()] = alg_name

CACHE_NAME_CONVRETER.update({
    "lru": "LRU", "fifo": "FIFO", "optimal": "Optimal",
    "arc": "ARC", "random": "Random", "mru": "MRU", "slru": "SLRU",
    "lru_k": "LRU_K", "lru_2": "LRU_2", "mimir": "mimir", "mithril": "Mithril", "amp": "AMP", "pg": "PG",
})

CACHE_NAME_CONVRETER.update( {"opt": "Optimal",
                        "rr": "Random",
                        "lfu": "LFUFast", "lfu_fast": "LFUFast", "lfufast": "LFUFast",
                        } )


def cache_name_to_class(cache_name):
    """
    used in PyMimircache
    convert cache replacement algorithm name to corresponding python cache class

    :param cache_name: name of cache
    :return: the class of given cache replacement algorithm
    """
    cache_class = None
    if cache_name.lower() in CACHE_NAME_CONVRETER:
        cache_standard_name = CACHE_NAME_CONVRETER[cache_name.lower()]
        cache_class = CACHE_NAME_TO_CLASS_DICT[cache_standard_name]

    if cache_class:
        return cache_class
    else:
        raise RuntimeError("cannot recognize given cache replacement algorithm {}, "
                           "supported algorithms {}".format(cache_name, CACHE_NAME_CONVRETER.values()))


__all__ = ["ALLOW_C_MIMIRCACHE", "INTERNAL_USE", "DEF_NUM_BIN_PROF", "DEF_NUM_THREADS", "DEF_EMA_HISTORY_WEIGHT",
           "C_AVAIL_CACHE", "C_AVAIL_CACHEREADER", "CACHE_NAME_CONVRETER", "CACHE_NAME_TO_CLASS_DICT",
           "cache_name_to_class", "INSTALL_PHASE"]
