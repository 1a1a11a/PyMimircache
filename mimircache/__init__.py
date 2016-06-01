""" mimircache for analyzing cache traces.

    developed by Juncheng Yang and Ymir group at Emory University
.. moduleauthor:: Juncheng Yang <peter.waynechina@gmail.com>

"""

import matplotlib

matplotlib.use('Agg')

from _version import __version__

import logging

# from mimircache.oldModule.basicLRUProfiler import basicLRUProfiler as basicLRUProfiler
from mimircache.cache.LRU import LRU as LRU
from mimircache.cacheReader.csvReader import csvCacheReader as csvReader
from mimircache.cacheReader.plainReader import plainCacheReader as plainReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader as vscsiReader
from mimircache.profiler.pardaProfiler import pardaProfiler as pardaProfiler
from mimircache.profiler.generalProfiler import generalProfiler as generalProfiler

from mimircache.profiler.heatmap import heatmap as heatmap

from mimircache.profiler.pardaProfiler import parda_mode as parda_mode
from mimircache.top.cachecow import cachecow as cachecow

logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)

# print(__version__)
