""" mimircache a cache trace analysis platform.

.. moduleauthor:: Juncheng Yang <peter.waynechina@gmail.com>, Ymir Vigfusson

"""

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

# import logging

from mimircache.cache.LRU import LRU as LRU
from mimircache.cacheReader.csvReader import csvCacheReader as csvReader
from mimircache.cacheReader.plainReader import plainCacheReader as plainReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader as vscsiReader
# from mimircache.oldModule.pardaProfiler import pardaProfiler as pardaProfiler
# from mimircache.oldModule.pardaProfiler import parda_mode as parda_mode
from mimircache.profiler.LRUProfiler import LRUProfiler as LRUProfiler
from mimircache.profiler.generalProfiler import generalProfiler as generalProfiler
from mimircache.profiler.heatmap import heatmap as heatmap
from mimircache.top.cachecow import cachecow as cachecow
from mimircache.const import *

from mimircache.const import init

init()


# from _version import __version__

# logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)

# print(__version__)
