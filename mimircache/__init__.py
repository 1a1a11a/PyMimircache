""" mimircache a cache trace analysis platform.

.. moduleauthor:: Juncheng Yang <peter.waynechina@gmail.com>, Ymir Vigfusson

"""

import matplotlib
matplotlib.use('Agg')

# import warnings
# warnings.filterwarnings("ignore")

# import logging

from mimircache.cache.LRU import LRU as LRU
from mimircache.cacheReader.csvReader import csvReader as csvReader
from mimircache.cacheReader.plainReader import plainReader as plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader as vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler as LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.generalProfiler import generalProfiler as generalProfiler
from mimircache.profiler.cHeatmap import cHeatmap
from mimircache.profiler.heatmap import heatmap as heatmap
from mimircache.profiler.twoDPlots import *
from mimircache.top.cachecow import cachecow as cachecow
from mimircache.const import *


init()


# from _version import __version__

# logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)

# print(__version__)
