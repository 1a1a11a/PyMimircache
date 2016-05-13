import os
import sys

# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

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
