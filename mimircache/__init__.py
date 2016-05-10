import logging

from mimircache.oldModule.basicLRUProfiler import basicLRUProfiler as basicLRUProfiler
from .cache.LRU import LRU as LRU
from .cacheReader.csvReader import csvCacheReader as csvReader
from .cacheReader.plainReader import plainCacheReader as plainReader
from .cacheReader.vscsiReader import vscsiCacheReader as vscsiReader
from .profiler.pardaProfiler import pardaProfiler as pardaProfiler
from .profiler.pardaProfiler import parda_mode as parda_mode
from .top.cachecow import cachecow as cacheCow

logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)
