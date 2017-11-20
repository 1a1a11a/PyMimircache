# coding=utf-8

""" mimircache a cache trace analysis platform.

.. moduleauthor:: Juncheng Yang <peter.waynechina@gmail.com>, Ymir Vigfusson

"""

import os, sys
try:
    import matplotlib
    matplotlib.use('Agg')
except:
    print("WARNING: fail to import matplotlib, plotting function may be limited", file=sys.stderr)


CWD = os.getcwd()
sys.path.extend([CWD, os.path.join(CWD, "..")])


from mimircache.const import *
from mimircache.profiler.LRUProfiler import LRUProfiler as LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.generalProfiler import generalProfiler as generalProfiler
from mimircache.profiler.cHeatmap import cHeatmap
from mimircache.profiler.heatmap import heatmap as heatmap
from mimircache.top.cachecow import cachecow as cachecow

from mimircache.version import __version__ as __version__



def init():
    init_cache_alg_mapping()




init()


# import logging
# logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)

