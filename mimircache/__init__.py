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
from mimircache.profiler.cLRUProfiler import CLRUProfiler as LRUProfiler
from mimircache.profiler.cGeneralProfiler import CGeneralProfiler
from mimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
from mimircache.profiler.cHeatmap import CHeatmap
from mimircache.profiler.pyHeatmap import PyHeatmap
from mimircache.top.cachecow import Cachecow

from mimircache.version import __version__



def init():
    init_cache_alg_mapping()




init()


# import logging
# logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)

