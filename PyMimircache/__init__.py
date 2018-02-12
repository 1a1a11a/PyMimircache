# coding=utf-8

""" PyMimircache a cache trace analysis platform.

.. moduleauthor:: Juncheng Yang <peter.waynechina@gmail.com>, Ymir Vigfusson

"""

import os
import sys
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception as e:
    print("WARNING: {}, fail to import matplotlib, "
          "plotting function may be limited".format(e), file=sys.stderr)


cwd = os.getcwd()
sys.path.extend([cwd, os.path.join(cwd, "..")])


from PyMimircache.const import *
if not INSTALL_PHASE:
    from PyMimircache.cacheReader.binaryReader import BinaryReader
    from PyMimircache.cacheReader.vscsiReader import VscsiReader
    from PyMimircache.cacheReader.csvReader import CsvReader
    from PyMimircache.cacheReader.plainReader import PlainReader

    from PyMimircache.profiler.cLRUProfiler import CLRUProfiler as CLRUProfiler
    from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
    from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
    from PyMimircache.profiler.cHeatmap import CHeatmap
    from PyMimircache.profiler.pyHeatmap import PyHeatmap
    from PyMimircache.top.cachecow import Cachecow

from PyMimircache.version import __version__



# import logging
# logging.basicConfig(filename="log", filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', level=logging.DEBUG)

