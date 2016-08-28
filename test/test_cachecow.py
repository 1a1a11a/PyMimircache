

import unittest
import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.generalProfiler import generalProfiler
from mimircache import cachecow

DAT_FOLDER = "../data/"
import os 
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../mimircache/data/"):
        DAT_FOLDER = "../mimircache/data/"

class cachecowTest(unittest.TestCase):
    def test1(self):
        CACHE_SIZE = 2000
        TIME_MODE = 'r'
        TIME_INTERVAL = 50000000
        c = cachecow()
        # c.open('../data/trace.txt')
        c.csv("{}/trace.csv".format(DAT_FOLDER), init_params={"header" :True, 'label_column' :4, 'real_time_column':1})
        # c.vscsi('{}/trace.vscsi'.format(DAT_FOLDER)

        p = c.profiler("LRU")
        p = LRUProfiler(c.reader)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[2000], 0.172851974146)

        # p = c.profiler("LRU_K", cache_size=CACHE_SIZE, cache_params={"K": 2}, num_of_threads=8)
        # hr = p.get_hit_rate()
        # self.assertAlmostEqual(hr[0], 0.0)
        # self.assertAlmostEqual(hr[100], 0.16544891893863678)
        #
        # c.heatmap('v', 1000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
        # c.heatmap('v', 1000, "rd_distribution", num_of_threads=8)
        #
        # c.differential_heatmap(TIME_MODE, TIME_INTERVAL, "hit_rate_start_time_end_time", cache_size=CACHE_SIZE,
        #                        algorithm1="LRU", algorithm2="MRU", cache_params2=None, num_of_threads=8)
        #
        # c.twoDPlot('v', 1000, "cold_miss")
        # c.evictionPlot('r', 10000000, "accumulative_freq", "Optimal", 1000)
        # c.evictionPlot('r', 10000000, "reuse_dist", "Optimal", 10000)

        c.plotHRCs(["LRU", "Optimal", "LFU", "LRU_K"], [None, None, None, {"K":2}])



if __name__ == "__main__":
    unittest.main()
