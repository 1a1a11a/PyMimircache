# coding=utf-8

import unittest
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
        c.csv("{}/trace.csv".format(DAT_FOLDER),
              init_params={"header" :True, 'label_column' :5, 'real_time_column':2})
        # c.vscsi('{}/trace.vscsi'.format(DAT_FOLDER))

        p = c.profiler("LRU")
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[2000], 0.172851974146)

        p = c.profiler("LRU_K", cache_size=CACHE_SIZE, cache_params={"K": 2}, num_of_threads=8)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)

        c.heatmap('v', "hit_rate_start_time_end_time",
                  time_interval=1000, num_of_threads=8, cache_size=2000)
        c.heatmap('v', "hit_rate_start_time_end_time",
                  num_of_pixels=100, num_of_threads=8, cache_size=2000)
        c.heatmap('v', "rd_distribution", time_interval=1000, num_of_threads=8)

        c.diffHeatmap(TIME_MODE, "hit_rate_start_time_end_time",
                      time_interval=TIME_INTERVAL,
                      cache_size=CACHE_SIZE,
                      algorithm1="LRU", algorithm2="MRU",
                      cache_params2=None, num_of_threads=8)

        c.twoDPlot("cold_miss_count", mode='v', time_interval=1000)
        c.twoDPlot("request_num", mode='v', time_interval=1000)
        c.twoDPlot("mapping")
        c.evictionPlot('r', 10000000, "accumulative_freq", "Optimal", 1000)
        c.evictionPlot('r', 10000000, "reuse_dist", "Optimal", 10000)

        c.plotHRCs(["LRU", "Optimal", "LFUFast", "LRU_K", "SLRU"], [None, None, None, {"K":2}])



if __name__ == "__main__":
    unittest.main()
