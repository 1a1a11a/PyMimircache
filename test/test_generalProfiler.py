

import unittest
from mimircache import *


DAT_FOLDER = "../data/"
import os
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../mimircache/data/"):
        DAT_FOLDER = "../mimircache/data/"

class generalProfilerTest(unittest.TestCase):
    def test1(self):
        CACHE_SIZE = 2000
        TIME_MODE = 'r'
        TIME_INTERVAL = 50000000

        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        # reader = plainReader("../mimircache/data/random.dat")
        reader = plainReader("../mimircache/data/trace.txt")
        # print(reader.get_num_of_total_requests())
        p = generalProfiler(reader, "Optimal", cache_size=1, bin_size=1,
                            cache_params={"reader": reader}, num_of_threads=1)
        # hr = p.get_hit_rate()
        hc = p.get_hit_count()
        p.plotHRC("../mimircache/1a1a11a/3_0.png")
        print(hc)
        # self.assertEqual(hr[0], 0)
        # self.assertAlmostEqual(hr[8], 0.18256463397498945)

        for i in range(1):
            cg = cGeneralProfiler(reader, "Optimal", cache_size=1, bin_size=1,
                                  cache_params={"reader": reader}, num_of_threads=1)
            hr2 = cg.get_hit_rate()
            hc2 = cg.get_hit_count()
            # cg.plotHRC("../mimircache/1a1a11a/3_1.png")
            print(hc2)
        self.assertAlmostEqual(hr2[8], 0.18256463397498945)



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

        # c.plotHRCs(["LRU", "Optimal", "LFU", "LRU_K"], [None, None, None, {"K":2}])



if __name__ == "__main__":
    unittest.main()

