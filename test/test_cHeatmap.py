


import unittest
import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.cHeatmap import cHeatmap


DAT_FOLDER = "../data/"
import os
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../mimircache/data/"):
        DAT_FOLDER = "../mimircache/data/"


class cHeatmapTest(unittest.TestCase):
    def test1(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        cH = cHeatmap()
        bpr = cH.gen_breakpoints(reader, 'r', 1000000)
        self.assertEqual(bpr[10], 53)
        bpv = cH.gen_breakpoints(reader, 'v', 1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
        cH.heatmap(reader, 'r', 10000000, "rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'r', 10000000, "future_rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8, cache_size=2000)
        cH.differential_heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", cache_size=2000,
                                algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
                                cache_params2=None, num_of_threads=8)


    def test2(self):
        reader = plainReader("{}/trace.txt".format(DAT_FOLDER))
        cH = cHeatmap()
        bpv = cH.gen_breakpoints(reader, 'v', 1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
        cH.heatmap(reader, 'v', 1000, "rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'v', 1000, "future_rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8,
                   cache_size=2000)
        cH.differential_heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", cache_size=2000,
                                algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
                                cache_params2=None, num_of_threads=8)


    def test3(self):
        reader = csvReader("{}/trace.csv".format(DAT_FOLDER), init_params={"header":True, "label_column":4})
        cH = cHeatmap()
        bpv = cH.gen_breakpoints(reader, 'v', 1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
        cH.heatmap(reader, 'v', 1000, "rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'v', 1000, "future_rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8,
                   cache_size=2000)
        cH.differential_heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", cache_size=2000,
                                algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
                                cache_params2=None, num_of_threads=8)


    def test4(self):
        reader = csvReader("{}/trace.csv".format(DAT_FOLDER), init_params={"header":True, "label_column":4, 'real_time_column':1})
        cH = cHeatmap()
        bpr = cH.gen_breakpoints(reader, 'r', 1000000)
        self.assertEqual(bpr[10], 53)
        bpv = cH.gen_breakpoints(reader, 'v', 1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
        cH.heatmap(reader, 'r', 10000000, "rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'r', 10000000, "future_rd_distribution", num_of_threads=8)
        cH.heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8,
                   cache_size=2000)
        cH.differential_heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time", cache_size=2000,
                                algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
                                cache_params2=None, num_of_threads=8)


if __name__ == "__main__":
    unittest.main()
