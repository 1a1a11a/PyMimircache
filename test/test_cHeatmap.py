# coding=utf-8


import unittest
import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.cacheReader.binaryReader import binaryReader
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
    def test1_vReader(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        cH = cHeatmap()
        bpr = cH.getBreakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)
        bpr = cH.getBreakpoints(reader, 'r', num_of_pixels=1000)
        # print(bpr)
        bpv = cH.getBreakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   time_interval=10000000, num_of_threads=4, cache_size=200)
        cH.heatmap(reader, 'r', "rd_distribution",
                   time_interval=10000000, num_of_threads=4)
        cH.heatmap(reader, 'r', "future_rd_distribution",
                   time_interval=10000000, num_of_threads=4)
        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   time_interval=10000000, algorithm="FIFO",
                   num_of_threads=4, cache_size=2000)
        cH.diffHeatmap(reader, 'r', "hit_rate_start_time_end_time",
                       cache_size=200, time_interval=100000000,
                       algorithm1="LRU", algorithm2="Optimal",
                       cache_params2=None, num_of_threads=4)


    def test2_pReader(self):
        reader = plainReader("{}/trace.txt".format(DAT_FOLDER))
        cH = cHeatmap()
        bpv = cH.getBreakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'v', "hit_rate_start_time_end_time",
                   time_interval=1000, num_of_threads=4, cache_size=2000)
        cH.heatmap(reader, 'v', "rd_distribution",
                   time_interval=1000, num_of_threads=4)
        cH.heatmap(reader, 'v', "future_rd_distribution",
                   time_interval=1000, num_of_threads=4)
        cH.heatmap(reader, 'v', "hit_rate_start_time_end_time",
                   time_interval=10000, algorithm="FIFO",
                   num_of_threads=4, cache_size=2000)

        cH.diffHeatmap(reader, 'v', "hit_rate_start_time_end_time",
                       time_interval=10000, cache_size=200,
                       algorithm1="LFU", algorithm2="Optimal",
                       cache_params2=None, num_of_threads=4)


    def test3_cReader_v(self):
        reader = csvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label_column":5})
        cH = cHeatmap()
        bpv = cH.getBreakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'v', "hit_rate_start_time_end_time",
                   num_of_pixels=24, num_of_threads=4, cache_size=2000)
        cH.heatmap(reader, 'v', "rd_distribution",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'v', "rd_distribution_CDF",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'v', "future_rd_distribution",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'v', "hit_rate_start_time_end_time",
                   num_of_pixels=24, algorithm="FIFO",
                   num_of_threads=4, cache_size=2000)

        cH.diffHeatmap(reader, 'v', "hit_rate_start_time_end_time",
                       time_interval=10000, cache_size=200,
                       algorithm1="SLRU", algorithm2="Optimal",
                       cache_params2=None, num_of_threads=4)


    def test4_cReader_r(self):
        reader = csvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label_column":5, 'real_time_column':2})
        cH = cHeatmap()
        bpr = cH.getBreakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)

        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   num_of_pixels=24, num_of_threads=4, cache_size=2000)
        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   num_of_threads=4, cache_size=2000)
        cH.heatmap(reader, 'r', "rd_distribution",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'r', "future_rd_distribution",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   num_of_pixels=100, algorithm="FIFO",
                   num_of_threads=4, cache_size=2000)

        cH.diffHeatmap(reader, 'r', "hit_rate_start_time_end_time",
                       time_interval=100000000, cache_size=200,
                       algorithm1="LFUFast", algorithm2="Optimal",
                       cache_params2=None, num_of_threads=4)



    def test5_bReader(self):
        reader = binaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})

        cH = cHeatmap()
        bpr = cH.getBreakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)
        bpv = cH.getBreakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   num_of_pixels=100, num_of_threads=4, cache_size=2000)
        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   num_of_threads=4, cache_size=2000)
        cH.heatmap(reader, 'r', "rd_distribution",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'r', "future_rd_distribution",
                   num_of_pixels=1000, num_of_threads=4)
        cH.heatmap(reader, 'r', "hit_rate_start_time_end_time",
                   num_of_pixels=100, algorithm="FIFO",
                   num_of_threads=4, cache_size=200)

        cH.diffHeatmap(reader, 'r', "hit_rate_start_time_end_time",
                       num_of_pixels=24, cache_size=200,
                       algorithm1="LRU", algorithm2="Optimal",
                       cache_params2=None, num_of_threads=4)

if __name__ == "__main__":
    unittest.main()
