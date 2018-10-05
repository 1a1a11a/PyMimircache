# coding=utf-8


import os
import unittest
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.profiler.pyHeatmap import PyHeatmap


DAT_FOLDER = "../data/"
import os
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class PyHeatmapTest(unittest.TestCase):
    def test1_vReader(self):
        print("test1 vReader")
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        pyh = PyHeatmap()

        pyh.heatmap(reader, 'r', "hr_st_et",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   cache_size=200, figname="vReader_hr_st_et_LRU.png")
        pyh.heatmap(reader, 'r', "hr_interval_size",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   cache_size=200, figname="vReader_hr_interval_size.png")

        pyh.heatmap(reader, 'r', "rd_distribution",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   figname="vReader_rd_dist.png")
        pyh.heatmap(reader, 'r', "future_rd_distribution",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   figname="vReader_frd_dist.png")
        pyh.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   time_interval=10000000, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="vReader_hr_st_et_FIFO.png")
        pyh.diff_heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                        cache_size=200, time_interval=100000000,
                        algorithm1="LRU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count(),
                        figname="vReader_diff_hr_st_et.png")


    def test2_pReader(self):
        print("test2 pReader r")
        reader = PlainReader("{}/trace.txt".format(DAT_FOLDER))
        pyh = PyHeatmap()

        pyh.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                   time_interval=1000, num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="pReader_hr_st_et_LRU.png")
        pyh.heatmap(reader, 'v', "rd_distribution",
                   time_interval=1000, num_of_threads=os.cpu_count(),
                   figname="pReader_rd_dist.png")
        pyh.heatmap(reader, 'v', "future_rd_distribution",
                   time_interval=1000, num_of_threads=os.cpu_count(),
                   figname="pReader_frd_dist.png")
        pyh.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                   time_interval=10000, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="pReader_hr_st_et_FIFO.png")

        pyh.diff_heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                        time_interval=10000, cache_size=200,
                        algorithm1="LFU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count(),
                        figname="pReader_diff_hr_st_et.png")


    def test3_cReader_v(self):
        print("test3 cReader v")
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label":5})
        pyh = PyHeatmap()

        # pyh.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
        #            num_of_pixel_of_time_dim=24,
        #            num_of_threads=os.cpu_count(), cache_size=2000,
        #            figname="hr_st_et_LRU_cReader_v.png")

        pyh.heatmap(reader, 'v', "rd_distribution",
                   num_of_pixel_of_time_dim=200,
                   num_of_threads=os.cpu_count(),
                   figname="cReader_rd_dist.png")

        pyh.heatmap(reader, 'v', "rd_distribution_CDF",
                   num_of_pixel_of_time_dim=1000,
                   num_of_threads=os.cpu_count(),
                   figname="cReader_rd_CDF_dist.png")

        pyh.heatmap(reader, 'v', "future_rd_distribution",
                   num_of_pixel_of_time_dim=1000,
                   num_of_threads=os.cpu_count(),
                   figname="cReader_frd_dist.png")

        pyh.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=24, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="cReader_hr_st_et_FIFO.png")

        pyh.diff_heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                        time_interval=10000, cache_size=200,
                        algorithm1="SLRU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count())


    def test5_bReader(self):
        print("bReader")
        reader = BinaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})

        pyh = PyHeatmap()

        pyh.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=100, num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="hr_st_et_LRU_bReader.png")

        pyh.heatmap(reader, 'r', "rd_distribution",
                   num_of_pixel_of_time_dim=1000, num_of_threads=os.cpu_count())
        pyh.heatmap(reader, 'r', "future_rd_distribution",
                   num_of_pixel_of_time_dim=1000, num_of_threads=os.cpu_count())
        pyh.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=100, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=200)

        pyh.diff_heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                        num_of_pixel_of_time_dim=24, cache_size=200,
                        algorithm1="LRU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count())

if __name__ == "__main__":
    unittest.main()

