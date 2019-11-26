# coding=utf-8


import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))
import unittest
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
from PyMimircache.profiler.cHeatmap import CHeatmap


DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class CHeatmapTest(unittest.TestCase):
    def test1_vReader(self):
        print("test1 vReader")
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        cH = CHeatmap()
        bpr = cH.get_breakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)
        bpr = cH.get_breakpoints(reader, 'r', num_of_pixel_of_time_dim=1000)
        bpv = cH.get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'r', "hr_st_et",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   cache_size=200, figname="vReader_hr_st_et_LRU.png")
        cH.heatmap(reader, 'r', "hr_st_size",
                   enable_ihr=True,
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   cache_size=200, figname="vReader_ihr_st_size.png")

        cH.heatmap(reader, 'r', "rd_distribution",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   figname="vReader_rd_dist.png")
        cH.heatmap(reader, 'r', "future_rd_distribution",
                   time_interval=10000000, num_of_threads=os.cpu_count(),
                   figname="vReader_frd_dist.png")
        cH.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   time_interval=10000000, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="vReader_hr_st_et_FIFO.png")
        cH.diff_heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                        cache_size=200, time_interval=100000000,
                        algorithm1="LRU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count(),
                        figname="vReader_diff_hr_st_et.png")


    def test2_pReader(self):
        print("test2 pReader r")
        reader = PlainReader("{}/trace.txt".format(DAT_FOLDER))
        cH = CHeatmap()
        bpv = cH.get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                   time_interval=1000, num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="pReader_hr_st_et_LRU.png")
        cH.heatmap(reader, 'v', "rd_distribution",
                   time_interval=1000, num_of_threads=os.cpu_count(),
                   figname="pReader_rd_dist.png")
        cH.heatmap(reader, 'v', "future_rd_distribution",
                   time_interval=1000, num_of_threads=os.cpu_count(),
                   figname="pReader_frd_dist.png")
        cH.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                   time_interval=10000, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="pReader_hr_st_et_FIFO.png")

        cH.diff_heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                        time_interval=10000, cache_size=200,
                        algorithm1="LFU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count(),
                        figname="pReader_diff_hr_st_et.png")


    def test3_c_reader_v(self):
        print("test3 c_reader v")
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label":5})
        cH = CHeatmap()
        bpv = cH.get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        # cH.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
        #            num_of_pixel_of_time_dim=24,
        #            num_of_threads=os.cpu_count(), cache_size=2000,
        #            figname="hr_st_et_LRU_c_reader_v.png")

        cH.heatmap(reader, 'v', "rd_distribution",
                   num_of_pixel_of_time_dim=200,
                   num_of_threads=os.cpu_count(),
                   figname="c_reader_rd_dist.png")

        cH.heatmap(reader, 'v', "rd_distribution_CDF",
                   num_of_pixel_of_time_dim=1000,
                   num_of_threads=os.cpu_count(),
                   figname="c_reader_rd_CDF_dist.png")

        cH.heatmap(reader, 'v', "future_rd_distribution",
                   num_of_pixel_of_time_dim=1000,
                   num_of_threads=os.cpu_count(),
                   figname="c_reader_frd_dist.png")

        cH.heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=24, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="c_reader_hr_st_et_FIFO.png")

        cH.diff_heatmap(reader, 'v', "hit_ratio_start_time_end_time",
                        time_interval=10000, cache_size=200,
                        algorithm1="SLRU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count())


    def test4_c_reader_r(self):
        print("test4 c_reader r")
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label":5, 'real_time':2})
        cH = CHeatmap()
        bpr = cH.get_breakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)

        cH.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=24, num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="hr_st_et_LRU_c_reader_r.png")

        cH.heatmap(reader, 'r', "rd_distribution",
                   num_of_pixel_of_time_dim=1000, num_of_threads=os.cpu_count())
        cH.heatmap(reader, 'r', "future_rd_distribution",
                   num_of_pixel_of_time_dim=1000, num_of_threads=os.cpu_count())
        cH.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=100, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=2000)

        cH.diff_heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                        time_interval=100000000, cache_size=200,
                        algorithm1="LFUFast", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count())



    def test5_bReader(self):
        print("bReader")
        reader = BinaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})

        cH = CHeatmap()
        bpr = cH.get_breakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)
        bpv = cH.get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        cH.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=100, num_of_threads=os.cpu_count(), cache_size=2000,
                   figname="hr_st_et_LRU_bReader.png")

        cH.heatmap(reader, 'r', "rd_distribution",
                   num_of_pixel_of_time_dim=1000, num_of_threads=os.cpu_count())
        cH.heatmap(reader, 'r', "future_rd_distribution",
                   num_of_pixel_of_time_dim=1000, num_of_threads=os.cpu_count())
        cH.heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                   num_of_pixel_of_time_dim=100, algorithm="FIFO",
                   num_of_threads=os.cpu_count(), cache_size=200)

        cH.diff_heatmap(reader, 'r', "hit_ratio_start_time_end_time",
                        num_of_pixel_of_time_dim=24, cache_size=200,
                        algorithm1="LRU", algorithm2="Optimal",
                        cache_params2=None, num_of_threads=os.cpu_count())

if __name__ == "__main__":
    unittest.main()
