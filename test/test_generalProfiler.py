# coding=utf-8
"""
this module provides unittest for cGeneralProfiler and pyGeneralProfiler,
it uses single cache replacement algorithm (FIFO) with different types of readers
the test of other algorithms are excluded and should be under cache folder

"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))
import unittest

from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader

from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler



DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class GeneralProfilerTest(unittest.TestCase):
    def test_FIFO_vscsi(self):
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p1 = CGeneralProfiler(reader, "FIFO", cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())
        p2 = PyGeneralProfiler(reader, 'FIFO', cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())

        hc1 = p1.get_hit_count()
        hc2 = p2.get_hit_count()
        self.assertEqual(hc1[0], 0)
        self.assertEqual(hc1[8], 187)
        self.assertListEqual(list(hc1), list(hc2))

        hr1 = p1.get_hit_ratio()
        hr2 = p2.get_hit_ratio()
        self.assertAlmostEqual(hr1[0], 0.0)
        self.assertAlmostEqual(hr2[0], 0.0)
        self.assertAlmostEqual(hr1[2], hr2[2])
        self.assertAlmostEqual(hr1[2], 0.148702055216)

        # get hit count again to make sure the value won't change
        hc = p1.get_hit_count()
        self.assertEqual(hc[0], 0)
        self.assertEqual(hc[8], 187)

        p1.plotHRC(figname="test_v_c.png", cache_unit_size=32*1024)
        p2.plotHRC(figname="test_v_py.png", cache_unit_size=32*1024)
        reader.close()


    def test_FIFO_plain(self):
        reader = PlainReader("{}/trace.txt".format(DAT_FOLDER))
        p1 = CGeneralProfiler(reader, "FIFO", cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())
        p2 = PyGeneralProfiler(reader, 'FIFO', cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())

        hc1 = p1.get_hit_count()
        hc2 = p2.get_hit_count()
        self.assertEqual(hc1[0], 0)
        self.assertEqual(hc1[8], 187)
        self.assertListEqual(list(hc1), list(hc2))

        hr1 = p1.get_hit_ratio()
        hr2 = p2.get_hit_ratio()
        self.assertAlmostEqual(hr1[0], 0.0)
        self.assertAlmostEqual(hr2[0], 0.0)
        self.assertAlmostEqual(hr1[2], hr2[2])
        self.assertAlmostEqual(hr1[2], 0.148702055216)

        # get hit count again to make sure the value won't change
        hc = p1.get_hit_count()
        self.assertEqual(hc[0], 0)
        self.assertEqual(hc[8], 187)

        p1.plotHRC(figname="test_p_c.png", cache_unit_size=32*1024)
        p2.plotHRC(figname="test_p_py.png", cache_unit_size=32*1024)
        reader.close()


    def test_FIFO_csv(self):
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, 'label':5, 'delimiter':','})
        p1 = CGeneralProfiler(reader, "FIFO", cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())
        p2 = PyGeneralProfiler(reader, 'FIFO', cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())

        hc1 = p1.get_hit_count()
        hc2 = p2.get_hit_count()
        self.assertEqual(hc1[0], 0)
        self.assertEqual(hc1[8], 187)
        self.assertListEqual(list(hc1), list(hc2))

        hr1 = p1.get_hit_ratio()
        hr2 = p2.get_hit_ratio()
        self.assertAlmostEqual(hr1[0], 0.0)
        self.assertAlmostEqual(hr2[0], 0.0)
        self.assertAlmostEqual(hr1[2], hr2[2])
        self.assertAlmostEqual(hr1[2], 0.148702055216)

        # get hit count again to make sure the value won't change
        hc = p1.get_hit_count()
        self.assertEqual(hc[0], 0)
        self.assertEqual(hc[8], 187)

        p1.plotHRC(figname="test_c_c.png", cache_unit_size=32*1024)
        p2.plotHRC(figname="test_c_py.png", cache_unit_size=32*1024)
        reader.close()


    def test_FIFO_bin(self):
        reader = BinaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})

        p1 = CGeneralProfiler(reader, "FIFO", cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())
        p2 = PyGeneralProfiler(reader, 'FIFO', cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())

        hc1 = p1.get_hit_count()
        hc2 = p2.get_hit_count()

        self.assertEqual(hc1[0], 0)
        self.assertEqual(hc1[8], 187)
        self.assertListEqual(list(hc1), list(hc2))

        hr1 = p1.get_hit_ratio()
        hr2 = p2.get_hit_ratio()
        self.assertAlmostEqual(hr1[0], 0.0)
        self.assertAlmostEqual(hr2[0], 0.0)
        self.assertAlmostEqual(hr1[2], hr2[2])
        self.assertAlmostEqual(hr1[2], 0.148702055216)

        # get hit count again to make sure the value won't change
        hc = p1.get_hit_count()
        self.assertEqual(hc[0], 0)
        self.assertEqual(hc[8], 187)

        p1.plotHRC(figname="test_b_c.png", cache_unit_size=32*1024)
        p2.plotHRC(figname="test_b_py.png", cache_unit_size=32*1024)
        reader.close()

    def test_cGeneralProfiler(self):
        reader = BinaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})

        p1 = CGeneralProfiler(reader, "FIFO", cache_size=2000, bin_size=200, num_of_threads=os.cpu_count())
        ea = p1.get_eviction_age()


if __name__ == "__main__":
    unittest.main()
