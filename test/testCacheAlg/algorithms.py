# coding=utf-8


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import unittest

from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
from PyMimircache.profiler.generalProfiler import generalProfiler


DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class algorithmTest(unittest.TestCase):

    def test_LRU_2(self):
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = CGeneralProfiler(reader, "LRU_2", cache_size=2000)

        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 164)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83455109596252441)

        hr = p.get_hit_ratio(begin=113852, end=113872, cache_size=5000)
        self.assertAlmostEqual(hr[1], 0.2)
        reader.close()


    def test_SLRU(self):
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = CGeneralProfiler(reader, "SLRU", cache_size=2000, cache_params={"N":2})

        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.1767423003911972)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 117)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.8232576847076416)

        hr = p.get_hit_ratio(begin=113852, end=113872, cache_size=5000)
        self.assertAlmostEqual(hr[1], 0.2)
        reader.close()


    def test_LFU_LFUFast(self):
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, 'label':5, 'delimiter':','})
        p  = CGeneralProfiler(reader, "LFU", cache_size=2000, num_of_threads=4)
        p2 = CGeneralProfiler(reader, "LFUFast", cache_size=2000, num_of_threads=4)

        hr  = p.get_hit_ratio()
        hr2 = p2.get_hit_ratio()
        self.assertCountEqual(hr, hr2)

        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.17430096864700317)

        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.82569903135299683)
        p2.plotHRC("test.png", cache_unit_size=32*1024)

        reader.close()


if __name__ == "__main__":
    unittest.main()
