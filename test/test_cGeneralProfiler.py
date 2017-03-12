# coding=utf-8


import unittest
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.cacheReader.binaryReader import binaryReader
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.generalProfiler import generalProfiler


DAT_FOLDER = "../data/"
import os
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../mimircache/data/"):
        DAT_FOLDER = "../mimircache/data/"


class cGeneralProfilerTest(unittest.TestCase):
    def test_FIFO(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=4)
        p2 = generalProfiler(reader, 'FIFO', cache_size=2000, num_of_threads=4)

        hr = p.get_hit_rate()
        hr2 = p2.get_hit_rate()

        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], hr2[100])
        self.assertAlmostEqual(hr[100], 0.16934804618358612)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 449)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83065193891525269)
        hr = p.get_hit_rate(begin=113852, end=113872, cache_size=5000)
        self.assertAlmostEqual(hr[1], 0.2)


    def test_Optimal(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "Optimal", cache_size=2000)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.28106996417045593)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 180)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.71893000602722168)
        hr = p.get_hit_rate(begin=113852, end=113872, cache_size=5000)
        self.assertAlmostEqual(hr[1], 0.2)


        reader = plainReader("{}/trace.txt".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "Optimal", cache_size=2000, num_of_threads=4)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.28106996417045593)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 180)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.71893000602722168)


        reader = csvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, 'label_column':5, 'delimiter':','})
        p = cGeneralProfiler(reader, "Optimal", cache_size=2000, num_of_threads=4)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.28106996417045593)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 180)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.71893000602722168)


        reader = binaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})
        p = cGeneralProfiler(reader, "Optimal", cache_size=2000, num_of_threads=4)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.28106996417045593)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 180)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.71893000602722168)


    def test_LRU_2(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "LRU_2", cache_size=2000)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 164)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83455109596252441)

        hr = p.get_hit_rate(begin=113852, end=113872, cache_size=5000)
        self.assertAlmostEqual(hr[1], 0.2)



    def test_SLRU(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "SLRU", cache_size=2000, cache_params={"N":2})

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.1767423003911972)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 117)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.8232576847076416)

        hr = p.get_hit_rate(begin=113852, end=113872, cache_size=5000)
        self.assertAlmostEqual(hr[1], 0.2)


    def test_LFU_LFUFast(self):
        reader = csvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, 'label_column':5, 'delimiter':','})
        p  = cGeneralProfiler(reader, "LFU", cache_size=2000, num_of_threads=4)
        p2 = cGeneralProfiler(reader, "LFUFast", cache_size=2000, num_of_threads=4)

        hr  = p.get_hit_rate()
        hr2 = p2.get_hit_rate()
        self.assertCountEqual(hr, hr2)

        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.17430096864700317)

        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.82569903135299683)



if __name__ == "__main__":
    unittest.main()
