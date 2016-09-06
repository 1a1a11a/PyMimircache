


import unittest
import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler
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
        p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=8)
        p2 = generalProfiler(reader, 'FIFO', cache_size=2000, num_of_threads=8)

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

        reader = plainReader("{}/trace.txt".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=8)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16934804618358612)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 449)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83065193891525269)

        reader = csvReader("{}/trace.csv".format(DAT_FOLDER), init_params={"header":True, 'label_column':4, 'delimiter':','})
        p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=8)

        hc = p.get_hit_count()
        self.assertEqual(hc[10], 449)
        self.assertEqual(hc[0], 0)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16934804618358612)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83065193891525269)

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
        p = cGeneralProfiler(reader, "Optimal", cache_size=2000, num_of_threads=8)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.28106996417045593)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 180)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.71893000602722168)


        reader = csvReader("{}/trace.csv".format(DAT_FOLDER), init_params={"header":True, 'label_column':4, 'delimiter':','})
        p = cGeneralProfiler(reader, "Optimal", cache_size=2000, num_of_threads=8)

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


        reader = plainReader("{}/trace.txt".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "LRU_2", cache_size=2000, num_of_threads=8)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 164)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83455109596252441)


        reader = csvReader("{}/trace.csv".format(DAT_FOLDER), init_params={"header":True, 'label_column':4, 'delimiter':','})
        p = cGeneralProfiler(reader, "LRU_2", cache_size=2000, num_of_threads=8)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 164)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83455109596252441)

    def test_LRU_K(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "LRU_K", cache_size=2000, cache_params={"K":2})

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

        reader = plainReader("{}/trace.txt".format(DAT_FOLDER))
        p = cGeneralProfiler(reader, "LRU_K", cache_size=2000, cache_params={"K":2}, num_of_threads=8)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)
        hc = p.get_hit_count()
        self.assertEqual(hc[10], 164)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertAlmostEqual(mr[-1], 0.83455109596252441)


if __name__ == "__main__":
    unittest.main()
