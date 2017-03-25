# coding=utf-8


import unittest
import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler

DAT_FOLDER = "../data/"
import os
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../mimircache/data/"):
        DAT_FOLDER = "../mimircache/data/"


class LRUProfilerTest(unittest.TestCase):
    def test_reader_v(self):
        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = LRUProfiler(reader)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[2000], 0.172851974146)
        hc = p.get_hit_count()
        self.assertEqual(hc[20002], 0)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertEqual(hr[-1], mr[-1])
        #
        rd = p.get_reuse_distance()
        print(rd)
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        c_LRUProfiler.get_future_reuse_dist(reader.cReader)

        hr = p.get_hit_rate(begin=113852, end=113872)
        print(hr)
        self.assertEqual(hr[8], 0.2)

        hr = p.get_hit_rate(cache_size=20)
        self.assertAlmostEqual(hr[1], 0.02357911)
        hr = p.get_hit_rate(cache_size=5, begin=113852, end=113872)
        self.assertAlmostEqual(hr[2], 0.05)
        reader.close()


    def test_reader_p(self):
        reader = plainReader("{}/trace.txt".format(DAT_FOLDER), data_type='c')
        p = LRUProfiler(reader)
        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[2000], 0.172851974146)

        hc = p.get_hit_count()
        self.assertEqual(hc[20002], 0)
        self.assertEqual(hc[0], 0)

        mr = p.get_miss_rate()
        self.assertEqual(hr[-1], mr[-1])
        rd = p.get_reuse_distance()
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        c_LRUProfiler.get_future_reuse_dist(reader.cReader)

        hr = p.get_hit_rate(begin=113852, end=113872)
        self.assertEqual(hr[8], 0.2)

        hr = p.get_hit_rate(cache_size=20)
        self.assertAlmostEqual(hr[1], 0.02357911)
        hr = p.get_hit_rate(cache_size=5, begin=113852, end=113872)
        self.assertAlmostEqual(hr[2], 0.05)
        reader.close()


    def test_reader_c(self):
        reader = csvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label_column":5})
        p = LRUProfiler(reader)

        rd = p.get_reuse_distance()
        print(rd)

        hr = p.get_hit_rate()
        self.assertAlmostEqual(hr[2000], 0.172851974146)
        hc = p.get_hit_count()
        self.assertEqual(hc[20002], 0)
        self.assertEqual(hc[0], 0)
        mr = p.get_miss_rate()
        self.assertEqual(hr[-1], mr[-1])

        rd = p.get_reuse_distance()
        print(rd)
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        c_LRUProfiler.get_future_reuse_dist(reader.cReader)

        hr = p.get_hit_rate(begin=113852, end=113872)
        self.assertEqual(hr[8], 0.2)

        hr = p.get_hit_rate(cache_size=20)
        self.assertAlmostEqual(hr[1], 0.02357911)
        hr = p.get_hit_rate(cache_size=5, begin=113852, end=113872)
        self.assertAlmostEqual(hr[2], 0.05)
        reader.close()


if __name__ == "__main__":
    unittest.main()
