# coding=utf-8
"""
    unittest for cLRUProfiler module

    Author: Juncheng Yang <peter.waynechina@gmail.com>

"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))

import unittest

from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader

DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class CLRUProfilerTest(unittest.TestCase):
    def test_reader_v(self):
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        p = CLRUProfiler(reader)

        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[2000], 0.172851974146)
        hc = p.get_hit_count()
        self.assertEqual(hc[20002], 0)
        self.assertEqual(hc[0], 0)


        rd = p.get_reuse_distance()
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        frd = p.get_future_reuse_distance()
        self.assertEqual(frd[20], 10)
        self.assertEqual(frd[21], 56)

        # begin end deprecated
        # hr = p.get_hit_ratio(begin=113852, end=113872)
        # self.assertEqual(hr[8], 0.2)
        # hr = p.get_hit_ratio(cache_size=5, begin=113852, end=113872)
        # self.assertAlmostEqual(hr[2], 0.05)

        hr = p.get_hit_ratio(cache_size=20)
        self.assertAlmostEqual(hr[1], 0.02357911)

        reader.close()


    def test_reader_p(self):
        reader = PlainReader("{}/trace.txt".format(DAT_FOLDER), data_type='c')
        p = CLRUProfiler(reader)
        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[2000], 0.172851974146)

        hc = p.get_hit_count()
        self.assertEqual(hc[20002], 0)
        self.assertEqual(hc[0], 0)

        rd = p.get_reuse_distance()
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        frd = p.get_future_reuse_distance()
        self.assertEqual(frd[20], 10)
        self.assertEqual(frd[21], 56)

        hr = p.get_hit_ratio(cache_size=20)
        self.assertAlmostEqual(hr[1], 0.02357911)

        reader.close()


    def test_reader_c(self):
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label":5})
        p = CLRUProfiler(reader)

        rd = p.get_reuse_distance()
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[2000], 0.172851974146)
        hc = p.get_hit_count()
        self.assertEqual(hc[20002], 0)
        self.assertEqual(hc[0], 0)

        rd = p.get_reuse_distance()
        self.assertEqual(rd[1024], -1)
        self.assertEqual(rd[113860], 1)

        frd = p.get_future_reuse_distance()
        self.assertEqual(frd[20], 10)
        self.assertEqual(frd[21], 56)

        hr = p.get_hit_ratio(cache_size=20)
        self.assertAlmostEqual(hr[1], 0.02357911)

        p.plotHRC("test.png", cache_unit_size=32*1024)

        reader.close()


if __name__ == "__main__":
    unittest.main()
