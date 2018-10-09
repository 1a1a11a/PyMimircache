# coding=utf-8


import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))
import unittest
from PyMimircache.cacheReader.csvReader import CsvReader
from PyMimircache.cacheReader.plainReader import PlainReader
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.profiler.profilerUtils import *
from PyMimircache.profiler.utils.dist import *
import PyMimircache.CMimircache.Heatmap as c_heatmap

DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"


class ProfilerUtilsTest(unittest.TestCase):
    def test1_vReader(self):
        reader = VscsiReader("{}/trace.vscsi".format(DAT_FOLDER))

        bpr = get_breakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)
        bpr = get_breakpoints(reader, 'r', num_of_pixel_of_time_dim=1000)
        self.assertEqual(bpr[10], 245)
        bpv = get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)

        c_next_access = c_heatmap.get_next_access_dist(reader.c_reader)
        py_next_access = get_next_access_dist(reader)
        self.assertListEqual(list(c_next_access), list(py_next_access))


    def test2_pReader(self):
        reader = PlainReader("{}/trace.txt".format(DAT_FOLDER))
        bpv = get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)


    def test3_cReader_v(self):
        reader = CsvReader("{}/trace.csv".format(DAT_FOLDER),
                           init_params={"header":True, "label":5})
        bpv = get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)


    def test5_bReader(self):
        reader = BinaryReader("{}/trace.vscsi".format(DAT_FOLDER),
                              init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})

        bpr = get_breakpoints(reader, 'r', time_interval=1000000)
        self.assertEqual(bpr[10], 53)
        bpv = get_breakpoints(reader, 'v', time_interval=1000)
        self.assertEqual(bpv[10], 10000)


if __name__ == "__main__":
    unittest.main()

