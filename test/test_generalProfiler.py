# coding=utf-8

import unittest
from mimircache import *


DAT_FOLDER = "../data/"
import os
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../mimircache/data/"):
        DAT_FOLDER = "../mimircache/data/"

class generalProfilerTest(unittest.TestCase):
    def test1(self):
        CACHE_SIZE = 2000
        BIN_SIZE   = 200
        TIME_MODE = 'r'
        TIME_INTERVAL = 50000000

        reader = vscsiReader("{}/trace.vscsi".format(DAT_FOLDER))
        # reader = plainReader("../mimircache/data/random.dat")
        # reader = plainReader("../mimircache/data/trace.txt")
        # print(reader.get_num_total_req())
        p = generalProfiler(reader, "Optimal", cache_size=CACHE_SIZE, bin_size=BIN_SIZE,
                            cache_params={"reader": reader}, num_of_threads=1)
        hr = p.get_hit_rate()
        self.assertEqual(hr[0], 0)
        print(hr)
        self.assertAlmostEqual(hr[8], 0.26610580300688491)

        cg = cGeneralProfiler(reader, "Optimal", cache_size=CACHE_SIZE, bin_size=BIN_SIZE,
                                  cache_params={"reader": reader}, num_of_threads=1)
        hr2 = cg.get_hit_rate()
        self.assertAlmostEqual(hr2[8], 0.26610580300688491)
        for i,j in zip(hr, hr2):
            self.assertAlmostEqual(i, j)




if __name__ == "__main__":
    unittest.main()

