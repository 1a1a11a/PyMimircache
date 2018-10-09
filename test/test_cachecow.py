# coding=utf-8

import unittest
import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from PyMimircache import Cachecow

DAT_FOLDER = "../data/"
if not os.path.exists(DAT_FOLDER):
    if os.path.exists("data/"):
        DAT_FOLDER = "data/"
    elif os.path.exists("../PyMimircache/data/"):
        DAT_FOLDER = "../PyMimircache/data/"

class cachecowTest(unittest.TestCase):
    def test_all(self):
        param_sets = [{"cache_size": 2000},
                      {"cache_size": 2000,
                        "time_mode": "v",
                        "time_interval": 2000},
                      {"cache_size": 2000,
                        "time_mode": "r",
                        "time_interval": 200 * 1000000}]

        self._test_overall()
        self._test_basic(param_sets)


    def _test_overall(self):
        c = Cachecow()
        c.csv("{}/trace.csv".format(DAT_FOLDER),
              init_params={"header" :True, 'label' :5, 'real_time':2, "delimiter": ","})
        self.assertEqual(len(c), 113872)
        stat = c.characterize("short", print_stat=False)
        for line in stat:
            if "number of requests" in line:
                self.assertEqual(int(line.split(":")[1].strip()), 113872)
            elif "number of uniq obj/blocks" in line:
                self.assertEqual(int(line.split(":")[1].strip()), 48973)
            elif "cold miss ratio" in line:
                self.assertAlmostEqual(int(line.split(":")[1].strip()), 0.4301)
            elif "number of obj/block accessed only once" in line:
                self.assertEqual(int(line.split(":")[1].strip()), 21048)
            elif "frequency mean" in line:
                self.assertAlmostEqual(int(line.split(":")[1].strip()), 2.33)
            elif "time span" in line:
                self.assertAlmostEqual(int(line.split(":")[1].strip()), 7199847246.0)


    def _test_basic(self, param_sets):
        c = Cachecow()
        c.open('{}/trace.txt'.format(DAT_FOLDER))
        for param_set in param_sets:
            if "time_mode" not in param_set:
                self._coretest(c, param_set)
        c.close()

        c = Cachecow()
        c.csv("{}/trace.csv".format(DAT_FOLDER),
              init_params={"header" :True, 'label' :5, 'real_time':2})
        for param_set in param_sets:
            self._coretest(c, param_set)
        c.close()

        c = Cachecow()
        c.vscsi('{}/trace.vscsi'.format(DAT_FOLDER))
        for param_set in param_sets:
            self._coretest(c, param_set)
        c.close()


    def _coretest(self, c, param_set):
        assert "cache_size" in param_set, "require cache_size"
        if param_set.get("time_mode", ) is None:
            self._coretest_basic(c, param_set["cache_size"])
        elif param_set.get("time_mode", ) == "r":
            assert "time_interval" in param_set, "require time_interval"
            self._coretest_time(c, param_set["cache_size"], "r", param_set["time_interval"])
            self._coretest_realtime(c, param_set["cache_size"], param_set["time_interval"])
        elif param_set.get("time_mode", ) == "v":
            assert "time_interval" in param_set, "require time_interval"
            self._coretest_time(c, param_set["cache_size"], "v", param_set["time_interval"])

    def _coretest_basic(self, c, cache_size):

        p = c.profiler("LRU")
        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[2000], 0.172851974146)

        p = c.profiler("LRU_K", cache_size=cache_size, cache_params={"K": 2},
                       num_of_threads=os.cpu_count())
        hr = p.get_hit_ratio()
        self.assertAlmostEqual(hr[0], 0.0)
        self.assertAlmostEqual(hr[100], 0.16544891893863678)

        c.twoDPlot("scan_vis")
        c.twoDPlot("popularity")
        c.twoDPlot("rd_popularity")
        c.twoDPlot("interval_hit_ratio", cache_size=cache_size)

        c.plotHRCs(["LRU", "Optimal", "LFU", "LRU_K", "SLRU", "ARC"],
                   [None, None, None, {"K":2}, None, None],
                   cache_size=cache_size, bin_size=cache_size//4+1)

        c.plotHRCs(["LRU", "LFUFast"], cache_unit_size=32*1024, figname="HRC_withSize.png")


    def _coretest_time(self, c, cache_size, time_mode, time_interval):
        c.heatmap(time_mode, "hit_ratio_start_time_end_time",
                  time_interval=time_interval, cache_size=cache_size,
                  num_of_threads=os.cpu_count())

        c.heatmap(time_mode, "hit_ratio_start_time_end_time",
                  num_of_pixels=8, cache_size=cache_size,
                  num_of_threads=os.cpu_count())

        c.heatmap(time_mode, "rd_distribution",
                  time_interval=time_interval, num_of_threads=os.cpu_count())

        c.diff_heatmap(time_mode, "hit_ratio_start_time_end_time",
                       time_interval=time_interval,
                       cache_size=cache_size,
                       algorithm1="LRU", algorithm2="MRU",
                       cache_params2=None, num_of_threads=os.cpu_count())

        c.twoDPlot("cold_miss_count", time_mode=time_mode, time_interval=time_interval)
        c.twoDPlot("cold_miss_ratio", time_mode=time_mode, time_interval=time_interval)


    def _coretest_realtime(self, c, cache_size, time_interval):
        c.twoDPlot("request_rate", time_mode='r', time_interval=time_interval)
        c.twoDPlot("rt_popularity", granularity=10 * 1000000)
        # c.evictionPlot('r', time_interval, "accumulative_freq", "Optimal", cache_size)
        # c.evictionPlot('r', time_interval, "reuse_dist", "Optimal", cache_size)



if __name__ == "__main__":
    unittest.main()
