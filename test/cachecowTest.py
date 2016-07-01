

import unittest
import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.generalProfiler import generalProfiler
from mimircache import cachecow

class cachecowTest(unittest.TestCase):
    def test1(self):
        c = cachecow()
        c.vscsi('../mimircache/data/trace.vscsi')
        c.differential_heatmap('r', 50000000, "hit_rate_start_time_end_time", cache_size=2000,
                        algorithm1="LRU", algorithm2="MRU",
                        cache_params2=None, num_of_threads=8)




if __name__ == "__main__":
    unittest.main()
