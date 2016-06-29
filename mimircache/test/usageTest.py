
from mimircache.top.cachecow import cachecow

import mimircache.c_cacheReader as c_cacheReader
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.cHeatmap import cHeatmap


def t1():
    reader = vscsiCacheReader('../data/trace.vscsi')
    p = LRUProfiler(reader)
    print(p.get_hit_rate())
    #
    hr = p.get_hit_rate(cache_size=20)
    print(hr)
    # print(hr)
    rd = p.get_reuse_distance(begin=113852, end=113872)
    # print(rd)
    #
    print(p.get_hit_rate())
    print(p.get_hit_count())
    print(p.get_miss_rate())
    print(p.get_hit_rate(begin=113852, end=113872, cache_size=5))


def t2():
    reader = vscsiCacheReader('../data/trace.vscsi')
    p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=8)

    hr = p.get_hit_rate()
    hc = p.get_hit_count()
    mr = p.get_miss_rate()
    print(mr)
    p.plotHRC()
    p.plotMRC()


def t3():
    reader = vscsiCacheReader('../data/trace.vscsi')
    cH = cHeatmap()

    # cH.heatmap(reader, 'r', 1000000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
    # cH.heatmap(reader, 'r', 1000000, "rd_distribution", num_of_threads=8)
    # cH.heatmap(reader, 'r', 1000000, "future_rd_distribution", num_of_threads=8)
    # cH.heatmap(reader, 'r', 1000000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8, cache_size=2000)
    cH.differential_heatmap(reader, 'r', 10000000, "hit_rate_start_time_end_time",
                            algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
                            cache_params2=None, num_of_threads=8, cache_size=2000)


t3()
