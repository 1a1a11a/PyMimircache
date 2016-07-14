

from mimircache import *


reader = csvReader('../mimircache/data/trace.csv', init_params={"header" :True, 'label_column' :4, 'delimiter' :','})
# reader = plainReader('../mimircache/data/trace.txt')

# p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=1)
# hr = p.get_hit_count()
# print(hr)


# hr = p.get_hit_rate()
# hc = p.get_hit_count()
# mr = p.get_miss_rate()

print(c_heatmap.get_next_access_dist(reader.cReader))




# cH = cHeatmap()
# bpv = cH.gen_breakpoints(reader, 'v', 1000)
#
# cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
# cH.heatmap(reader, 'v', 1000, "rd_distribution", num_of_threads=8)
# cH.heatmap(reader, 'v', 1000, "future_rd_distribution", num_of_threads=8)
# cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8,
#            cache_size=2000)
# cH.differential_heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", cache_size=2000,
#                         algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
#                         cache_params2=None, num_of_threads=8)