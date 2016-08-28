#
#
# from mimircache import *
# import mimircache.c_heatmap as c_heatmap
#
# TIME_MODE = 'r'
# TIME_INTERVAL = 2
# CACHE_SIZE = 12000
#
#
# # reader = csvReader('../mimircache/data/wiki2.csv', init_params={"header": False, 'label_column': 2,
#                                                                 # 'real_time_column': 1, 'delimiter': ','})
# # reader = plainReader('../mimircache/data/trace.txt')
# c = cachecow()
# c.csv('../mimircache/data/wiki.csv', init_params={"header": False, 'label_column': 2,
#                                                                 'real_time_column': 1, 'delimiter': ','})
# # c.plotHRCs(["LRU", "Optimal", "LFU"])
#
# # print(cHeatmap().gen_breakpoints(c.reader, 'r', 2))
# #
# # import sys
# # sys.exit(1)
#
#
# c.heatmap(TIME_MODE, TIME_INTERVAL, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=CACHE_SIZE)
# c.heatmap(TIME_MODE, TIME_INTERVAL, "rd_distribution", num_of_threads=8)
#
# c.differential_heatmap(TIME_MODE, TIME_INTERVAL, "hit_rate_start_time_end_time", cache_size=CACHE_SIZE,
#                        algorithm1="LRU", algorithm2="Optimal", cache_params2=None, num_of_threads=8)
#
# c.twoDPlot(TIME_MODE, TIME_INTERVAL, "cold_miss")
# c.twoDPlot(TIME_MODE, TIME_INTERVAL, "request_num")
#
# # p = cGeneralProfiler(reader, "FIFO", cache_size=2000, num_of_threads=1)
# # hr = p.get_hit_count()
# # print(hr)
#
#
# # hr = p.get_hit_rate()
# # hc = p.get_hit_count()
# # mr = p.get_miss_rate()
#
# # print(c_heatmap.get_next_access_dist(reader.cReader))
#
#
#
#
# # cH = cHeatmap()
# # bpv = cH.gen_breakpoints(reader, 'v', 1000)
# #
# # cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", num_of_threads=8, cache_size=2000)
# # cH.heatmap(reader, 'v', 1000, "rd_distribution", num_of_threads=8)
# # cH.heatmap(reader, 'v', 1000, "future_rd_distribution", num_of_threads=8)
# # cH.heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", algorithm="FIFO", num_of_threads=8,
# #            cache_size=2000)
# # cH.differential_heatmap(reader, 'v', 1000, "hit_rate_start_time_end_time", cache_size=2000,
# #                         algorithm1="LRU_K", algorithm2="Optimal", cache_params1={"K": 2},
# #                         cache_params2=None, num_of_threads=8)