


from mimircache import *


FILE = "../data/wiki.csv"
INIT_PARAMS = {"real_time_column":1, "label_column":2}

r = csvReader(FILE, init_params=INIT_PARAMS)
print(r.get_num_of_total_requests())
# count = 0
# for i in r:
#     count += 1
# print(count)

# import sys
# sys.exit(1)
c = cachecow()
c.csv(FILE, init_params=INIT_PARAMS)

# for i, l in enumerate(c):
#     print("{}: {}".format(i, l))

# c.twoDPlot('v', 100000, 'cold_miss', figname="cold_miss_100000_v.png")
# c.twoDPlot('r', 100, 'cold_miss', figname="cold_miss_100_r.png")
# c.twoDPlot('r', 100, 'request_num', figname="request_num_100_r.png")


# p1 = c.profiler("LRU")
# s = p1.plotHRC(figname="HRC_LRU.png", auto_resize=True)

# p2 = c.profiler("LRU_2", cache_size=640000, num_of_threads=8)
p2 = cGeneralProfiler(r, "LRU_2", cache_size=64000, num_of_threads=8)
p2.plotHRC("HRC_Optimal.png", num_of_threads=8)


c.heatmap('v', 10000, "hit_rate_start_time_end_time", cache_size=20000, figname="heatmap_v.png", num_of_threads=8)
c.heatmap('r', 100, "hit_rate_start_time_end_time", cache_size=20000, figname="heatmap_r.png", num_of_threads=8)
c.differential_heatmap('r', 1000, "hit_rate_start_time_end_time", cache_size=20000, algorithm1="LRU",
                       algorithm2="Optimal", figname="diff_heatmap_r.png", num_of_threads=8)


