

from mimircache import *
from mimircache.profiler.evictionStat import *

def eviction_stat_reuse_dist_test(reader):
    # eviction_stat_reuse_dist_plot(reader, "Optimal", 1000, 'r', 10000000)
    eviction_stat_reuse_dist_plot(reader, "Optimal", 200, 'v', 100)

def eviction_stat_freq_test(reader):
    # eviction_stat_reuse_dist_plot(reader, "Optimal", 1000, 'r', 10000000)
    eviction_stat_freq_plot(reader, "Optimal", 200, 'v', 100, accumulative=True)
    eviction_stat_freq_plot(reader, "Optimal", 200, 'v', 100, accumulative=False)



if __name__ == "__main__":
    reader = vscsiReader("../data/trace.vscsi")
    reader = csvReader("../mimircache/data/trace.csv", init_params={"header": True, 'label_column': 4, 'real_time_column': 1})

    # reader = plainReader("multi1.trc")
    # reader = plainReader()
    print(reader.get_num_of_total_requests())
    print(reader.get_num_of_unique_requests())
    eviction_stat_reuse_dist_test(reader)
    # eviction_stat_freq_test(reader)


