# coding=utf-8
"""
a series of functions that runs Mithril
"""


import os, sys, time, pickle, glob
from collections import defaultdict
sys.path.append("../")
sys.path.append("./")
from PyMimircache import *
from PyMimircache.bin.conf import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from PyMimircache.profiler.twoDPlots import *

################################## Global variable and initialization ###################################
TRACE_DIR = "/home/jason/ALL_DATA/Akamai/day/"
NUM_OF_THREADS = 40

BINARY_INIT_PARAMS = {"label": 2, "real_time": 1, "fmt": "<LL"}




############################################# sub functions #############################################


def plot_heatmap(dat="/home/jason/ALL_DATA/MSR/msr_original/hm_1.csv"):
    c = Cachecow()
    c.csv(dat, init_params={"real_time": 1, "label": 5, "delimiter": ","})

    # iterate through the dat
    for req in c:
        print(req)
        break

    # some info about the trace
    print(c.stat())

    # plot hit ratio curve, we have more parameters to help do better plotting
    # or you can plot yourself as the function will return a dict hit ratio array
    # c.plotHRCs(["LRU", "LFU", "Optimal", "ARC", "SLRU"], cache_size=8000)

    # decay_coef=0.2, time_mode="v", time_interval=10000
    c.twoDPlot(plot_type="interval_hit_ratio", cache_size=800, time_mode="v", time_interval=2000) # , time_mode="r", time_interval=200000000)

    # start time-end time hit ratio heatmaps
    c.heatmap(plot_type="hr_st_et", cache_size=800, time_mode="r", time_interval=200000000)


def get_stat(dat="proj_0.csv"):
    c = Cachecow()
    # c.csv("/home/jason/ALL_DATA/MSR/msr_original/{}".format(dat), init_params={"real_time": 1, "label": 5, "delimiter": ","})
    c.csv(dat, init_params={"real_time": 1, "label": 5, "delimiter": ","})
    print(c.stat())
    # c.heatmap(plot_type="hr_st_et", cache_size=20000, time_mode="v", time_interval=10000)



############################################ main functions ##############################################
def run(dat):
    # reader = binaryReader(dat, init_params=BINARY_INIT_PARAMS)
    c = Cachecow()
    # c.binary(dat, init_params=AKAMAI_BIN_MAPPED)
    c.binary(dat, init_params={"label":1, 'real_time':2, "fmt": "<LL"})
    # c.binary("../data/trace.vscsi", init_params={"fmt": "<3I2H2Q", "label": 6, "real_time": 7, "size": 2})

    # size = c.num_of_unique_request()//2000
    size = 200000
    print("num of request {}, size = {}".format(c.num_of_req(), size))

    # p = cGeneralProfiler(c.reader, "mimir", size, bin_size=size//20,
    #                      cache_params=DEFAULT_PARAM, num_of_threads=NUM_OF_THREADS)
    # hr = p.get_hit_rate()
    c.plotHRCs(["LRU", "LFUFast"], cache_params=[None, None],
               auto_resize=False, cache_size=size, bin_size=size // 40,
               figname="test_{}.1028.png".format(time.time()),
               num_of_threads=NUM_OF_THREADS)

    return None


def mytest2():
    c = Cachecow()
    c.open("/home/jason/testJ")
    print(c.get_reuse_distance()[:20])


def mytest3():
    with open("testDat", "w") as ofile:
        for i in range(20):
            ofile.write("{}\n".format(1))

    c = Cachecow()
    c.open("testDat")
    # c.vscsi("../data/trace.vscsi")
    # c.plotHRCs(["Optimal"], cache_size=200, bin_size=20)
    print(c.get_reuse_distance())

def run_test():
    fd = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppe:
        for f in glob.glob("/home/jason/ALL_DATA/MSR/msr_original/*.csv"):
            fd[ppe.submit(get_stat, f)] = f

    for future in as_completed(fd):
        _ = future.result()


def test2():
    with open("w102.txt", "w") as ofile:
        with VscsiReader("/home/cloudphysics/traces/w102_vscsi1.vscsitrace") as reader:
        # with VscsiReader("../data/trace.vscsi") as reader:
            for r in CLRUProfiler(reader).get_reuse_distance():
                ofile.write("{}\n".format(r))

def mytest4():
    # reader = VscsiReader("../../data/trace.vscsi")


    reader = CsvReader("../../data/trace.csv", init_params={"real_time": 1, "label": 5, "delimiter": ","})
    p = CGeneralProfiler(reader, "LRU", 2000, 2000)
    print(p.get_hit_ratio())
    # print(p.get_eviction_age()[1, 200:240])


def mytest5():
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    req_freq_d = reader.get_req_freq_distribution()
    freq_count_d = defaultdict(int)
    for v in req_freq_d.values():
        freq_count_d[v] += 1


def mytest6():
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    popularity_2d(reader, plot_type="req", figname="req_popularity_185.232.99.68.anon.1.png")
    c = Cachecow()
    c.csv("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)


def mytest7():
    with open("t", "w") as ofile:
        for i in range(3):
            for i in range(2):
                ofile.write("{}\n".format(i))

    reader = VscsiReader("../data/trace.vscsi")
    reader = PlainReader("t")
    p = CGeneralProfiler(reader=reader, cache_alg="LRU", cache_size=2000, bin_size=800)
    hr = p.get_hit_result()
    print(hr.shape)
    print(hr)

def mytest8():
    from PyMimircache.cache.INTERNAL.ASig0414 import ASig0414
    # reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    reader = VscsiReader("../data/trace.vscsi")
    alg = ASig0414(cache_size=8000)
    for req in reader:
        alg.access(req)


###########################################################################################################
if __name__ == "__main__":

    mytest8()


