# coding=utf-8


import os
import sys
from PyMimircache import *
from PyMimircache.bin.conf import *
from PyMimircache.utils.timer import MyTimer


DATA = "/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon"
DATA = "/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.183.anon.1"
CACHE_SIZE = 800000
NUM_OF_THREADS = 48



def mytest1():
    c = Cachecow()
    c.csv(DATA, AKAMAI_CSV3)
    c.plotHRCs(["LRU", "LFUFast", "ARC", "SLRU", "Optimal"],  # , "LRU_2"
               [None, None, None, {"N": 2}, None],
               cache_size=CACHE_SIZE, bin_size=CACHE_SIZE // NUM_OF_THREADS // 20 + 1,
               auto_resize=True, num_of_threads=NUM_OF_THREADS,
               # use_general_profiler=True,
               save_gradually=True,
               figname="{}_HRC_autoSize.png".format(os.path.basename(DATA)))

def mytest2():
    CACHE_SIZE = 200000
    mt = MyTimer()
    c = Cachecow()
    # c.csv(DATA, AKAMAI_CSV3)
    c.vscsi("/home/cloudphysics/traces/w60_vscsi1.vscsitrace")
    mt.tick()
    p = cGeneralProfiler(c.reader, "LRU", CACHE_SIZE, bin_size=CACHE_SIZE)
    print(p.get_hit_ratio(cache_size=CACHE_SIZE, bin_size=CACHE_SIZE))
    mt.tick()

    p = cGeneralProfiler(c.reader, "akamai", CACHE_SIZE, bin_size=CACHE_SIZE)
    print(p.get_hit_ratio(cache_size=CACHE_SIZE, bin_size=CACHE_SIZE))
    mt.tick()

    # p.plotHRC()


def mytest3():
    c = Cachecow()
    c.csv(DATA, AKAMAI_CSV3)
    c.plotHRCs(["LRU", "akamai", "Optimal"],
               cache_size=800000, bin_size=CACHE_SIZE // NUM_OF_THREADS // 2 + 1,
               auto_resize=False, num_of_threads=NUM_OF_THREADS,
               ylimit=(0.6, 0.8),
               # use_general_profiler=True,
               save_gradually=True,
               figname="{}_HRC_autoSize.png".format(os.path.basename(DATA)))

def mytest3_2():
    c = Cachecow()
    c.csv(DATA, AKAMAI_CSV3)
    c.plotHRCs(["LRU", "akamai", "Optimal"],
               cache_size=CACHE_SIZE, bin_size=CACHE_SIZE // NUM_OF_THREADS // 2 + 1,
               auto_resize=False, num_of_threads=NUM_OF_THREADS,
               ylimit=(0.6, 0.8),
               # use_general_profiler=True,
               save_gradually=True,
               figname="{}_HRC_autoSize.png".format(os.path.basename(DATA)))

def mytest4():
    c = Cachecow()
    c.vscsi("../../data/trace.vscsi")
    c.plotHRCs(["LRU", "Optimal"],
               cache_size=CACHE_SIZE, bin_size=CACHE_SIZE // NUM_OF_THREADS // 20 + 1,
               auto_resize=True, num_of_threads=NUM_OF_THREADS,
               # use_general_profiler=True,
               ylimit=(0.6, 0.98),
               save_gradually=True,
               figname="{}_HRC_autoSize.png".format("test"))



if __name__ == "__main__":
    mytest1()
    # mytest2()
    # mytest3()
    # mytest3()

