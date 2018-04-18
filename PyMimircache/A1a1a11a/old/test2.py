# coding=utf-8

from collections import defaultdict
import pickle
from PyMimircache import *
from PyMimircache.cacheReader.binaryReader import BinaryReader
import PyMimircache.CMimircache.CacheReader as c_cacheReader


DATA_PATH = "/root/cache/mimircache/mimircache/A1a1a11a/SLRU_DATA/"


def test1():
    c = Cachecow()
    # c_cacheReader.setup_reader("../data/trace.txt", 'c')
    # c.open("../data/trace.txt")
    # c.vscsi("../data/trace.vscsi")
    reader = BinaryReader("/root/disk2/ALL_DATA/traces/w96_vscsi1.vscsitrace", open_c_reader=False,
                          init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})
    reader2 = VscsiReader("/root/disk2/ALL_DATA/traces/w96_vscsi1.vscsitrace")
    # c.binary("../data/trace.vscsi", init_params={"label":6, "real_time":7, "fmt": "<3I2H2Q"})
    print(len(reader))
    print(len(reader2))
    for i in range(239518):
        reader.read_one_req()
        reader2.read_one_req()

    for i in range(10):
        print("{}: {}".format(reader.read_one_req(), reader2.read_one_req()))

def test2():
    reader = VscsiReader(DATA_PATH + "w96.vscsi")
    p = cGeneralProfiler(reader, "SLRUML", 2000, bin_size=250, num_of_threads=8,
                         cache_params={"hint_loc": DATA_PATH+"w96.y"})
    p.get_hit_rate()


def test3():
    reader = VscsiReader(DATA_PATH + "test.vscsi")
    bReader = BinaryReader(DATA_PATH + "test.realY", init_params={"fmt": "<b", "label": 1})
    for i, j in zip(reader, bReader):
        print("{}: {}".format(i, j))




if __name__ == "__main__":
    import time
    t1 = time.time()
    test3()

    print("time {}".format(time.time() - t1))
