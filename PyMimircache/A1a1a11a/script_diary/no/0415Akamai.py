#coding=utf-8
"""
get rd and frd 
"""

import os, sys, time, pickle, glob, multiprocessing
from PyMimircache import *
from PyMimircache.bin.conf import *


BINARY_READER_PARAMS = {"label": 1, "real_time": 2, "fmt": "<LL"}



def get_rd_frd(dat):
    # reader = binaryReader(dat, init_params=BINARY_READER_PARAMS)
    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    dat_name = os.path.basename(reader.file_loc).split(".")[0]
    if os.path.exists("/home/jason/ALL_DATA/Akamai/day/frd/{}".format(dat_name)):
        return

    print("begin {}".format(dat_name))

    t1 = time.time()
    p = LRUProfiler(reader)
    rd = p.get_reuse_distance()
    with open("/home/jason/ALL_DATA/Akamai/day/rd/{}".format(dat_name), 'wb') as ofile:
        pickle.dump(rd, ofile, protocol=4)
    del rd
    print("get rd for {} in {}".format(dat_name, time.time() - t1))

    t1 = time.time()
    frd = p.get_future_reuse_distance()
    with open("/home/jason/ALL_DATA/Akamai/day/frd/{}".format(dat_name), 'wb') as ofile:
        pickle.dump(frd, ofile, protocol=4)
    del frd
    print("get frd for {} in {}".format(dat_name, time.time() - t1))



############################# RUNNABLE ############################
def batch_rd_frd():
    # dat_list = [f for f in glob.glob("/home/jason/ALL_DATA/Akamai/day/*.mapped.bin.LL")]
    dat_list = [f for f in glob.glob("/home/jason/ALL_DATA/Akamai/day/*.sort")]
    print("{} dat".format(len(dat_list)))

    with multiprocessing.Pool(24) as p:
        p.map(get_rd_frd, dat_list)


def cal_big_file():
    get_rd_frd("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")



if __name__ == "__main__":
    batch_rd_frd()
    cal_big_file()
