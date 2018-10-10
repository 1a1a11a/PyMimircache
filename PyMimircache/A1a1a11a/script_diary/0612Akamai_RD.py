# coding=utf-8
"""
get rd and frd using C functions
"""

import os, sys, time, pickle, glob, multiprocessing
from PyMimircache import *
from PyMimircache.bin.conf import *


BINARY_READER_PARAMS = {"label": 1, "real_time": 2, "fmt": "<LL"}

FOLDER = "/home/jason/ALL_DATA/Akamai/day/"
FOLDER = "/home/jason/ALL_DATA/akamai_new_logs/"

def get_rd_frd(dat, folder=FOLDER):
    # reader = binaryReader(dat, init_params=BINARY_READER_PARAMS)
    
    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    # dat_name = os.path.basename(reader.file_loc).split(".")[0]
    dat_name = os.path.basename(reader.file_loc)
    if os.path.exists("{}/rd/{}.rd".format(folder, dat_name)) and \
        os.path.exists("{}/day/frd/{}.frd".format(folder, dat_name)):
        return


    print("input {}, begin {}".format(dat, dat_name))

    t1 = time.time()
    p = LRUProfiler(reader)
    p.save_reuse_dist("{}/rd/{}.rd".format(folder, dat_name), "rd")
    print("get rd for {} in {}".format(dat_name, time.time() - t1))

    t1 = time.time()
    p.save_reuse_dist("{}/day/frd/{}.frd".format(folder, dat_name), "frd")
    print("get frd for {} in {}".format(dat_name, time.time() - t1))









############################# RUNNABLE ############################
def batch_rd_frd():
    # dat_list = [f for f in glob.glob("/home/jason/ALL_DATA/Akamai/day/*.mapped.bin.LL")]
    dat_list = [f for f in glob.glob("/home/jason/ALL_DATA/Akamai/day/*.sort")]
    print("{} dat".format(len(dat_list)))

    with multiprocessing.Pool(12) as p:
        p.map(get_rd_frd, dat_list)


def batch_rd_frd2():
    dat_list = [f for f in glob.glob("{}/*.anon".format(FOLDER))]
    print("{} dat".format(len(dat_list)))

    with multiprocessing.Pool(12) as p:
        p.map(get_rd_frd, dat_list)




def cal_big_file():
    get_rd_frd("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")



if __name__ == "__main__":
    # batch_rd_frd()
    batch_rd_frd2()
    # cal_big_file()
                                                                               