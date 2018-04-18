# coding=utf-8

from collections import deque, defaultdict
import pickle, os, sys
import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyMimircache import *
from PyMimircache.bin.conf import *


TEST_FILE = "/home/jason/ALL_DATA/Akamai/dataCenterSplitted/binary/1008/160.2.65.82"
# TEST_FILE = "/home/jason/ALL_DATA/Akamai/dataCenter/1736"

DAT = "/home/jason/ALL_DATA/Akamai/day/20161001.sort.mapped.bin.LL"
DAT = "/home/jason/ALL_DATA/Akamai/day/test.mapped.bin.LL"

DAT_CSV = "/home/jason/ALL_DATA/Akamai/day/20161001.sort"

BINARY_READER_PARAMS = {"label": 1, "real_time": 2, "fmt": "<LL"}

# PATH = "/root/disk2/ALL_DATA/Akamai/binary/1392/185.197.29.30"
# PATH = "/run/shm/test.bin"



def queue(reader):
    """
    the queue theory described by Ymir
    :param dat:
    :return:
    """
    if 1:
        p = LRUProfiler(reader)
        frd = p.get_future_reuse_distance()
        l = [0]
        s = set()

        # # new
        # rd = p.get_reuse_distance()
        # c_frd = 0
        # for i in frd:
        #     if i == -1:
        #         c_frd += 1
        # c_rd = 0
        # for i in rd:
        #     if i == -1:
        #         c_rd += 1
        # print("num of uniq {}: {} {}".format(reader.get_num_of_uniq_req(), c_frd, c_rd))
        #
        # req = []
        # req_dict = {}
        # for i in reader:
        #     if type(reader) == binaryReader:
        #         req.append(i)
        #     else:
        #         if i in req_dict:
        #             req.append(req_dict[i])
        #         else:
        #             req_dict[i] = len(req_dict)+1
        #             req.append(len(req_dict))
        # reader.reset()


    for i, req in zip(frd, reader):
        if req in s:
            exist = True
        else:
            exist = False
        if i != -1:
            if exist:
                l.append(l[-1])
            else:
                l.append(l[-1] + 1)
                s.add(req)
        else:
            if exist:
                l.append(l[-1] -1)
                s.remove(req)
            else:
                l.append(l[-1])
        # with open("/tmp/queue_{}.pickle".format(os.path.basename(reader.file_loc)), 'wb') as ofile:
        #     pickle.dump(l, ofile)
    # plt.plot(l[: int(len(l)*0.6) ])
    plt.plot(l)
    # plt.yscale("log")
    plt.xlabel("virtual time")
    plt.ylabel("queue size")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda y, _:'{:.2%}'.format(y/len(l))))
    plt.savefig("0621queue/queue_{}.png".format(os.path.basename(reader.file_loc)), dpi=600)
    plt.clf()










######################################### BATCH ######################################

def batch_akamai_queue_mjolnir():
    TRACE_DIR = "/home/jason/ALL_DATA/Akamai/day/"
    for f in os.listdir(TRACE_DIR):
        if "LL" in f:
            print(f)
            reader = BinaryReader("{}/{}".format(TRACE_DIR, f), init_params=AKAMAI_BIN_MAPPED2)
            queue(reader)

def batch_cphy_queue_mjolnir():
    for i in range(106, 1, -1):
        if os.path.exists("/home/cloudphysics/traces/w{}_vscsi1.vscsitrace".format(i)):
            v = 1
        else:
            v = 2
        print(i, end="\t")
        queue(VscsiReader("/home/cloudphysics/traces/w{}_vscsi{}.vscsitrace".format(i, v)))







if __name__ == "__main__":
    batch_akamai_queue_mjolnir()


    # queue(csvReader("/home/jason/ALL_DATA/Akamai/day/test0.csv", init_params=AKAMAI_CSV))
    # queue(binaryReader("/home/jason/ALL_DATA/Akamai/day/test3.mapped.bin.LL", init_params=BINARY_READER_PARAMS))
    # queue(binaryReader("/home/jason/ALL_DATA/Akamai/day/20161002.sort.mapped.bin.LL", init_params=BINARY_READER_PARAMS))

    # queue(plainReader("../../data/trace.txt"))
    # queue(binaryReader("../../data/trace.vscsi", init_params=VSCSI1_BIN))
    # queue(vscsiReader("../../data/trace.vscsi"))



    # queue(binaryReader(DAT, init_params=BINARY_READER_PARAMS))
    # queue(binaryReader(TEST_FILE, init_params=BINARY_READER_PARAMS))
    # run1(DAT, N=100)
    # run(DAT, N=1000)
    # run(DAT, N=10000)
    # run(DAT, N=100000)
    # run(DAT, N=1000000)
    # run2(DAT, Ns=[100, 1000, 10000])
    # run2(TEST_FILE, Ns=[100, 1000, 10000])

    # run3(TEST_FILE, N=100000)
    # run3(DAT, N=100000)

    # load_save(DAT_CSV)

    # batch_run1_mjolnir()
    # batch_run2_mjolnir()
