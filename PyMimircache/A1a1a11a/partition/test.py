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






def run3(dat, N):
    r = BinaryReader(dat, init_params={"label": 1, "real_time": 2, "fmt": "<LL"})
    print("{} total {}".format(os.path.basename(dat), len(r)))

    l_all = []
    s = set()
    for n, i in enumerate(r):
        s.add(i)
        if n % N == 0 and n!=0:
            l_all.append(s)
            s = set()

    check_set = l_all[-1]
    l_uniq = [len(check_set)*100/N]
    for i in range(len(l_all)-2, -1, -1):
        for e in l_all[i]:
            if e in check_set:
                check_set.remove(e)
        l_uniq.append(len(check_set)*100/N)

    l_uniq.reverse()
    print("l len {}".format(len(l_uniq)))

    plt.xlabel("#{} requests".format(N))
    plt.ylabel("Percentage")
    plt.plot(l_uniq)
    # plt.boxplot(l_all)
    # plt.loglog(l_uniq)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("{}_uniq_log2_{}.pdf".format(os.path.basename(dat), N))
    # print(l_all)
    plt.clf()



def load_save(dat):
    import pickle
    d = defaultdict(list)

    with open(dat) as ifile:
        for line in ifile:
            lineSplitted = line.strip("\n").split("\t")
            t = (lineSplitted[0])
            node = lineSplitted[1]
            ds = (lineSplitted[2])
            traffic = (lineSplitted[3])
            key = lineSplitted[4]

            d[key].append((t, node, ds, traffic))

    d_freq_distr = defaultdict(dict)
    for k, v in d.items():
        d_freq_distr[len(v)][k] = d[k]

    with open("{}.pickle".format(dat), "wb") as ofile:
        pickle.dump(d_freq_distr, ofile)



######################################### BATCH ######################################

def batch_run1_mjolnir():
    for f in os.listdir("/home/jason/ALL_DATA/Akamai/day/"):
        if "LL" in f:
            print(f)
            run1("{}/{}".format("/home/jason/ALL_DATA/Akamai/day/", f), N=10000)


def batch_run2_mjolnir():
    for f in os.listdir("/home/jason/ALL_DATA/Akamai/day/"):
        if "LL" in f:
            print(f)
            run2("{}/{}".format("/home/jason/ALL_DATA/Akamai/day/", f),
                 Ns=[100, 1000, 10000, 100000, 1000000])


if __name__ == "__main__":
    # queue(csvReader("/home/jason/ALL_DATA/Akamai/day/test0.csv", init_params=AKAMAI_CSV))
    # queue(binaryReader("/home/jason/ALL_DATA/Akamai/day/test3.mapped.bin.LL", init_params=BINARY_READER_PARAMS))
    queue(BinaryReader("/home/jason/ALL_DATA/Akamai/day/20161002.sort.mapped.bin.LL", init_params=BINARY_READER_PARAMS))

    # queue(plainReader("../../data/trace.txt"))
    # queue(binaryReader("../../data/trace.vscsi", init_params=VSCSI1_BIN))
    # queue(vscsiReader("../../data/trace.vscsi"))


    for i in range(93, 1, -1):
        if os.path.exists("/home/cloudphysics/traces/w{}_vscsi1.vscsitrace".format(i)):
            v = 1
        else:
            v = 2
        print(i, end="\t")
        # queue(binaryReader("/home/cloudphysics/traces/w{}_vscsi1.vscsitrace".format(i), init_params=VSCSI1_BIN))
        queue(VscsiReader("/home/cloudphysics/traces/w{}_vscsi{}.vscsitrace".format(i, v)))

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
