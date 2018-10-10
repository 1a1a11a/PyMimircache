# coding=utf-8

"""
this module currently offers function that are used to visualize Akamai data 

"""


import os, sys, time
from collections import defaultdict
from PyMimircache import *
import matplotlib
import matplotlib.pyplot as plt


def test1(dat, figname, CDF=False, fig_folder="accessDistr"):

    """
    given a trace file, this function plots the distribution of #request  
    :param dat: 
    :return: 
    """
    if not os.path.exists(dat) or (not os.path.isfile(dat)) or os.stat(dat).st_size == 0:
        return
    if os.path.exists("{}/{}.png".format(fig_folder, figname)):
        return

    # reader = binaryReader(dat, init_params={"label":1, "fmt": "<LL"})
    reader = CsvReader(dat, init_params={"label_column": 5, "real_time_column":1, "delimiter": "\t"})
    d = defaultdict(int)
    for i in reader:
        d[i] += 1

    l = [0] * int(max(d.values())+1)
    for k,v in d.items():
        l[v] += 1
    if CDF:
        for i in range(1, len(l)):
            l[i] = l[i-1] + l[i]
        figname = "{}/{}_CDF.png".format(fig_folder, figname)
    else:
        figname = "{}/{}.png".format(fig_folder, figname)

    for k,v in d.items():
        if v>100:
            print("{}: {}".format(v, k))

    plt.plot(l)
    plt.xlim(xmin=1)
    plt.xlabel("number of access")
    plt.ylabel("number of items")
    plt.xscale("log")
    plt.yscale("log")
    print("save fig {}".format(figname))
    plt.savefig(figname)
    plt.clf()



# 1a1a11a: ML interval
# 1a1a11a: Akamai


def test2(dir, fig_folder="reqSpread"):
    """
    given the splitted datacenter dir, 
    this function plots the number of requests vs the number of nodes a requests partitioned to  
    :param dir: 
    :param fig_folder: 
    :return: 
    """
    import pickle
    print("data center {} has {} nodes".format(dir[dir.rfind('/')+1:], len(os.listdir(dir))))
    xtick_len = len(os.listdir(dir)) + 1
    if os.path.exists("{}/{}.pickle".format(fig_folder, dir[dir.rfind("/")+1:])):
        with open("{}/{}.pickle".format(fig_folder, dir[dir.rfind("/")+1:]), 'rb') as ifile:
            d = pickle.load(ifile)
    else:
        d = defaultdict(set)
        for f in os.listdir(dir):
            if f != 'complete':
                reader = BinaryReader("{}/{}".format(dir, f), init_params={"label": 1, "fmt": "<LL"})
                for r in reader:
                    d[r].add(f)
        if not os.path.exists("{}/pickle".format(fig_folder)):
            os.makedirs("{}/pickle".format(fig_folder))

        with open("{}/pickle/{}.pickle".format(fig_folder, dir[dir.rfind("/")+1:]), 'wb') as ofile:
            pickle.dump(d, ofile)

    l_count = [0] * xtick_len
    for k,v in d.items():
        # if len(v) >= len(l_count):
        #     l_count.extend( [0]*(len(v) - len(l_count) + 1) )
        l_count[len(v)] += 1

    plt.plot(range(len(l_count)), l_count)
    plt.xlabel("#nodes item spread on")
    plt.ylabel("#items")
    plt.savefig("{}/{}.png".format(fig_folder, dir[dir.rfind("/")+1:]))
    plt.clf()


def test3(dir, access_num=(2, 3, 4), relative=False, fig_folder="partitionFig", figname=None):
    """
    this function plots the #requests vs #nodes the requests are spread on for a given access freq 
    for example, given access freq 3, for all items that have been accessed 3 times, how many nodes 
    these 3 requests are spread on 
    :param dir: dir to folder 
    :param access_num: list of freq 
    :param relative: the y axis is absolute number of requests or the percentage 
    :param fig_folder: 
    :return: 
    """
    import pickle
    if figname is None:
        figname = "{}/{}.png".format(fig_folder, dir[dir.rfind("/")+1:])

    print("data center {} has {} nodes".format(dir[dir.rfind('/')+1:], len(os.listdir(dir))))
    xtick_len = len(os.listdir(dir)) + 1
    if os.path.exists("{}/{}.pickle".format(fig_folder, dir[dir.rfind("/")+1:])):
        with open("{}/{}.pickle".format(fig_folder, dir[dir.rfind("/")+1:]), 'rb') as ifile:
            d = pickle.load(ifile)
    else:
        d = {}      # key -> [set, int]
        for f in os.listdir(dir):
            if f != 'complete':
                reader = BinaryReader("{}/{}".format(dir, f), init_params={"label": 1, "fmt": "<LL"})
                for r in reader:
                    if r not in d:
                        d[r] = [set(), 0]
                    d[r][0].add(f)
                    d[r][1] += 1

        if not os.path.exists("{}".format(fig_folder)) or \
                not os.path.exists("{}/pickle".format(fig_folder)):
            os.makedirs("{}/pickle".format(fig_folder))
        with open("{}/pickle/{}.pickle".format(fig_folder, dir[dir.rfind("/")+1:]), 'wb') as ofile:
            pickle.dump(d, ofile)

    for t in access_num:
        l_count = [0] * min(xtick_len, t+1)
        for k,v in d.items():
            if v[1] == t:
                l_count[len(v[0])] += 1
        if relative:
            sum_count = sum(l_count)
            for i in range(len(l_count)):
                l_count[i] = l_count[i]/sum_count

        plt.plot(range(len(l_count)), l_count, label="freq {}".format(t))

    plt.xlabel("#nodes item spread on")
    plt.ylabel("#items")
    plt.legend(loc="best")

    plt.savefig(figname)
    plt.clf()




def test1_batch():
    for folder in os.listdir(TRACE_DIR):
        if os.path.isdir("{}/{}".format(TRACE_DIR, folder)):
            test1("{}/{}/complete".format(TRACE_DIR, folder), figname=folder)


def test2_batch():
    for folder in os.listdir(TRACE_DIR):
        if os.path.isdir("{}/{}".format(TRACE_DIR, folder)):
            test2("{}/{}".format(TRACE_DIR, folder))




def test3_batch():
    for folder in os.listdir(TRACE_DIR):
        if os.path.isdir("{}/{}".format(TRACE_DIR, folder)):
            test3("{}/{}".format(TRACE_DIR, folder), relative=False,
                  access_num=range(2, 8), figname="{}/{}_{}".format("partitionFig", folder, 8))
            # test3("{}/{}".format(TRACE_DIR, folder), relative=False,
            #       access_num=range(8, 16), figname="{}/{}_{}".format("partitionFig", folder, 16))
            # test3("{}/{}".format(TRACE_DIR, folder), relative=False,
            #       access_num=range(16, 24), figname="{}/{}_{}".format("partitionFig", folder, 24))





if __name__ == "__main__":
    TRACE_DIR = "/root/disk2/ALL_DATA/Akamai/binary"
    DIR = "/root/disk2/ALL_DATA/Akamai/binary/1010"
    DIR = "/home/jason/ALL_DATA/Akamai/dataCenter/"

    test1("{}/{}".format(DIR, "1240"), CDF=False, figname="1240")
    # test2(DIR)
    # test3(DIR, access_num=[2, 3, 4, 5, 6, 7, 8, 9 ], relative=False)
    # test3(DIR, access_num=range(5, 20), relative=False)



    ######################## batch job ###########################
    # test2_batch()
    # test3_batch()