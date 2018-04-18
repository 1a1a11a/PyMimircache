# coding=utf-8
"""
this module computes rd_distribution for hot objects
"""

import os
import sys
import time
import math
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from PyMimircache import *
from PyMimircache.bin.conf import *
from PyMimircache.cacheReader.traceStat import TraceStat
from PyMimircache.utils.timer import MyTimer


NO_PLOT_WHEN_EXIST = False


def plot_rd_distribution(dat, dat_type, top_N=100, cdf=True, plot_type="rd"):
    reader = get_reader(dat, dat_type)
    folder_name = "0412_{}DistriCDF_{}".format(plot_type, os.path.basename(dat))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # select top N obj
    top_N_obj_pair = TraceStat(reader, top_N_popular=top_N).get_top_N()
    access = {}
    for obj, freq in top_N_obj_pair:
        access[obj] = []

    if plot_type == "rd":
        # get reuse distance
        rd_list = CLRUProfiler(reader).get_reuse_distance()

    elif plot_type == "last_access":
        # get last access distance
        rd_list = []
        last_t_dict = {}
        for n, req in enumerate(reader):
            last_access_time = last_t_dict.get(req, n+1)
            rd_list.append(n - last_access_time)
            last_t_dict[req] = n
        reader.reset()
    else:
        print("error {}".format(plot_type))
        return

    # now get accesses
    for n, req in enumerate(reader):
        if req in access:
            access[req].append(rd_list[n])
    reader.reset()

    for obj, freq in top_N_obj_pair:
        if len(access[obj]) != freq:
            raise RuntimeError("obj {} number of access is different from freq {} {}".format(obj, len(access[obj]), freq))
        # transform a list of rd into the form of a list of count of rd
        # max_rd = -1
        max_rd = max(access[obj])
        # for rd in access[obj]:
        #     if rd > max_rd:
        #         max_rd = rd

        # num_bucket = int(math.ceil( math.log(max_rd, 2)) + 2)
        num_bucket = max_rd + 2
        l = [0] * num_bucket

        for rd in access[obj]:
            # 0: rd -1
            # 1: rd 0
            # 2: rd 1, 2
            # 3: rd 3, 4, 5, 6
            # 4: rd 7, 8, 9, 10, 11, 12, 13, 14
            # l[ int(math.ceil(math.log(rd+2, 2))) ] += 1
            if rd != -1:            # filter out cold miss
                l[rd] += 1

        if cdf:
            for i in range(1, len(l)):
                l[i] = l[i-1] + l[i]
            for i in range(len(l)):
                l[i] = l[i]/l[-1]
        # access[obj] = l

        plot([i for i in range(len(l))], l,
             xlabel="reuse distance" if plot_type=="rd" else plot_type, ylabel="percentage of count (cdf)", logY=False,
             figname="{}/{}_{}.png".format(folder_name, freq, obj))


    print("begin plotting")
    # for obj, freq in top_N_obj_pair:
    #     rd_count_list = access[obj]
    #     plot([i for i in range(len(rd_count_list))], rd_count_list,
    #     # plot([-1, 0] + [2**i-2 for i in range(2, len(rd_count_list))], rd_count_list,
    #          xlabel="reuse distance", ylabel="count", logY=True,
    #          figname="{}/{}_{}.png".format(folder_name, freq, obj))

    print("{} finshed".format(dat))




def plot(x, y, xlabel, ylabel, figname, logX=True, logY=False, text=None):
    # print(x)
    # print(y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    if logX:
        plt.xscale("log")
    if logY:
        plt.yscale("log")
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0%}'.format((x + 1) / len(y))))
    if text:
        plt.text(0, plt.ylim()[1]*0.8, "{}".format(text))
    plt.savefig(figname)
    # plt.savefig(figname.replace("png", "pdf"))
    plt.clf()



def plot_LFUOracle(dat, dat_type, folder_name=None):
    """
    what is the maximum hit ratio if we can cache all obj with freq > FREQ,
    the plot is hit ratio VS freq
    :param dat:
    :param dat_type:
    :return:
    """
    reader = get_reader(dat, dat_type)
    figname = "{}LFUOracle_{}_log.png".format("{}/".format(folder_name) if folder_name else "", os.path.basename(reader.file_loc))
    if NO_PLOT_WHEN_EXIST and os.path.exists(figname):
        return

    trace_stat = TraceStat(reader, keep_access_freq_list=True)
    num_req = (float) (trace_stat.num_of_requests)
    # a sorted descending list of obj, freq
    access_freq_list = trace_stat.get_access_freq_list()
    l = [access_freq_list[0][1]-1]
    for i in range(1, len(access_freq_list)):
        l.append(l[i-1] + access_freq_list[i][1]-1)


    plot(x=[i+1 for i in range(len(l))], y=[i/num_req for i in l],
         xlabel="cache size (obj)",
         ylabel="hit ratio",
         logX=True, logY=False, figname=figname)


def run_CPHY():
    t = MyTimer()
    for i in range(106, 0, -1):
        plot_LFUOracle(i, dat_type="cphy")
        t.tick("{}".format(i))






if __name__ == "__main__":
    t = MyTimer()
    TRACE_DIR, NUM_OF_THREADS = initConf("cphy", "variable")
    # plot_rd_distribution("small", dat_type="vscsi", top_N=8)
    # plot_rd_distribution("w92", dat_type="cphy", top_N=10000)
    plot_rd_distribution("/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.183.anon.1", dat_type="akamai3", top_N=200000, plot_type="last_access", cdf=False)
    # plot_rd_distribution("/home/jason/ALL_DATA/akamai3/layer/1/clean0922/oneDayData.sort", dat_type="akamai3", top_N=200000)
    # plot_rd_distribution("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3", top_N=200000, plot_type="last_access")
    # plot_rd_distribution("/home/jason/ALL_DATA/akamai3/layer/1/19.28.122.183.anon.1".format(TRACE_DIR), dat_type="akamai3", top_N=120)

    # for f in os.listdir("/home/jason/ALL_DATA/akamai3/original/"):
    #     plot_LFUOracle("/home/jason/ALL_DATA/akamai3/original/{}".format(f), dat_type="akamai3", folder_name="akamai3")
    #
    # sys.exit(1)
    # plot_LFUOracle("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3")
    # run_CPHY()


    # plot_LFUOracle("small", dat_type="vscsi")
    # t.tick()
    # plot_LFUOracle("{}/w92_vscsi1.vscsitrace".format(TRACE_DIR), dat_type="vscsi")
    # t.tick()
    # plot_LFUOracle("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon".format(TRACE_DIR), dat_type="akamai3")
    # t.tick()





