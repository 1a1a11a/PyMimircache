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

import numpy as np
import pylab
from scipy.optimize import curve_fit

from PyMimircache import *
from PyMimircache.bin.conf import *
from PyMimircache.cacheReader.traceStat import TraceStat
from PyMimircache.utils.timer import MyTimer
from PyMimircache.A1a1a11a.myUtils.prepareRun import *
from PyMimircache.utils.printing import *
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import *

PLOT_ON_EXIST = True



def sigmoid_fit_plot(xdata, ydata, func_name,
                     xlabel=None, ylabel=None,
                     xlog=True, figname="sigmoid_fit.png", **kwargs):

    popt, sigmoid_func = sigmoid_fit(xdata, ydata, func_name)


    if xlog:
        # xbase = kwargs.get("xlogbase", 1.2)
        # start = math.log(xdata[0], xbase)
        # end = math.log(xdata[-1], xbase)
        # x = np.logspace(start, end, base=xbase, num=2000)
        x = np.geomspace(xdata[0], xdata[-1], num=2000)
        plt.xscale("log")
    else:
        x = np.linspace(xdata[0], xdata[-1], 2000)
    y = sigmoid_func(x, *popt)

    plt.plot(xdata, ydata, 'o', label='data')
    plt.plot(x, y, label='fit')
    plt.legend(loc='best')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.savefig(figname)
    print("fig saved {}".format(figname))
    plt.clf()



def trace_fit_sigmoid(dat, dat_type, top_N=100, dist_type="rd", func_name="tanh"):
    BASE = 1.08
    folder_name = "1106_SigmoidFitting_truncate_{}/{}/{}".format(dist_type, os.path.basename(dat), func_name)
    folder_name = "1106_SigmoidFitting_{}/{}/{}".format(dist_type, os.path.basename(dat), func_name)
    # folder_name = "test"
    output_name = "temp"
    if prepare(output_name, folder_name, PLOT_ON_EXIST):
        return

    dat_dist_list_filename = transform_datafile_to_dist_list(dat, dat_type, dist_type)
    # ofile = open("sample", "w")
    with open(dat_dist_list_filename) as ifile:
        for ind in range(top_N):
            line = ifile.readline()

            # if ind < 892:
            #     continue
            # print("read line {}".format(ind))
            req, dist_list = line.split(":")
            dist_list = [int(i) for i in dist_list.strip("[] \n").split(",")]

            dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=4, log_base=BASE)

            # now find the beginning and end of dist_count_list
            if "truncate" in folder_name:
                pos_begin = 0
                for i in range(len(dist_count_list)):
                    if dist_count_list[i] != dist_count_list[pos_begin]:
                        pos_begin = i
                        break
                if pos_begin > 10:
                    pos_begin -= 10
                pos_end = len(dist_count_list) - 1
                for i in range(pos_end, 0, -1):
                    if dist_count_list[i] < 0.98:
                        pos_end = i
                        break
                if len(dist_count_list) - 11 > pos_end:
                    pos_end += 10

                xdata = [BASE ** i for i in range(pos_begin, pos_end)]
                ydata = dist_count_list[pos_begin:pos_end]
            else:
                # no truncate
                xdata = [BASE ** i for i in range(len(dist_count_list))]
                ydata = dist_count_list
            try:
                sigmoid_fit_plot(xdata, ydata,
                                 xlabel=dist_type, ylabel="count",
                                 func_name=func_name,
                                 figname="{}/{}.png".format(folder_name, ind + 1))
            except Exception as e:
                print("{}: {}".format(ind, e))
            # print(dist_count_list)




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
    print("saved to {}".format(figname))
    # plt.savefig(figname.replace("png", "pdf"))
    plt.clf()





def test(dat="/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3"):
    from PyMimircache.profiler.twoDPlots import popularity_2d
    reader = get_reader(dat, dat_type)
    popularity_2d(reader, logX=True, logY=False, cdf=True, plot_type="req", figname="popularityReq_19.28.122.183.anon.png")


def run_fitting():
    for func_name in ["sigmoid1", "sigmoid2", "richard", "logistic", "gompertz", "tanh", "arctan"]:
        try:
            trace_fit_sigmoid(dat="/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3",
                        top_N=20000, dist_type="rd", func_name=func_name)
        except Exception as e:
            print(e)


def run_CPHY():
    t = MyTimer()
    for i in range(106, 0, -1):
        t.tick("{}".format(i))






if __name__ == "__main__":
    t = MyTimer()
    # transform_datafile_to_dist_list("small", "vscsi")
    # transform_datafile_to_dist_list("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3")
    # run_fitting()
    # trace_fit_sigmoid(dat="/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3",
    #             top_N=2000, dist_type="rd", func_name="arctan")

    t.tick()


