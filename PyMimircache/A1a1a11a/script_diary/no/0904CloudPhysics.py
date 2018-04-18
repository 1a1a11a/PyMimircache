# coding=utf-8

from PyMimircache import Cachecow
from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.profiler.cHeatmap import CHeatmap
from PyMimircache.bin.conf import *


from concurrent.futures import ProcessPoolExecutor, as_completed


import os
import sys
import math
from collections import deque, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


TRACE_DIR, NUM_OF_THREADS = initConf("cphy", "variable")


def plot_heatmap(dat, mode="v", time_interval=5120, cache_size=200000, fig_folder="0904HeatmapsV"):

    c = Cachecow()
    # c.vscsi("{}/{}".format(TRACE_DIR, dat))
    c.vscsi(dat)

    figname = "{}/{}_heatmap_LRU_{}_{}.png".format(fig_folder, os.path.basename(dat), cache_size, mode)

    if os.path.exists(figname):
        return
    # c.heatmap(mode, "hit_ratio_start_time_end_time", algorithm="LRU",
    #           time_interval=time_interval,
    #           cache_size=cache_size, num_of_threads=NUM_OF_THREADS,
    #           figname=figname)
    c.heatmap(mode, "rd_distribution",
              time_interval=time_interval,
              figname=figname)

    print("{} done".format(figname))


def run():
    if not os.path.exists("0904HeatmapsV"):
        os.makedirs("0904HeatmapsV")

    plot_heatmap("../../data/trace.vscsi", time_interval=200, cache_size=0)
    # plot_heatmap("w106_vscsi1.vscsitrace", cache_size=200000)
    # plot_heatmap("w106_vscsi1.vscsitrace", mode="r", time_interval=1000 * 1000000) # , cache_size=800000)
    sys.exit(1)
    for i in range(66, 0, -1):
        if i<10:
            dat = "w0{}_vscsi1.vscsitrace".format(i)
        else:
            dat = "w{}_vscsi1.vscsitrace".format(i)
        if os.path.exists("{}/{}".format(TRACE_DIR, dat)):
            plot_heatmap_v(dat)
        else:
            dat = dat.replace("vscsi1", "vscsi2")
            plot_heatmap(dat)


def test():
    xydict = np.load("xydict.np")
    # cHeatmap().draw_heatmap(xydict)
    cmap = plt.cm.jet
    cmap.set_bad('w', 1.)
    plot_array = xydict

    plt.title("Heatmap")

    img = plt.imshow(plot_array, vmin=0, vmax=1, interpolation='nearest', origin='lower',
                             cmap=cmap, aspect='auto')

    plt.savefig("heatmap.png")


if __name__ == "__main__":
    run()
    # test()


