# coding=utf-8

from PyMimircache import *
from collections import defaultdict
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def freq_distr(reader):
    """
    calculate the frequency distribution of requests in the reader,
    return dict[frequency] -> dict[block] -> [ts]
    :param reader:
    :param figname:
    :return:
    """
    d = defaultdict(int)
    d2 = defaultdict(list)
    for n, r in enumerate(reader):
        d[r] += 1
        d2[r].append(n)

    d_freq_distr = defaultdict(dict)
    for k, v in d.items():
        d_freq_distr[v][k] = d2[k]

    reader.reset()
    return d_freq_distr



def plot_access_pattern(reader=None, freq=20, d_freq_distr=None, pixel_limit=6000, sortby="time",
                        xlabel="virtual time/%", figname=None):
    assert reader is not None or d_freq_distr is not None, "must provide reader or d_freq_distr"
    if reader:
        d_freq_distr = freq_distr(reader)

    if figname is None:
        if reader:
            figname = "{}_{}.pdf".format(reader.file_loc[reader.file_loc.rfind("/")+1:], freq)
        else:
            figname = "{}.pdf".format(freq)

    d = d_freq_distr[freq]
    if len(d)<10:
        return
    else:
        print("plotting freq {}".format(freq))
    counter = 1
    if sortby == "time":
        l = sorted(d.items(), key=lambda x:x[1][0])
    elif sortby == "block" or sortby == "label":
        l = sorted(d.items(), key=lambda x:x[0])

    else:
        print("cannot recognize sortby parameter {}".format(sortby))
        return None

    sample_ratio = 1
    if len(l) > pixel_limit//freq:
        sample_ratio = len(l)//(pixel_limit//freq)

    for n, x in enumerate(l):
        if n % sample_ratio == 0:
            if sortby == "block" or sortby == "label":
                plt.scatter(x[1], [x[0]]*len(x[1]))
            else:
                plt.scatter(x[1], [counter]*len(x[1]))
            counter += 1

    plt.gca().xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: '{:2.0f}%'.format(
            (x-plt.xlim()[0]) * 100 / (plt.xlim()[1] - plt.xlim()[0])))
    )
    plt.title("item access pattern", fontsize=16)
    if sortby == "block" or sortby == "label":
        plt.ylabel("time")
    elif sortby == 'time':
        plt.ylabel("items", fontsize=16)

    plt.xlabel(xlabel, fontsize=16)
    plt.savefig(figname)
    plt.clf()





if __name__ == "__main__":
    # reader = vscsiReader("../../data/trace.vscsi")
    reader = CsvReader("/home/A1a1a11a/ALL_DATA/Akamai/201610.all.sort.clean",  # "/scratch/jason/20161001.sort",
                       init_params={"header":False, "delimiter":"\t",
                                    "label_column":5, 'real_time_column':2})
    if not os.path.exists("freq_aka"):
        os.makedirs("freq_aka")

    d_freq_distr = freq_distr(reader)
    # plot_access_pattern_key(freq=2, d_freq_distr=d_freq_distr)
    for i in range(1, 10000000):
        if i % 1000000 == 0:
            print(i)
        plot_access_pattern(freq=i, d_freq_distr=d_freq_distr, sortby='time')