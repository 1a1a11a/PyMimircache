# coding=utf-8



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from collections import defaultdict
from PyMimircache import *
from PyMimircache.bin.conf import *



def plot_popularity(dat, o_dir = "0702popularity", cdf=False):

    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    reader = CsvReader(dat, init_params=AKAMAI_CSV)
    d = reader.get_req_freq_distribution()
    d_count = defaultdict(int)
    max_freq = -1
    for _, v in d.items():
        d_count[v] += 1
        if v > max_freq:
            max_freq = v

    l = [0] * max_freq
    if not cdf:
        for k, v in d_count.items():
            l[k-1] = v
    else:
        for k, v in d_count.items():
            l[-k] = v
        for i in range(1, len(l)):
            l[i] = l[i-1]+l[i]
        for i in range(0, len(l)):
            l[i] = l[i] / l[-1]


    # plt.plot(l)
    if not cdf:
        plt.xlabel("freq")
        plt.ylabel("num of obj")
        plt.loglog(l)
        plt.savefig("{}/{}.png".format(o_dir, os.path.basename(dat)), dpi=600)
        plt.clf()
    else:
        plt.xlabel("cumulative ratio of objs")
        plt.ylabel("cumulative ratio of requests")
        plt.semilogy(l)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0%}'.format((x+1)/len(l))))
        plt.savefig("{}/{}_cdf.png".format(o_dir, os.path.basename(dat)), dpi=600)
        plt.clf()




def run_akamai1():
    for i in range(1, 31):
        t = i
        if i<10:
            t = '0' + str(i)
        plot_popularity("/home/jason/ALL_DATA/Akamai/day/201610{}.sort".format(t))
    plot_popularity("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean")


def run_akamai2():
    PATH = "/home/jason/ALL_DATA/akamai_new_logs/"
    for f in os.listdir(PATH):
        if os.path.isfile("{}/{}".format(PATH, f)):
            print(f)
            plot_popularity("{}/{}".format(PATH, f), cdf=True)


if __name__ == "__main__":
    # plot_popularity("/home/jason/ALL_DATA/Akamai/day/20161001.sort")
    run_akamai2()
