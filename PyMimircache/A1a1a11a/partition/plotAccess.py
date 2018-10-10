# coding=utf-8


import os, sys, time
from PyMimircache import *
from collections import defaultdict
import matplotlib.pyplot as plt
from PyMimircache.A1a1a11a.scripts.accessPattern import plot_access_pattern


################################ HELPER FUNCTION ##############################
def freq_distr(reader):
    """
    calculate the frequency distribution of requests in the reader,
    return dict[frequency] -> dict[items] -> [real_ts]
    :param reader:
    :param figname:
    :return:
    """
    import pickle
    if os.path.exists("/tmp/freq_distr_{}".format(reader.file_loc.split('/')[-2])):
        with open("/tmp/freq_distr_{}".format(reader.file_loc.split('/')[-2]), 'rb') as ifile:
            d_freq_distr = pickle.load(ifile)
    else:
        d = defaultdict(int)
        d2 = defaultdict(list)
        rt = reader.read_time_req()
        while rt:
            d[rt[1]] += 1
            d2[rt[1]].append(rt[0])
            rt = reader.read_time_req()

        d_freq_distr = defaultdict(dict)
        for k, v in d.items():
            d_freq_distr[v][k] = d2[k]

        reader.reset()
        with open("/tmp/freq_distr_{}".format(reader.file_loc.split('/')[-2]), 'wb') as ofile:
            pickle.dump(d_freq_distr, ofile)


    return d_freq_distr


################################ MAIN FUNCTION ##############################

def plt_access(dat, freq):
    """
    this function plot the access pattern for given freq 
    :param dat: 
    :param freq: 
    :return: 
    """
    # with csvReader(dat, init_params={"label_column": 5, "real_time_column": 1, "delimiter": '\t'}) as reader:
    with BinaryReader(dat, init_params={"fmt": "<LL", "label": 2, "real_time": 1}) as reader:
        d = freq_distr(reader)
        plot_access_pattern(d_freq_distr=d, freq=freq, pixel_limit=120,
                            sortby="time", xlabel="real time",
                            figname="{}_{}.png".format(dat.split("/")[-2], freq))


def plt_access_interval(dat, freq, CDF=True, folder="access_interval", figname=None, save=True, clf=True):
    """
    plot the #obj vs access interval for obj with freq 
    :param dat: 
    :param freq: 
    :param CDF: 
    :param figname: 
    :return: 
    """
    with BinaryReader(dat, init_params={"fmt": "<LL", "label": 2, "real_time": 1}) as reader:
        d = freq_distr(reader)
        l = []
        for k, v in d[freq].items():
            for i in range(1, len(v)):
                interval = int(v[i] - v[i-1])
                if len(l) <= interval:
                    l.extend([0] * (interval - len(l) + 1))
                l[interval] += 1
    if CDF:
        for i in range(1, len(l)):
            l[i] = l[i] + l[i-1]
        plt.ylabel("count (CDF)")
        if figname is None:
            figname = "{}_{}_CDF.pdf".format(dat.split("/")[-2], freq)
    else:
        plt.ylabel("count")
        if figname is None:
            figname = "{}_{}.pdf".format(dat.split("/")[-2], freq)

    plt.plot(l, label="freq {}".format(freq))
    plt.xlabel("interval(s)")
    if save:
        plt.legend(loc="best")
        plt.savefig("{}/{}".format(folder, figname))
    if clf:
        plt.clf()


################################ BATCH FUNCTION ##############################
def batch_plt_access_interval():
    DIR = "/root/disk2/ALL_DATA/Akamai/binary/"
    for folder in os.listdir(DIR):
        if os.path.isdir("{}/{}".format(DIR, folder)):
            print(folder)
            if os.path.exists("access_interval/{}.pdf".format(folder)):
                continue
            for i in range(2, 6):
                plt_access_interval("{}/{}/complete".format(DIR, folder), freq=i, clf=False, save=False)
            plt_access_interval("{}/{}/complete".format(DIR, folder), freq=6, clf=True, save=True,
                                figname="{}.pdf".format(folder))



if __name__ == "__main__":
    DIR = "/root/disk2/ALL_DATA/Akamai/binary/2025"
    # plt_access_interval("{}/complete".format(DIR), freq=4)
    batch_plt_access_interval()