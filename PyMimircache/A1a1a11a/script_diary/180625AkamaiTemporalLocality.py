

import os, sys, time, random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyMimircache.bin.conf import *



def plt_lifetime_distribution(reader=None, dat=None, dat_type=None, cdf=True):
    if reader is None:
        assert dat is not None and dat_type is not None
        reader = get_reader(dat, dat_type)
    else:
        dat = reader.file_loc

    figname_base = "180625Lifetime_akamai4/lifetimeDistriCDF_{}".format(os.path.basename(dat))
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    first_access_time = {}
    last_access_time  = {}
    lifetime_dict = {}
    max_lifetime = 0
    access_freq_dict = defaultdict(int)

    all_data = []
    for req in reader:
        all_data.append(req)
    reader.reset()

    for shuffle_data in [False, True]:
        if shuffle_data:
            random.shuffle(all_data)
            figname = figname_base + ".shuffle.png"
        else:
            figname = figname_base + ".png"
        for n, req in enumerate(all_data):
            # if req not in first_access_time:
            #     first_access_time[req] = n
            first_access_time[req] = first_access_time.get(req, n)
            last_access_time[req] = n
            access_freq_dict[req] += 1

        for obj in first_access_time.keys():
            lifetime = last_access_time[obj] - first_access_time[obj]
            # NEW: remove cold miss
            if lifetime > 0:
                lifetime_dict[obj] = lifetime
                if lifetime > max_lifetime:
                    max_lifetime = lifetime

        lifetime_count_list = [0] * (max_lifetime + 1)
        for lifetime in lifetime_dict.values():
            lifetime_count_list[lifetime] += 1

        print("life time count {}".format(lifetime_count_list[:20]))

        if cdf:
            for i in range(1, len(lifetime_count_list)):
                lifetime_count_list[i] = lifetime_count_list[i - 1] + lifetime_count_list[i]
            for i in range(len(lifetime_count_list)):
                lifetime_count_list[i] = lifetime_count_list[i] / lifetime_count_list[-1]

        plot_kwargs = {
            "xlabel": "lifetime",
            "ylabel": "percentage of count (cdf)",
            "logY": False,
            "logX": True,
            "figname": figname[:-4] + "2.png"
        }
        plot([i for i in range(len(lifetime_count_list))], lifetime_count_list, **plot_kwargs)


        plot_kwargs = {
            "xlabel": "lifetime",
            "ylabel": "percentage of count (cdf)",
            "logY": False,
            "logX": False,
            "figname": figname[:-4] + "2.png"
        }
        plot([i for i in range(len(lifetime_count_list))], lifetime_count_list, **plot_kwargs)


        plot_kwargs = {
            "xlabel": "lifetime",
            "ylabel": "percentage of count (cdf)",
            "logY": True,
            "logX": True,
            "figname": figname[:-4] + "3.png"
        }
        plot([i for i in range(len(lifetime_count_list))], lifetime_count_list, **plot_kwargs)

        print("lifetime distribution {} plotted".format(figname))


def plot(x, y, xlabel, ylabel, figname, logX=True, logY=False, text=None, **kwargs):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    if "xticks" in kwargs:
        plt.gca().xaxis.set_major_formatter(kwargs["xticks"])
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


if __name__ == "__main__":
    FLD = "/home/jason/ALL_DATA/akamai4/splitByType/"

    # plt_lifetime_distribution(reader=CsvReader(os.path.join(FLD, "nyc.ecomm"), init_params=AKAMAI_CSV4))
    # sys.exit(1)

    for f in os.listdir(FLD):
        if f.count(".") == 1:
            print(os.path.join(FLD, f))
            dat = os.path.join(FLD, f)
            # print("{} {}".format(FLD, f))
            plt_lifetime_distribution(reader=CsvReader(dat, init_params=AKAMAI_CSV4))
