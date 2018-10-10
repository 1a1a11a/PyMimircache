# coding=utf-8
"""
(using save rd and frd to) compute the unique curve
"""

from PyMimircache import *
from PyMimircache.bin.conf import *
import os, time, sys, glob
from collections import deque, defaultdict
import matplotlib.pyplot as plt


def cal(dat, inteval):
    ret_list = []
    current_set = set()
    current_deque = deque()
    # reader = binaryReader("/home/jason/ALL_DATA/Akamai/day/{}.sort.mapped.bin.LL".format(dat), init_params=AKAMAI_BIN_MAPPED2)
    reader = BinaryReader("/home/jason/ALL_DATA/Akamai/day/{}.mapped.bin.LL".format(dat), init_params=AKAMAI_BIN_MAPPED2)
    counter = 0
    for r in reader:
        current_deque.append(r)
        counter += 1
        if counter == inteval:
            ret_list.append(len(set(current_deque))/inteval)
            counter = 0
            current_deque.clear()

        # if len(current_deque) == inteval:
        #     rm = current_deque.popleft()
        #     current_set.remove(rm)
        # elif len(current_deque) > inteval:
        #     raise RuntimeError("deque size {}, interval {}".format(len(current_deque), inteval))
        #
        # current_deque.append(r)
        # current_set.add(r)
        # counter += 1
        # if counter == inteval:
        #     ret_list.append(len(current_set)/inteval)
        #     counter = 0
    return ret_list


def run1(dat, N, folder="0616Uniq/"):
    """
    this function plots the number of unique obj in every N requests
    :param dat:
    :param N:
    :return:
    """

    if os.path.exists("{}/{}_{}.png".format(folder, os.path.basename(dat), N)):
        return
    # reader = binaryReader("/home/jason/ALL_DATA/Akamai/day/{}.sort.mapped.bin.LL".format(dat),
    #                       init_params=AKAMAI_BIN_MAPPED2)
    reader = BinaryReader("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean.mapped.bin.QL",
                          init_params=AKAMAI_BIN_MAPPED)
    print("{} total {}".format(os.path.basename(dat), len(reader)))
    s = set()
    l = []
    for n, i in enumerate(reader):
        s.add(i)
        if n % N == 0 and n!=0:
            l.append(len(s)/N)
            s.clear()
    fig = plt.plot(l)
    plt.xlabel("virtual time")
    plt.ylabel("percent of uniq obj in {} req".format(N))
    # print(l_all)
    # plt.boxplot(l_all)
    # fig.autofmt_xdate()

    plt.savefig("{}/{}_{}.png".format(folder, os.path.basename(dat), N))
    plt.clf()


def run2(dat, Ns, folder="0616UniqBox/"):
    """
    this function plot the boxplot of percent of unqi obj in every N requests,

    :param dat:
    :param Ns: a list of N
    :return:
    """
    if os.path.exists("{}/{}_box.png".format(folder, os.path.basename(dat))):
        return

    # r = binaryReader("/home/jason/ALL_DATA/Akamai/day/{}.sort.mapped.bin.LL".format(dat),
    #                       init_params=AKAMAI_BIN_MAPPED2)
    # r = binaryReader("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean.mapped.bin.QL",
    #                  init_params=AKAMAI_BIN_MAPPED)
    r = CsvReader(dat, init_params=AKAMAI_CSV3)
    print("{} total {}".format(os.path.basename(dat), len(r)))

    l_ratios = []
    l_deque = []
    l_set = []
    for _ in range(len(Ns)):
        l_ratios.append(list())
        l_deque.append(deque())
        l_set.append(defaultdict(int))

    for n, i in enumerate(r):
        if n % 10000000 == 0:
            print("{}: {}".format(time.time(), n))
        for idx in range(len(l_deque)):
            l_deque[idx].append(i)
            l_set[idx][i] += 1

            if len(l_deque[idx]) > Ns[idx]:
                obj = l_deque[idx].popleft()
                l_set[idx][obj] -= 1
                if l_set[idx][obj] == 0:
                    del l_set[idx][obj]

            if n != 0 and n % Ns[idx] == 0:
                l_ratios[idx].append(len(l_set[idx]) / Ns[idx])

    print("{}: {}: {}".format(len(l_ratios[0]), len(l_ratios[1]), len(l_ratios[2])))
    plt.xticks(rotation=90)
    plt.xlabel("#obj")
    plt.ylabel("Percent of unique obj")
    fig = plt.boxplot(l_ratios)
    # fig.autofmt_xdate()
    plt.tight_layout()

    plt.xticks(range(1, 1 + len(Ns)), Ns)
    plt.savefig("{}/{}_box.png".format(folder, os.path.basename(dat)))
    plt.clf()


def batch_plot1():
    dat_list = [f for f in glob.glob("/home/jason/ALL_DATA/Akamai/day/*.mapped.bin.LL")]
    for f in dat_list:
        if 'test' in f:
            continue
        dat = os.path.basename(f).split('.')[0]
        for j in [100, 1000, 10000, 100000, 1000000]:
            print("{} {}".format(dat, j))
            run1(dat, j)

def batch_plot2():
    dat_list = [f for f in glob.glob("/home/jason/ALL_DATA/Akamai/day/*.mapped.bin.LL")]
    for f in dat_list:
        if 'test' in f:
            continue
        dat = os.path.basename(f).split('.')[0]
        print("{}".format(dat))
        run2(dat, [100, 400, 1600, 6400, 25600, 102400, 409600, 4**7*100, 4**8*100, 4**9*100])

def plot_big_file():
    dat = "201610"
    # for j in [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]:
    #     print("{} {}".format(dat, j))
    #     run1(dat, j)

    run2(dat, [1000 * 2**i for i in range(1, 10)])

def batch_akamai3():
    PATH = "/home/jason/ALL_DATA/akamai3/original/"
    for f in os.listdir(PATH):
        if 'anon' not in f:
            continue
        print(f)
        dat = "{}/{}".format(PATH, f)
        run2(dat, [100, 400, 1600, 6400, 25600, 102400, 409600, 4 ** 7 * 100, 4 ** 8 * 100, 4 ** 9 * 100], folder="0912Akamai3Uniq")

def batch_akamai4():
    PATH = "/home/jason/ALL_DATA/akamai3/layer/1/"
    for f in os.listdir(PATH):
        if 'anon' not in f:
            continue
        print(f)
        dat = "{}/{}".format(PATH, f)
        run2(dat, [100, 400, 1600, 6400, 25600, 102400, 409600, 4 ** 7 * 100, 4 ** 8 * 100, 4 ** 9 * 100], folder="0912Akamai3-Layer1-Uniq")


if __name__ == "__main__":
    t1 = time.time()
    # print(cal("test", 1000))
    # run1("test", 1000)
    # batch_plot1()
    # batch_plot2()
    # plot_big_file()
    batch_akamai3()
    # batch_akamai4()
    print("time {}".format(time.time() - t1))