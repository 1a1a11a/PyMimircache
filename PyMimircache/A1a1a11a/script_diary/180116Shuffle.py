

import math
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tempfile import NamedTemporaryFile, SpooledTemporaryFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from PyMimircache import *
from PyMimircache.bin.conf import get_reader
from PyMimircache.A1a1a11a.myUtils.prepareRun import *
from PyMimircache import CLRUProfiler
from PyMimircache import PlainReader


def cal_dist(dat):
    last_vt = {}
    dists = []
    for n, r in enumerate(dat):
        if r in last_vt:
            dists.append(n - last_vt[r])
        else:
            dists.append(-1)
        last_vt[r] = n
    return dists, [0]*len(dat)

def eval_dat(dat):

    # cal rd
    rd = None
    with NamedTemporaryFile(mode="w") as t:
    # with open("/run/shm/testJ", "w") as t:
        for r in dat:
            t.write("{}\n".format(r))
            # t.write("{}\n".format(r).encode("utf8"))
        t.flush()
        reader = PlainReader(t.name)
        p = CLRUProfiler(reader, no_load_rd=True)
        rd = p.get_reuse_distance()
        hr = p.get_hit_ratio()

    return rd, hr

def eval_rds(rds):
    max_rd = 0
    min_rd = 1000000000000
    sum_rd = 0
    n_cold = 0

    for rd in rds:
        if rd == -1:
            n_cold += 1
            continue
        if rd > max_rd:
            max_rd = rd
        if rd < min_rd:
            min_rd = rd
        sum_rd += rd

    avg_rd = sum_rd/(len(rds) - n_cold)
    return len(rds), min_rd, max_rd, avg_rd, n_cold

def eval_LRU_HR(hr, cache_sizes=(20, 200, 2000, 20000)):
    HRs = []
    for size in cache_sizes:
        HRs.append(hr[size])
    return HRs


def _calKL(p, q, epsilon=0.01):
    """
    p q are two lists of rd count cdf percentage

    :param p:
    :param q:
    :param epsilon:
    :return:
    """
    # assert len(p) == len(q), "lists have different len {} {}".format(len(p), len(q))
    new_p = p[:]
    new_q = q[:]

    length = max(len(p), len(q))
    if len(p) < length:
        new_p.extend([p[-1]]*(length-len(p)))
    if len(q) < length:
        new_q.extend([q[-1]]*(length-len(q)))

    sum_p = sum(new_p)
    sum_q = sum(new_q)
    new_p = [new_p[i]/sum_p for i in range(len(new_p))]
    new_q = [new_q[i]/sum_q for i in range(len(new_p))]

    assert abs(sum(new_p)-1) < 0.0001 and abs(sum(new_q)-1) < 0.0001, "{} {}".format(sum(new_p), sum(new_q))
    epsilon /= max(len(new_p), len(new_p))
    # print(new_p[:100])
    # print(new_q[:100])
    # print("{} {}".format(len(new_p), len(new_q)))

    # smoothing
    i = 0
    while new_q[i] == 0:
        new_q[i] = epsilon
        i += 1
    if i > 0:
        change_for_rest = epsilon * (i-1) / (len(new_q) - (i-1))
        while i < len(new_q):
            new_q[i] -= change_for_rest
            i += 1

    KL = 0
    for i in range(len(new_p)):
        pi = new_p[i]
        qi = new_q[i]
        if pi == 0:
            continue
        KL += pi * math.log(pi/qi, math.e)
    return KL


def plot_rd_distribution(rds, dat_name, ofolder="0202ShuffleRDDistrLog"):
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)

    # this is avoid different shuffling has different range of RD, which will be a problem for KL
    # rd_count_list = [0] * len(rds)
    # for rd in rds:
    #     if rd != -1:
    #         rd_count_list[rd] += 1
    #
    # count_all = sum(rd_count_list)
    # rd_count_list[0] = rd_count_list[0] / count_all
    # for i in range(1, len(rd_count_list)):
    #     rd_count_list[i] = rd_count_list[i] / count_all
    #     rd_count_list[i] += rd_count_list[i-1]

    rd_count = defaultdict(int)
    for rd in rds:
        if rd != -1:
            rd_count[rd] += 1

    max_rd = max(rd_count.keys())
    rd_count_non_negative = sum(rd_count.values())

    rd_count_list = [0] * (max_rd + 1)
    for rd, count in rd_count.items():
        rd_count_list[rd] = count / rd_count_non_negative

    for i in range(1, len(rd_count_list)):
        rd_count_list[i] += rd_count_list[i-1]

    if "sorted" in dat_name:
        return rd_count_list

    plt.plot(rd_count_list)
    plt.xlabel("Reuse Distance")
    plt.ylabel("Percentage (CDF)")
    plt.xscale("log")
    plt.grid()
    plt.savefig("{}/{}_cdf.png".format(ofolder, dat_name))
    plt.clf()
    return rd_count_list

def _myshuffle(dat, n):
    half_size = len(dat)//2
    residue = len(dat) % 2
    for i in range(n):
        dat_new = dat[:]
        if residue:
            dat_new[-1] = dat[half_size]
        for j in range(half_size):
            dat_new[2*j] = dat[j]
            dat_new[j*2+1] = dat[half_size + residue + j]
        dat = dat_new

    return dat

def _myshuffle2(dat, n):
    dat_new = dat[:]
    for i in range(n):
        random.shuffle(dat_new)
    return dat_new


def mytest_myshuffle():
    print(_myshuffle([1,2,3,4,5,6,7,8], 1))
    print(_myshuffle([1,2,3,4,5,6,7,8,9], 1))
    print(_myshuffle([1,2,3,4,5,6,7,8], 2))
    print(_myshuffle([1,2,3,4,5,6,7,8,9], 2))
    print(_myshuffle([1,2,3,4,5,6,7,8], 3))
    print(_myshuffle([1,2,3,4,5,6,7,8,9], 3))



def mytest_shuffle_times(dat, dat_type, n=120):
    reader = get_reader(dat, dat_type)
    req = []
    for r in reader:
        req.append(r)

    data_info = [eval_dat(req)]
    rd_distr_list_ori = plot_rd_distribution(data_info[-1][0], "shuffleTimes_ori")

    req.sort()
    data_info.append(eval_dat(req))
    rd_distr_list_sorted = plot_rd_distribution(data_info[-1][0], "shuffleTimes_sorted")
    print("KL between ori and sorted {}".format(_calKL(rd_distr_list_ori, rd_distr_list_sorted)))

    # data_info = [cal_dist(req)]
    # req.sort()
    # data_info.append(cal_dist(req))


    for i in range(n):
        req = _myshuffle(req, 1)
        data_info.append(eval_dat(req))
        rd_distr = plot_rd_distribution(data_info[-1][0], "shuffleTimesRandomShuffle_{}_{}".format(dat, i+1))
        print("shuffle {} times KL with origin and sorted {:.2e} {:.2e}".format(i+1,
                    _calKL(rd_distr_list_ori, rd_distr), _calKL(rd_distr_list_sorted, rd_distr)))

    # for i in range(len(data_info)):
    #     if i == 0:
    #         print("original         ", end="")
    #     elif i == 1:
    #         print("sorted           ", end="")
    #     else:
    #         print("shuffle {} times ".format(i-1), end="")
    #     print("{!s:<60} {!s:<60}".
    #           format(eval_rds(data_info[i][0]), ["{:.2f}".format(hr) for hr in eval_LRU_HR(data_info[i][1])]))


def mytest_partial_shuffle(dat, dat_type,
                           shuffle_range=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.92, 0.95, 0.98, 1.0),
                           shuffle_times=(1, 2, 3, 5, 8, 16, 24, 38, 56, 80, 120, 200),
                           output_to_file=False):
    reader = get_reader(dat, dat_type)
    req = []
    for r in reader:
        req.append(r)


    data_info = [eval_dat(req)]
    rd_distr_list_ori = plot_rd_distribution(data_info[-1][0], "shuffleRegion_{}_ori".format(dat))
    req.sort()
    data_info.append(eval_dat(req))
    rd_distr_list_sorted = plot_rd_distribution(data_info[-1][0], "shuffleRegion_{}_sorted".format(dat))
    req_sort = req


    ofile = None
    if output_to_file:
        ofolder = "0202ShuffleRegionKL"
        if not os.path.exists(ofolder):
            os.makedirs(ofolder)
        ofile = open("{}/regionShuffleKL.{}".format(ofolder, dat), "w")


    s = "{} KL between ori and sorted {}".format(dat, _calKL(rd_distr_list_ori, rd_distr_list_sorted))
    print(s)
    if ofile:
        ofile.write("{}\n".format(s))


    for i in range(len(shuffle_range)):
        req = req_sort[:]
        shuffle_len = int(len(req) * shuffle_range[i])
        for j in range(1, shuffle_times[-1]+1):
            begin = random.randint(0, len(req)-shuffle_len)
            req = req[:begin] + _myshuffle(req[begin:begin+shuffle_len], 1) + req[begin+shuffle_len:]
            if j in shuffle_times:
                rd, hr = eval_dat(req)
                rd_distr = plot_rd_distribution(rd, dat_name="shuffleRegion_{}_{}_{}".format(dat, shuffle_range[i], j))
                s = "shuffle range {} {} times KL with origin and sorted {:.2e} {:.2e}".format(shuffle_range[i], j,
                            _calKL(rd_distr_list_ori, rd_distr), _calKL(rd_distr_list_sorted, rd_distr))
                print(s)
                if ofile:
                    ofile.write("{}\n".format(s))
                    ofile.flush()

    if ofile:
        ofile.close()

        # print("shuffle range {}".format(shuffle_range[i]))
        # data_info.append(eval_dat(req))


    return




    for i in range(len(data_info)):
        if i == 0:
            print("original           ", end="")
        elif i == 1:
            print("sorted             ", end="")
        else:
            print("shuffle range {:.2f} ".format(shuffle_range[i-2]), end="")
        print("{!s:<60} {!s:<60}".
              format(eval_rds(data_info[i][0]), ["{:.2f}".format(hr) for hr in eval_LRU_HR(data_info[i][1])]))


def run_parallel(func, arg_list, kwargs_list=None):
    futures_dict = {}
    results_dict = {}

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ppe:
        if kwargs_list == None:
            kwargs_list = [{}] * len(arg_list)
        for args, kwargs in zip(arg_list, kwargs_list):
            futures_dict[ppe.submit(func, *args, **kwargs)] = (args, kwargs)
        for futures in as_completed(futures_dict):
            results_dict[futures_dict[futures]] = futures.result()

    return futures_dict


if __name__ == "__main__":
    # run_parallel(mytest_partial_shuffle, [(dat, "cphy") for dat in range(106, 0, -1)])
    # run_parallel(mytest_partial_shuffle, [(106, "cphy"), (92, "cphy"), (78, "cphy")])
    run_parallel(mytest_partial_shuffle, [(92, "cphy"), (78, "cphy")])
    # mytest_myshuffle()
    # mytest_shuffle_times("small", "cphy")
    # mytest_partial_shuffle("small", "cphy")
    # mytest_partial_shuffle("w92", "cphy")
