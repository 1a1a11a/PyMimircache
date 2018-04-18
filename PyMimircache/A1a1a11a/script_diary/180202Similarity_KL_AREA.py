# coding=utf-8
"""


"""

import os, sys, math, time, glob, socket, pickle, multiprocessing
import numpy as np
from PyMimircache import *
from PyMimircache.cacheReader.binaryWriter import TraceBinaryWriter
from PyMimircache.bin.conf import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


TRACE_DIR, NUM_OF_THREADS = initConf(trace_type="Akamai", trace_format="csv")

############################# CONST #############################
CACHE_SIZE = 2000
PLOT_ON_EXIST = True


############################ METRIC ##############################
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
    if min(len(p), len(q)) == 0:
        print("find one interval with no non-negative rd")
        return -1
    # else:
    #     print("length {} {} {}".format(length, len(p), len(q)))


    if len(p) < length:
        new_p.extend([p[-1]]*(length-len(p)))
    if len(q) < length:
        new_q.extend([q[-1]]*(length-len(q)))

    sum_p = sum(new_p)
    sum_q = sum(new_q)
    new_p = [new_p[i]/sum_p for i in range(len(new_p))]
    new_q = [new_q[i]/sum_q for i in range(len(new_p))]

    assert abs(sum(new_p)-1) < 0.0001 and abs(sum(new_q)-1) < 0.0001, "{} {}".format(sum(new_p), sum(new_q))
    epsilon /= length
    # print(new_p[:100])
    # print(new_q[:100])
    # print("{} {}".format(len(new_p), len(new_q)))

    # smoothing
    i = 0
    while new_q[i] == 0:
        new_q[i] = epsilon
        i += 1
    if i > 0:
        change_for_rest = epsilon * (i) / (len(new_q) - (i))
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


def transform_rd_list_to_rd_count(rd_list, percentage=True, normalize=False, cdf=True):
    rd_cnt = [0]* (np.max(rd_list) + 1)
    for rd in rd_list:
        if rd != -1:
            rd_cnt[rd] +=1

    if percentage:
        sum_cnt = sum(rd_cnt)
        for i in range(0, len(rd_cnt)):
            rd_cnt[i] = rd_cnt[i] / sum_cnt

    if cdf:
        for i in range(1, len(rd_cnt)):
            rd_cnt[i] += rd_cnt[i-1]

        if normalize:
            sum_cnt = sum(rd_cnt)
            for i in range(0, len(rd_cnt)):
                rd_cnt[i] = rd_cnt[i] / sum_cnt

    return rd_cnt


def cal_area(p, q):
    """
    p and q are two lists of rd_count cdf

    """

    # length = max(len(p), len(q))
    # if len(p) < length:
    #     p.extend([p[-1]] * (length - len(p)))
    # elif len(q) < length:
    #     q.extend([q[-1]] * (length - len(q)))
    #
    # return (sum(p) - sum(q))/length

    return sum(p) / len(p) - sum(q) / len(q)


############################ FUNC ################################
def plot_similarity_with_moving_window(dat, dat_type,
                                       similarity_type = "Area",
                                       window_size=20000,
                                       init_window_shift=0,
                                       moving_window_percent=0.1,
                                       ofolder="0218Similarity2"):


    reader = get_reader(dat, dat_type)
    rds = CLRUProfiler(reader).get_reuse_distance()
    similarity_score_list = []

    # ofolder = "{}{}/{}".format(ofolder, similarity_type, os.path.basename(dat))
    ofolder = "{}{}".format(ofolder, similarity_type)
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    figname = "{}/{}_{}_{}_{}.png".format(ofolder, os.path.basename(dat), window_size,
                                          init_window_shift,
                                          moving_window_percent)

    if not PLOT_ON_EXIST and os.path.exists(figname):
        return

    init_window_shift_size = int(window_size*init_window_shift)
    init_window = rds[init_window_shift_size : init_window_shift_size + window_size]
    init_window_cdf_cnt_percent = transform_rd_list_to_rd_count(init_window,
                                                                percentage=True,
                                                                normalize=False,
                                                                cdf=True)

    moving_window_size = int(window_size * moving_window_percent)
    current_window_pointer = init_window_shift_size + moving_window_size

    while current_window_pointer + window_size < len(rds):
        current_window = rds[current_window_pointer : current_window_pointer + window_size]
        current_window_cdf_cnt_percent = transform_rd_list_to_rd_count(current_window,
                                                                       percentage=True,
                                                                       normalize=False,
                                                                       cdf=True)

        if similarity_type == "KL":
            score = _calKL(init_window_cdf_cnt_percent, current_window_cdf_cnt_percent)
        elif "Area" in similarity_type:
            score = cal_area(init_window_cdf_cnt_percent, current_window_cdf_cnt_percent)
        else:
            print("unknown similarity measure   {}".format(similarity_type))
            return

        similarity_score_list.append(score)
        current_window_pointer += moving_window_size

    if similarity_type == "AreaFirstDerivative":
        plot_data = [similarity_score_list[i+1] - similarity_score_list[i] for i in range(len(similarity_score_list)-1)]
    else:
        plot_data = similarity_score_list

    plt.plot(plot_data)
    plt.xlabel("Time (v)")
    plt.ylabel(similarity_type)
    plt.tight_layout()
    plt.savefig(figname)
    print(figname)
    plt.clf()


def heatmap_KL(dat, dat_type, time_mode, time_interval, similarity_type, ofolder="0204KLHeatmap"):
    reader = get_reader(dat, dat_type)

    ofolder = "{}{}".format(ofolder, similarity_type)
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    figname = "{}/{}_{}_{}.png".format(ofolder, os.path.basename(dat), time_mode, time_interval)

    if not PLOT_ON_EXIST and os.path.exists(figname):
        return

    ph = PyHeatmap()
    ph.heatmap(reader, time_mode=time_mode, plot_type="KL_st_et", time_interval=time_interval, figname=figname)



############################ HELPER ################################
def run_parallel(func, fixed_args, change_args_list, max_workers=os.cpu_count()):
    futures_dict = {}
    results_dict = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ppe:
        for arg in change_args_list:
            futures_dict[ppe.submit(func, *fixed_args, *arg)] = arg
        for futures in as_completed(futures_dict):
            results_dict[futures_dict[futures]] = futures.result()

    return futures_dict


############################ RUNNABLE ################################
def run_akamai_seq(dat_folder, func, arg_list):
    dat_list = [f for f in glob.glob("{}/*.sort".format(dat_folder))]
    print("{} dat".format(len(dat_list)))
    for f in dat_list:
        print(f)
        func(dat="{}".format(f))


def run_akamai_parallel(dat_folder, func):
    dat_list = [f for f in glob.glob(dat_folder)]
    print("{} dat".format(len(dat_list)))

    with multiprocessing.Pool(os.cpu_count()) as p:
        p.map(func, dat_list)


def run_akamai_big_file(func, *args):
    func("/home/jason/ALL_DATA/Akamai/201610.all.sort.clean", *args)


def run_small_test(dat="w106", dat_type="cphy",):
    for ws in [200000]:
        # for iws in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]:
        # for mwp in reversed([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]):
        for mwp in reversed([0.02]):
            for st in ["Area", "AreaFirstDerivative"]:
                plot_similarity_with_moving_window(dat, dat_type,
                                                   similarity_type=st,
                                                   window_size=ws,
                                                   init_window_shift=0,
                                                   moving_window_percent=mwp)
                # break


def run_KL_parallel(dat_list=("small", ), dat_type="cphy"):
    arg_list = []

    for dat in dat_list:
        for iws in [0]:
            for mwp in reversed([0.02]):
                for ws in [200000]:
                    arg_list.append((dat, dat_type, "Area", ws, iws, mwp))
                    arg_list.append((dat, dat_type, "AreaFirstDerivative", ws, iws, mwp))
    run_parallel(plot_similarity_with_moving_window, fixed_args=(), change_args_list=arg_list, max_workers=48)


def run_KL_heatmap_parallel(dat_list=("small", ), dat_type="cphy"):
    arg_list = []
    for dat in dat_list:
        for ws in [3600 * 1000000, 600 * 1000000]:
            arg_list.append((dat, dat_type, "r", ws, "KL"))
    run_parallel(heatmap_KL, fixed_args=(), change_args_list=arg_list, max_workers=1)


if __name__ == "__main__":
    # run_KL_parallel(dat_list=["w{}".format(i) for i in range(106, 0, -1)])
    # run_KL_heatmap_parallel(dat_list=["w{}".format(i) for i in range(106, 0, -1)])
    # run_KL_parallel(dat_list=("/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", ), dat_type="akamai3")
    # run_KL_parallel()
    # for i in range(106, 0, -1):
    #     run_small_test("w{}".format(i))
    run_small_test()

