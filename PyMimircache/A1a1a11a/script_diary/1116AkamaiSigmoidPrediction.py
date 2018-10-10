# coding=utf-8

"""
    This function provides the fitting and prediction of (reuse) distance of objs according its sigmoid distance distribution curve


"""

import os
import sys
import time
import math
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pprint import pprint, pformat
from concurrent.futures import ProcessPoolExecutor, as_completed

from PyMimircache.utils.timer import MyTimer
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import *



BASE = 1.08

ENABLE_RESULT_PRINTING = True
PRINT_FITTING_FUNCTION = True # False
PRINT_DIST_COUNT_LIST = False


def generate_sample(dat, dat_type, top_N=100000, dist_type="rd", output_name="sample"):
    """
    generate sample data for prediction
    :param dat:
    :param dat_type:
    :param top_N:
    :param dist_type:
    :return:
    """

    dat_dist_list_filename = transform_datafile_to_dist_list(dat, dat_type, dist_type)
    ofile = open(output_name, "w")
    with open(dat_dist_list_filename) as ifile:
        for ind in range(top_N):
            line = ifile.readline()

            # code for generating samples
            if ind%10 == 0:
                ofile.write("{}".format(line))
                req, rd_list = line.split(":")

                rd_list = rd_list.strip("\n[]: ").split(",")
                print("{}: {} ts".format(ind, len(rd_list)))
            continue



def sigmoid_predict(xdata, ydata, func_name, yrange=(0.05, 0.95),
                    min_dist=-1, max_cache_size=10**8, max_attemps=10000):
    """
        this function fits the data and gives prediction

    :param xdata:
    :param ydata:
    :param func_name:
    :param yrange:
    :param min_dist:
    :param max_cache_size:
    :param max_attemps:
    :return:
    """


    x, x_min, x_max = -1, -1, -1
    if min_dist == -1:
        min_dist = 1

    popt, sigmoid_func = sigmoid_fit(xdata, ydata, func_name)
    if PRINT_FITTING_FUNCTION:
        print("{} {}".format(func_name, popt))

    if func_name == "arctan":
        func_inv = get_func("{}_inv".format(func_name))
        x_min = int(func_inv(yrange[0], *popt))
        x_max = int(func_inv(yrange[1], *popt))
        if x_min < 0:
            x_min = 0
    else:
        base = math.pow(max_cache_size, 1/max_attemps)
        for i in range(max_attemps):
            lastx = x
            x = int(base ** i)
            if x == lastx:
                continue
            y = sigmoid_func(x, *popt)
            if x_min == -1 and y > yrange[0]:
                x_min = x
                if x_min <= min_dist:
                    x_min = 0
            if x_max == -1 and y > yrange[1]:
                x_max = x
                break
    return x_min, x_max


def print_dist_count_list(dist_count_list, base):
    """
    a function for plotting the distance count list


    :param dist_count_list:
    :param base: the distance count list is a exponential increased array, ith element is the count of distance base**i
    """
    d = {}
    for n, count in enumerate(dist_count_list):
        d[int(base**n)] = dist_count_list[n]
    print_rd_list_origin = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    print_rd_list_origin_pos = 0
    print_rd_list = []
    for rd in sorted(d.keys()):
        if rd > print_rd_list_origin[print_rd_list_origin_pos]:
            print_rd_list.append(rd)
            print_rd_list_origin_pos += 1
            if print_rd_list_origin_pos == len(print_rd_list_origin):
                break
        print("\tdist_count ", end="\t")
        print(", ".join("%d: %.2f"%(rd, d[rd]) for rd in print_rd_list))


def train_and_predict(rd_list, init_training_size=20, decay_coefficient=1.0,
                      min_dist=-1, yrange=(0.05, 0.95), func_name="arctan"):
    """
        given a rd_list, using init_training_size points for fitting,
        and using the points left for prediction
        the fitting process happens every init_training_size points,
        and decay coefficient dictates the weight of history

        for example, if an obj is accessed 120 times, init_training_size=20,
        then the first 20 points will be used for training, the next 20 points will be used for verifying predictions,
        when it reaches the 40th point, another fitting is done using 40 data points
        with decay_coefficient penalized on the first 20 points

    :param rd_list:
    :param init_training_size: number of data points used for initial training
    :param decay_coefficient:
    :param min_dist: the minimal considered distance, if it is larger than 0,
                        then distance < min_dist have the same count as the count of min_dist
    :param yrange:  the range of confidence
    :param func_name:
    :return:
    """

    nx = 1

    if len(rd_list) < init_training_size:
        print("rd_list size less than {}".format(init_training_size))
        return
    # the training data points
    rd_list_train = rd_list[:init_training_size]
    # convert distance list to distance count list
    dist_count_list = transform_dist_list_to_dist_count(rd_list_train,
                                                        log_base=BASE,
                                                        cdf=True,
                                                        normalization=False)
    # normalize it, we don't do it last step because we want to keep the non-normalized data for adding points later
    dist_count_list_normalized = [i/dist_count_list[-1] for i in dist_count_list]
    predict_results = sigmoid_predict(xdata=[BASE ** i for i in range(len(dist_count_list_normalized))],
                                      ydata=dist_count_list_normalized,
                                      func_name=func_name, min_dist=min_dist, yrange=yrange)
    print(predict_results)
    plt.plot([BASE ** i for i in range(len(dist_count_list_normalized))], dist_count_list_normalized)
    plt.xscale("log")
    plt.savefig("a_{}.png".format(nx))

    if ENABLE_RESULT_PRINTING and PRINT_DIST_COUNT_LIST:
        print_dist_count_list(dist_count_list_normalized, base=BASE)

    # stat of prediction
    correct = 0
    wrong_left = 0
    wrong_right = 0
    new_rd = []
    for n, rd in enumerate(rd_list[init_training_size:]):
        if predict_results[0] <= rd <= predict_results[1]:
            correct += 1
        else:
            if rd < predict_results[0]:
                wrong_left += 1
            elif rd > predict_results[1]:
                wrong_right += 1
            else:
                print("error {} {}".format(rd, predict_results))

        new_rd.append(rd)
        if len(new_rd) == init_training_size:
            # now is time to do the fitting again
            dist_count_list = [i * decay_coefficient for i in dist_count_list]
            for rd in new_rd:
                add_one_rd_to_dist_list(rd, dist_count_list, 1-decay_coefficient, base=BASE)

            # if I don't use ewma, just use a whole bunch of history, seems not good
            # dist_count_list = transform_dist_list_to_dist_count(rd_list[:init_training_size + n],
            #                                                     log_base=BASE,
            #                                                     cdf=True,
            #                                                     normalization=False)

            dist_count_list_normalized = [i/dist_count_list[-1] for i in dist_count_list]
            predict_results = sigmoid_predict(xdata=[BASE ** i for i in range(len(dist_count_list_normalized))],
                                              ydata=dist_count_list_normalized,
                                              func_name=func_name, min_dist=min_dist, yrange=yrange)
            # this is for observing fitting
            nx += 1
            print("{} {}".format(nx, predict_results))
            # plt.plot([BASE ** i for i in range(len(dist_count_list_normalized))], dist_count_list_normalized)
            # plt.xscale("log")
            # plt.savefig("a_{}.png".format(nx))

            new_rd.clear()


    precision = correct/(correct+wrong_left+wrong_right)
    precision_without_leftwrong = (correct + wrong_left) / (correct + wrong_left + wrong_right)
    if ENABLE_RESULT_PRINTING:
        if predict_results[1] > 100000:
            print(rd_list)
        print("\t{!s:<20} - {:>8}/({:^6}+{:^6}+{:^6}) \t\t{:.2f} {:.2f}".
              format(predict_results, correct, correct, wrong_left, wrong_right,
                     precision, precision_without_leftwrong ))
    return precision


def cal_precision(dat, decay_coefficient, yrange, min_dist, func_name, training_size):
    """
    this function takes a dat file as input, the data file consists obj and a list of reuse distance,
    the format is obj : [rd1, rd2, ... rdn]

    see train_and_predict for params

    :param dat:
    :param decay_coefficient:
    :param yrange:
    :param min_dist:
    :param func_name:
    :param training_size:
    :return:
    """
    precision_list = []
    with open(dat) as ifile:
        for n, line in enumerate(ifile):
            req, rd_list = line.split(":")
            rd_list = rd_list.strip("\n[]: ").split(",")
            rd_list = [int(rd) for rd in rd_list]

            if len(rd_list) > 2000:
                continue

            # filter out data if rd_list size is less than 2.5 times training size
            if len(rd_list)<=training_size*2.5:
                break

            if ENABLE_RESULT_PRINTING:
                print("{}\t{}".format(n, req), end=": ")
            try:
                precision = train_and_predict(rd_list, decay_coefficient=decay_coefficient, yrange=yrange,
                                              min_dist=min_dist, func_name=func_name, init_training_size=training_size)
                precision_list.append(precision)
            except Exception as e:
                print(e)

        precision_list_np = np.array(precision_list)
        if ENABLE_RESULT_PRINTING:
            print("{} precision avg {:.2f}, min {:.2f}, max {:.2f}, median {:.2f}".format(
                len(precision_list), np.average(precision_list_np), min(precision_list),
                max(precision_list), np.median(precision_list_np)))
        return precision_list


def run1():
    DEFAULT_DECAY_COEFFICIENT = 0.5
    DEFAULT_YRANGE = (0.05, 0.95)
    DEFAULT_MIN_DIST = 20
    DEFAULT_FUNCNAME = "tanh2"
    DEFAULT_TRAINING_SIZE = 20

    decay_coefficients  =   [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    yranges             =   [(0.05, 0.95), (0.05, 0.80), (0.05, 0.60)]
    min_dists           =   [-1, 1, 5, 20, 80, 200, 2000]
    func_names          =   ["sigmoid1", "sigmoid2", "richard", "logistic", "gompertz", "tanh2", "arctan2"]
    training_sizes      =   [2, 3, 5, 8, 12, 20, 50, 120, 200]

    plot_list = []
    for param in [decay_coefficients, yranges, min_dists, func_names, training_sizes]:
        # try:
        if 1:
            print(param)
            decay_coefficient = DEFAULT_DECAY_COEFFICIENT
            yrange = DEFAULT_YRANGE
            min_dist = DEFAULT_MIN_DIST
            func_name = DEFAULT_FUNCNAME
            training_size = DEFAULT_TRAINING_SIZE

            for p in param:
            # for ewma_coefficient in decay_coefficients:
                if param == decay_coefficients:
                    decay_coefficient = p
                    xlabel = "decay coefficient"
                    break
                elif param == yranges:
                    yrange = p
                    xlabel = "yrange"
                elif param == min_dists:
                    min_dist = p
                    xlabel = "minimal distance"
                elif param == func_names:
                    func_name = p
                    xlabel = "function"
                elif param == training_sizes:
                    training_size = p
                    xlabel = "training size"
                else:
                    raise RuntimeError("unknown param {}".format(param))

                precision_list = cal_precision("sample", decay_coefficient, yrange, min_dist, func_name, training_size)
                plot_list.append(precision_list)
                print(p)
            if len(plot_list) == 0:
                continue
            plt.boxplot(plot_list, labels=param)
            plt.ylabel("precision")
            plt.xlabel(xlabel)
            plt.savefig("{}.png".format(xlabel))
            plt.clf()
            plot_list.clear()
        # except Exception as e:
        #     print(e)


def run1_parallel(num_threads=4):
    DEFAULT_DECAY_COEFFICIENT = 0.5
    DEFAULT_YRANGE = (0.05, 0.95)
    DEFAULT_MIN_DIST = 20
    DEFAULT_FUNCNAME = "tanh2"
    DEFAULT_TRAINING_SIZE = 20

    decay_coefficients  =   [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    yranges             =   [(0.05, 0.95), (0.05, 0.80), (0.05, 0.60)]
    min_dists           =   [-1, 1, 5, 20, 80, 200, 2000]
    func_names          =   ["sigmoid1", "sigmoid2", "richard", "logistic", "gompertz", "tanh2", "arctan2"]
    training_sizes      =   [2, 3, 5, 8, 12, 20, 50, 120, 200]

    plot_dict = defaultdict(dict)
    future_dict = {}
    with ProcessPoolExecutor(max_workers=num_threads) as pp:
        for param in [decay_coefficients, yranges, min_dists, func_names, training_sizes]:
            print(param)
            decay_coefficient = DEFAULT_DECAY_COEFFICIENT
            yrange = DEFAULT_YRANGE
            min_dist = DEFAULT_MIN_DIST
            func_name = DEFAULT_FUNCNAME
            training_size = DEFAULT_TRAINING_SIZE

            for p in param:
            # for ewma_coefficient in decay_coefficients:
                if param == decay_coefficients:
                    decay_coefficient = p
                    xlabel = "decay coefficient"
                elif param == yranges:
                    yrange = p
                    xlabel = "yrange"
                elif param == min_dists:
                    min_dist = p
                    xlabel = "minimal distance"
                elif param == func_names:
                    func_name = p
                    xlabel = "function"
                elif param == training_sizes:
                    training_size = p
                    xlabel = "training size"
                else:
                    raise RuntimeError("unknown param {}".format(param))

                future = pp.submit(cal_precision, "sample", decay_coefficient, yrange, min_dist, func_name, training_size)
                future_dict[future] = (xlabel, p)

        count = 0
        for future in as_completed(future_dict):
            xlabel, p = future_dict[future]
            try:
                precision_list = future.result()
                plot_dict[xlabel][p] = precision_list
            except Exception as e:
                print("{}".format(e))
            count += 1
            print("{}/{}".format(count, len(future_dict)))

            # put it here for plotting every time
            for xlabel, precision_list_dict in plot_dict.items():
                plot_list = []
                xlabels = []
                for p, precision_list in sorted(precision_list_dict.items(), key=lambda x: x[0]):
                    xlabels.append(p)
                    plot_list.append(precision_list)

                plt.boxplot(plot_list, labels=xlabels)
                plt.ylabel("precision")
                plt.xlabel(xlabel)
                plt.savefig("{}_parallel.png".format(xlabel))
                plt.clf()



if __name__ == "__main__":
    t = MyTimer()
    # generate_sample(dat="/home/jason/ALL_DATA/akamai3/original/19.28.122.183.anon", dat_type="akamai3")

    # cal_precision("sample", decay_coefficient=1.0, yrange=(0.0001, 0.95),
    #               min_dist=-1, func_name="arctan", training_size=20)

    # train_and_predict([-1, 11505, 20182, 4308, 29975, 27140, 9200, 5861, 5343, 32366, 58722, 7138, 66441, 83951, 114162, 62329, 10129, 109049, 42856, 61571, 166250, 14240, 38126, 18345, 120647, 133586, 59769, 25366, 33852, 95165, 2196, 40468, 33702, 29512, 71669, 23178, 4699, 73062, 50572, 22551, 60136, 61565, 6860, 12330, 26070, 10496, 5163, 14303, 12736, 17505, 1900, 299, 67661, 19487, 22064, 17037, 27626, 14601, 79381, 11010, 19495, 40361, 27596, 19679, 20120, 20502, 42439, 16303, 1626, 6839, 10873, 10353, 12961, 15840, 562, 2417, 3814, 17584, 31936, 35701, 9007, 9815, 11863, 3789, 13940, 3274, 2613, 13241, 9333, 15472, 63420, 15662, 14505, 72159, 6497, 46772, 51224, 41746, 51687, 11797, 23895, 9113, 15552, 17508, 7852, 25870, 1382, 5075, 80745, 8573, 49781, 53464, 12092, 52264, 25530, 18866, 21061, 4052, 23145, 5351, 10461, 954, 2536, 37310, 22736, 563, 17248, 13794, 20514, 6430, 39479, 14373, 26002, 1470, 45028, 5075, 8146, 14432, 3833, 41652, 6934, 23, 8020, 66468, 17549, 17853, 101653, 15383, 13873, 50432, 3901, 34459, 663, 29055, 70993, 49850, 31062, 43479, 4916, 74205, 35510, 14266, 19233, 44246, 28439, 755, 4501, 2007, 211914, 37803, 16869, 2173, 25264, 134518, 105317, 115073, 18308, 6993, 39132, 36700, 5506, 9245, 4463, 26876, 50134, 2869, 5372, 24167, 52481, 61438, 57480, 43198, 83948, 76518, 82036, 19472, 8486, 6432, 11160, 41, 40507, 20712, 9067, 8552, 4851, 6120, 67290, 251, 21025, 1802, 12471, 30024, 20179, 33221, 12550, 4045, 16921, 44104, 21031, 39111, 2341, 27572, 21251, 17584, 14267, 3891, 3501, 47520, 10177, 3879, 16861, 613, 28366, 1771, 1341, 3893, 10412, 8207, 10725, 35015, 9850, 11406, 69199, 103928, 65792, 16369, 27384, 36090, 5421, 78218, 27488, 14936, 48913, 31079, 10355, 7039, 7030, 3601, 10529, 18272, 91043, 10473, 20197, 44797, 29751, 11152, 16849, 42909, 21406, 2854, 27692, 39501, 1465, 20929, 6180, 1445, 3755, 10761, 2313, 28249, 48006, 66177, 17402, 25348, 19952, 39282, 57084, 3228, 19066, 32069, 21824, 60877, 7925, 2967, 14897, 13944, 24263, 43787, 82191, 8947, 27483, 75241, 55126, 73748, 39703, 127316, 20987, 94977, 68183, 73320, 5467, 27470, 25506, 9157, 8713, 10875, 5386, 56653, 15480, 33482, 2246, 18627, 76786, 48965, 21495, 45437, 71249, 2830, 8557, 19106, 62235, 12395, 56294, 45989, 30539, 12264, 15557, 11204, 29009, 7448, 98373, 49107, 28170, 71516, 45909, 32538, 29948, 9040, 63128, 14650, 4522, 9012, 36653, 7917, 19483, 32878, 12805, 77784, 32550, 84052, 2461, 17358, 5490, 42250, 4659, 4118, 10862, 32277, 22347, 36544, 10341, 6599, 84197, 15815, 12582, 78791, 16102, 14594, 26969, 1930, 5598, 46280, 52357, 29717, 16311, 26231, 93316, 22349, 32136, 21183, 5263, 2755, 79624, 13635, 6501, 35713, 76742, 23113, 46696, 6335, 47315, 4071, 38045, 36225, 5429, 1674, 17925, 122370, 20659, 39303, 61876, 42570, 62413, 33382, 52931, 13757, 40143, 32841, 56862, 6480, 6717, 13208, 50378, 29606, 2386, 25934, 25495, 69440, 50780, 27301, 9930, 8852, 34761, 44274, 7983, 39216, 12749, 67337, 32849, 11987, 17472, 10107, 6144, 6081, 17598, 31437, 10060, 15439, 2325, 3014, 6088, 366, 27014, 34952, 1290, 16584, 8148, 18805, 15707, 22668, 14308, 10374, 14226, 19024, 14002, 9730, 7817, 3073, 1066, 11304, 4014, 5282, 4772, 2496, 8921, 28701, 24471, 1242, 13542, 27337, 25816, 9009, 5753, 3677, 14733, 8805, 23063, 5484, 2396, 4214, 8560, 4127, 4881, 9049, 20268, 12508, 55796, 9160, 5067, 11235, 3959, 21859, 17776, 18535, 11081, 23530, 25895, 7516, 33587, 71672, 43524, 41457, 12469, 8352, 33744, 10594, 20560, 29926, 26190, 18979, 36376, 49646, 82063, 6135, 22596, 12521, 9994, 21765, 18964, 8564, 4464, 57094, 1808, 43527, 27226, 21482, 34527, 31171, 13377, 40598, 10674, 23958, 20094, 16781, 29903, 6717, 22610, 41304, 22359, 73556, 8413, 5517, 16136, 23942, 31945, 27876, 16939, 2244, 924, 63244, 56316, 50651, 607, 1627, 51935, 5050, 4747, 28258, 9876, 14631, 43916, 12512, 76019, 33857, 48187, 2180, 13942, 50397, 4057, 30567, 23794, 47471, 3638, 95007, 23461, 57146, 23343, 123347, 91965, 51316, 36030, 81130, 53116, 61790, 21028, 1352, 48274, 44216, 78783, 141325, 12401, 194274, 104108, 19856, 65998, 22305, 29924, 24934, 5765, 12437, 41879, 15374, 40882, 13405, 13365, 38722, 775, 1086, 30346, 102191, 32699, 7443, 19504, 2696, 26343, 2987, 51384, 83726, 22023, 5235, 44028, 27172, 6400, 433, 42485, 32893, 1444, 41683, 24729, 7823, 47828, 19791, 1005, 28571, 23744, 36, 14193, 11226, 33024, 29643, 44474, 14497, 23236, 6216, 12855, 4156, 17858, 22780, 2504, 7523, 31867, 19400, 6603, 27828, 28000, 24086, 36654, 41260, 4268, 30452, 14484, 2602, 1732, 35989, 23859, 45910, 10904, 41568, 30771, 56107, 17694, 51787, 20345, 4709, 32437, 55795, 56380, 21594, 17196, 8109, 24024, 42656, 16839, 9840, 36693, 1699, 19250, 18466, 59342, 38362, 44490, 6247, 31058, 12194, 129784, 68403, 5615, 34747, 129455, 180146, 109014, 15498, 124133, 22648, 103917, 110453, 22490, 132889, 34425, 41929, 29074, 10124, 28244, 12860, 46676, 63307, 48490, 28285, 3129, 20839, 30431, 140031, 200733, 17735, 51291, 54954, 36415, 118940, 63144, 4524, 8018, 37557, 18320, 28205, 30579, 20675, 35445, 17584, 2408, 4352, 8468, 21349, 13490, 18952, 40820, 35903, 51301, 21584, 4225, 31395, 40860, 98433, 7670, 61352, 618, 14799, 16175, 76758, 17506, 11790, 38498, 5606, 23289, 89131, 7070, 12460, 7877, 25036, 25069, 61420, 5113, 6594, 40863, 8737, 27155, 41813, 10626, 9256, 55240, 8148, 13084, 23025, 10650, 53486, 52557, 8104, 41060, 13797, 965, 6348, 6586, 3224, 28797, 1899, 6125, 15605, 14589, 18243, 25571, 14518, 1220, 6449, 20853, 11798, 3473, 40623, 72095, 60234, 16569, 4284, 27752, 21050, 38097, 28335, 11851, 14336, 19911, 69311, 12906, 13085, 84161, 73005, 55032, 67561, 15537, 37192, 22137, 64890, 57354, 38918, 113261, 21165, 14674, 17767, 596, 6222, 6415, 84011, 2024, 21455, 10877, 9459, 13595, 11883, 2683, 11838, 11538, 11912, 34018, 8816, 21644, 8170, 36926, 19626, 1922, 3415, 18627, 336, 11344, 23642, 24793, 5031, 20124, 4836, 26450, 25873, 29656, 50237, 22894, 15912, 14902, 8729, 87624, 12211, 13596, 8068, 3564, 1735, 19101, 43547, 20248, 5717, 14452, 43027, 21948, 64588, 12353, 20281, 15562, 24714, 955, 982, 21967, 75160, 36085, 2696, 78093, 6116, 39354, 9058, 14129, 16468, 2177, 10873, 46, 4490, 14192, 13805, 7705, 18575, 32392, 11284, 32256, 6287, 27936, 218, 16531, 28560, 2534, 11730, 38374, 27837, 24678, 17845, 22749, 5513, 19642, 35536, 13526, 49203, 24742, 5999, 52079, 19172], \
    train_and_predict([113314, 48070, 144301, 168231, 186415, 83735, 198933, 45833, 182557, 213454, 182067, 60822, 11895, 121506, 98262, 118005, 141117, 88945, 63555, 5582], \
                      init_training_size=20, decay_coefficient=0.5,
                      min_dist=-1, yrange=(0.05, 0.95), func_name="arctan")

    # run1()
    # run1_parallel(num_threads=os.cpu_count())
    # mytest1()


    t.tick()


