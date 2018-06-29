# coding=utf-8


import os, sys, time
from collections import deque, defaultdict, Counter
import numpy as np
from PyMimircache.bin.conf import *
from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
import bisect
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pprint import pprint


USE_ROBUT_SCALER = False
SCALE_FEATURE = True
PATH_PREFIX = ""
if "node" in socket.gethostname():
    PATH_PREFIX = "/research/jason/DL/cphy/"
OUTPUT_FOLDER = PATH_PREFIX + "180612Features"

if USE_ROBUT_SCALER:
    from sklearn.preprocessing import robust_scale as scale
    from sklearn.preprocessing import RobustScaler as Scaler
else:
    from sklearn.preprocessing import scale as scale
    from sklearn.preprocessing import StandardScaler as Scaler


np.seterr(all="raise")



if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

class BinaryTree:
    __slots__ = ["left", "right", "value", "num_left_children", "num_right_children"]
    def __init__(self,value):
        self.left = None
        self.right = None
        self.value = value
        self.num_left_children  = 0
        self.num_right_children = 0
        # self.parent = None

    def get_left_child_value(self):
        return self.left.value if self.left else None
    
    def get_right_child_value(self):
        return self.right.value if self.right else None

    # def get_parent_value(self):
    #     return self.parent.value if self.parent else None

    def set_value(self, value):
        self.value = value
        
    def get_value(self):
        return self.value

    def insert(self, value):
        assert value != self.value
        if value > self.value:
            if self.right is None: 
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)
            self.num_right_children += 1
        else:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
            self.num_left_children += 1

    def get_min(self):
        if self.left is None:
            return self.value
        else:
            return self.left.get_min()

    def get_max(self):
        if self.right is None:
            return self.value
        else:
            return self.right.get_max()

    # def del_min(self, min):
    #     if self.left and self.left.left:
    #         self.num_left_children -= 1
    #         self.left.del_min(min)
    #     else:
    #         if self.left is None:
    #             assert self.value == min
    #             raise RuntimeError("not supported in del_min")
    #         else:
    #             assert self.left.

    def find(self, value):
        """
        if exact value can be found, then return the node, if there is no such value, then return the one larger

        :param value:
        :return:
        """
        if self.value == value:
            return self
        elif value > self.value:
            if self.right is None:
                return self
                # return self.parent
            else:
                return self.right.find(value)
        elif value < self.value:
            if self.left is None:
                return self
            else:
                return self.left.find(value)
        else:
            raise RuntimeError("{} ? {}".format(self.value, value))

    def __str__(self):
        return "value {}, num_left_children {}, num_right_children {}, left ({}), right ({})".format(
            self.value, self.num_left_children, self.num_right_children,
            self.get_left_child_value(), self.get_right_child_value()
        )

def print_tree(tree):
    if tree is not None:
        print_tree(tree.get_left_child())
        print(tree.get_node_value())
        print_tree(tree.get_right_child())

def mytest_bt():
    bt = BinaryTree(80)
    for i in [56, 30, 60, 20]:
        bt.insert(i)
    ########
    # print_tree(bt)
    print(bt.find(56))


class FeatureExtractor:
    def __init__(self, dat=None, dat_type=None, n_neighbours=20, time_interval=2000, normalization_type="standard",
                 n_past_req_list=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                                      131072, 262144, 524288, 1048576),
                 history_len=20, no_warning=False,
                 **kwargs):
        """
        remove head_cutoff=0.2, tail_cutoff=0.2 because cutoff should not been done here

        :param dat:
        :param dat_type:
        :param n_neighbours:
        :param time_interval:
        :param n_past_req_list:
        :param kwargs:
        """
        self.n_neighbours = n_neighbours
        self.time_interval = time_interval
        self.normalization_type = normalization_type
        self.n_past_req_list = n_past_req_list
        self.history_len = history_len
        self.no_warning = no_warning

        if dat:
            self.need_to_compute = self.initialize(dat, dat_type)

        # assert not os.path.exists(self.output_file_nocold), "output file exists"
        ####### label and time #######
        self.labels = None
        self.access_time = None
        self.non_cold_miss_ind = None

        ####### dist ########
        self.reuse_dist = None
        self.future_reuse_dist = None
        self.last_access_dist = None
        self.next_access_dist = None

        ####### time ########
        self.reuse_time = None
        self.future_reuse_time = None

        ####### Other ########
        self.freq_cnt = defaultdict(int)
        self.freq_list = None
        self.opt_rank = None
        self.no_future_reuse_ind = None

        ####### history #######
        self.rd_list_per_ts = None
        self.rt_list_per_ts = None
        self.rd_list_stat = None
        self.rt_list_stat = None


        ####### neighbour ########
        self.left_neighbour_freq = None
        self.left_neighbour_rd = None
        self.left_neighbour_rt = None

        ####### stream ########
        ### this should be in virtual time, real time is not robust to burst
        self.request_rate_list = None
        # self.request_rate_delta_list = []
        self.cold_miss_count_list = None
        # self.cold_miss_rate_delta_list = []


        ######## scaler ########
        self.rd_scaler = None
        self.rt_scaler = None
        self.dist_scaler = None


    def initialize(self, dat, dat_type):
        self.reader = get_reader(dat, dat_type)
        if dat_type == "cphy":
            self.time_unit_adjustment = 1e-6
        elif "akamai" in dat_type.lower():
            self.time_unit_adjustment = 1
        else:
            print("unknown dat type" + dat_type)

        self.output_file_X_nocold = "{}/{}.X.noColdMiss".format(OUTPUT_FOLDER, os.path.basename(dat))
        self.output_file_X_nonan = "{}/{}.X.noColdMiss.noNan".format(OUTPUT_FOLDER, os.path.basename(dat))
        self.output_file_Y_nocold = "{}/{}.Y.noColdMiss".format(OUTPUT_FOLDER, os.path.basename(dat))

        if not SCALE_FEATURE:
            self.output_file_X_nocold += "noScale"
            self.output_file_X_nonan += "noScale"
            self.output_file_Y_nocold += "noScale"

        if "node" in socket.gethostname() and os.path.exists(self.output_file_X_nonan):
            return False
        else:
            return True

    def extract(self, feature_types=("label", "freq", "left_neighbour_freq",
                                     "access_time", "freq_in_past_n_req",
                                     "rd", "rt", "dist", "rd_list", "rt_list",

                                     "left_neighbour_rd", "left_neighbour_rt",

                                     "rd_list_stat", "rt_list_stat", "left_neighbour_stat",

                                     ),
                prediction_types=("opt_rank", "frd", "frt", "fdist", "no_future_reuse"),
                **kwargs):
        """
        main function for extracting features and trueY

        :param feature_types:
        :param prediction_types:
        :param norm:
        :param kwargs:
        :return:
        """

        if 'dat' in kwargs and 'dat_type' in kwargs:
            self.need_to_compute = self.initialize(kwargs["dat"], kwargs["dat_type"])

        if not self.need_to_compute:
            return

        X = None
        Y = None
        X_names = []
        Y_names = []

        for t in feature_types + prediction_types:
            assert getattr(self, "get_{}".format(t), None) is not None, "{} is not supported".format(t)


        # for filtering out cold miss
        self.get_non_cold_miss_ind()
        self.kept_rec_ind = np.copy(self.non_cold_miss_ind)
        for i in range(self.history_len):
            self.kept_rec_ind[i] = False
        print("kept_rec_ind                 {}\t\t\t\t{}".format(
            np.argwhere(self.kept_rec_ind).flatten().shape,
            np.argwhere(self.kept_rec_ind).flatten(),
        ))
        # print(np.argwhere(self.kept_rec_ind).flatten()[:12])

        for feature_type in feature_types:
            if feature_type in ["label", "freq", "left_neighbour_freq", ]:
                if not kwargs.get("no_warning", False):
                    print("{} ignored".format(feature_type))
                continue

            t_start = time.time()
            feature = getattr(self, "get_{}".format(feature_type))()
            # remove all cold miss
            print("{:<30} {}".format(feature_type, feature.shape), end="\t\t\t\t")

            if feature_type not in ["rd_list_stat", "rt_list_stat"]:
            # if "stat" not in feature_type:
                # for these two types, cold miss has already been removed
                if "left_neighbour" in feature_type:
                    feature = feature[self.kept_rec_ind[self.history_len:]]
                else:
                    feature = feature[self.kept_rec_ind]

            if feature_type in ["label"]:
                X_names.append("label")

            elif feature_type in ["access_time", "freq_in_past_n_req", ]:
                if SCALE_FEATURE:
                    feature = Scaler().fit_transform(feature)
                # feature = scale(feature, axis=0)
                if feature_type == "access_time":
                    X_names.extend(("min", "hour", "day"))
                elif feature_type == "freq_in_past_n_req":
                    X_names.extend(["freq_in_past_{}".format(n) for n in self.n_past_req_list])

            elif feature_type in ["rd_list_stat", "rt_list_stat", "left_neighbour_stat"]:
                if SCALE_FEATURE:
                    feature = Scaler().fit_transform(feature)
                # feature = scale(feature, axis=0)
                if feature_type in ["rd_list_stat", "rt_list_stat"]:
                    base = feature_type[:7]
                    # mean, median, max, std
                    X_names.extend(["{}_{}".format(base, t) for t in ["mean", "median", "max", "variance"]])
                elif feature_type in ["left_neighbour_stat"]:
                    # cold_miss, mean, median, std, freq_mean, freq_median, freq_std
                    X_names.extend(["left_neighbour_{}".format(t) for t in ["cold_miss", "rd_mean", "rd_median", "rd_variance",]])

            elif feature_type in ["rd", "rt", "dist", ]:
                # TODO: need to carefully examine how to scale
                setattr(self, "{}_scaler".format(feature_type), Scaler())
                if SCALE_FEATURE:
                    feature = getattr(self, "{}_scaler".format(feature_type)).fit_transform(feature)
                X_names.extend(["scaled_" + feature_type])

            elif feature_type in ["hit_in_past_n_req", ]:
                X_names.extend(["hit_past_{}".format(n) for n in self.n_past_req_list])


            elif feature_type in ["freq", "left_neighbour_freq", ]:
                # no boundary will cause problem
                raise RuntimeError("this can be a problem, currently not supported")

            elif feature_type in ["rd_list", "rt_list"]:
                X_names.extend(["{}_{}".format(feature_type, i) for i in range(self.history_len, 0, -1)])
                assert getattr(self, "{}_scaler".format(feature_type.split("_")[0])) is not None, \
                    "please extract {} before {}".format(feature_type.split("_")[0], feature_type)
                if SCALE_FEATURE:
                    scaler = getattr(self, "{}_scaler".format(feature_type.split("_")[0]))
                    feature = FeatureExtractor.myscale(feature, scaler)


            elif feature_type in ["left_neighbour_rd", "left_neighbour_rt", ]:
                assert getattr(self, "{}_scaler".format(feature_type.split("_")[-1])) is not None, \
                    "please extract {} before {}".format(feature_type.split("_")[-1], feature_type)
                X_names.extend(["{}_{}".format(feature_type, i+1) for i in range(self.history_len)])
                if SCALE_FEATURE:
                    scaler = getattr(self, "{}_scaler".format(feature_type.split("_")[-1]))
                    feature = FeatureExtractor.myscale(feature, scaler)


            else:
                raise RuntimeError("{} unknown".format(feature_type))

            if X is None:
                X = feature
            else:
                X = np.column_stack((X, feature))
            print("{:.2f} s".format(time.time() - t_start))

        for y_type in prediction_types:
            t_start = time.time()

            Y_names.append(y_type)
            y = getattr(self, "get_{}".format(y_type))()
            print("{:<30} {}".format(y_type, y.shape), end="\t\t\t\t")

            # remove all cold miss
            y = y[self.kept_rec_ind]

            if Y is None:
                Y = y
            else:
                Y = np.column_stack((Y, y))

            print("{:.2f} s".format(time.time() - t_start))


        print("x.shape={}, y.shape={}".format(X.shape, Y.shape))
        # print(X)
        # print(Y)

        # print("X mean {}, std {}, max {}, min {}".format(np.nanmean(X, axis=0), np.nanstd(X, axis=0),
        #                                                  np.nanmax(X, axis=0), np.nanmin(X, axis=0),
        #                                                  ))
        # print("Y mean {}, std {}, max {}, min {}".format(np.nanmean(Y, axis=0), np.nanstd(Y, axis=0),
        #                                                  np.nanmax(Y, axis=0), np.nanmin(Y, axis=0),
        #                                                  ))


        np.savetxt(self.output_file_X_nocold, X, header=",".join(X_names), fmt="%s", delimiter=",")
        np.savetxt(self.output_file_Y_nocold, Y, header=",".join(Y_names), fmt="%s", delimiter=",")


        # find cols that do not have nan
        non_nan_col = np.logical_not(np.any(np.isnan(X), axis=0))
        X = X[:, non_nan_col]
        X_names2 = np.array(X_names)[non_nan_col]
        print("X without nan shape " + str(X.shape))
        np.savetxt(self.output_file_X_nonan, X, delimiter=",", header=",".join(X_names2), fmt="%s")

        # polynomial features
        X = PolynomialFeatures(2).fit_transform(X)
        print("polynomial X shape " + str(X.shape))
        np.savetxt(self.output_file_X_nonan + "poly2", X, delimiter=",")

        return X, Y

    def gen_train_val_test(self, features, head_cutoff=0.2, tail_cutoff=0.2, ratio=(6,2,2), **kwargs):
        train_file = self.output_file_nocold.replace("feature", "train")
        validation_file = self.output_file_nocold.replace("feature", "validation")
        test_file = self.output_file_nocold.replace("feature", "test")

        train_sz = int(len(features) * (1 - head_cutoff - tail_cutoff) * ratio[0] / sum(ratio))
        validation_sz = int(len(features) * (1 - head_cutoff - tail_cutoff) * ratio[1] / sum(ratio))
        test_sz = int(len(features) * (1 - head_cutoff - tail_cutoff) * ratio[2] / sum(ratio))

        output_start = int(len(features) * head_cutoff)
        with open(train_file, "w") as train_ofile:
            for i in range(output_start, output_start+train_sz):
                train_ofile.write(",".join(features[i]) + "\n")

        with open(validation_file, "w") as validation_ofile:
            for i in range(output_start+train_sz, output_start+train_sz+validation_sz):
                validation_ofile.write(",".join(features[i]) + "\n")

        with open(test_file, "w") as test_ofile:
            for i in range(output_start+train_sz+validation_sz, output_start+train_sz+validation_sz+test_sz):
                test_ofile.write(",".join(features[i]) + "\n")


    ################################ Utils ##################################
    @staticmethod
    def normalize(l, normalization_type="standard"):
        array = np.array(l)
        if normalization_type == "standard":
            return (array - np.mean(array))/np.std(array)
        elif normalization_type == "feature":
            return (array - np.min(array))/(np.max(array) - np.min(array))
        else:
            raise RuntimeError(normalization_type + "not supported")


    ################################ Utils ###############################
    @staticmethod
    def myscale(feature, scaler):
        # temporarily set nan to max_value, scale then change it back to nan
        max_value = np.nanmax(feature) * 2
        feature[np.isnan(feature)] = max_value
        max_value = scaler.transform(np.array([[max_value]]))[0, 0]
        feature = scaler.transform(feature)
        feature[feature == max_value] = np.nan
        return feature

    ########################### These require O(1) space to obtain in running cache ##########################
    ######################## current access related ######################
    def get_label(self):
        raise RuntimeError("label cannot be used until word embedding is implemented, oneHotEncoding is too expensive")
        if self.labels:
            return self.labels

        labels = []
        for req in self.reader:
            labels.append(req)

        self.labels = np.array(labels, dtype=np.str)[np.newaxis].T
        self.reader.reset()
        return self.labels

    def get_access_time(self):
        """
        the time of each access

        :return: a list of tuple (min, hour, day)
        """
        
        if self.access_time:
            return self.access_time

        record = self.reader.read_time_req()  # t: (time, request)
        init_t = record[0]
        access_time = []

        while record:
            # adjust time into unit of min
            t = (record[0] - init_t) * self.time_unit_adjustment // 60
            # transfrom in day in the week, hour in the day, min in the hour (shift from trace start)
            t_min = t % 60
            t_hour = t // 60 % 24
            t_day = t // 60 //24 % 7
            access_time.append((t_min, t_hour, t_day))
            record = self.reader.read_time_req()  # t: (time, request)

        self.access_time = np.array(access_time, dtype=np.float)
        self.reader.reset()
        return self.access_time


    ############################ Require O(M) (num of obj) space to obtain in running cache ##########################
    def get_non_cold_miss_ind(self):
        if self.non_cold_miss_ind:
            return self.non_cold_miss_ind

        s = set()
        cold_miss_ind = []
        for req in self.reader:
            cold_miss_ind.append(req not in s)
            s.add(req)

        self.non_cold_miss_ind = np.logical_not(np.array(cold_miss_ind, dtype=np.bool))
        self.reader.reset()
        return self.non_cold_miss_ind


    def get_no_future_reuse(self):
        if self.no_future_reuse_ind:
            return self.no_future_reuse_ind

        no_future_reuse_ind = []
        fdist_all = self.get_fdist()

        for fdist in fdist_all:
            if fdist == -1:
                no_future_reuse_ind.append(True)
            else:
                no_future_reuse_ind.append(False)

        self.no_future_reuse_ind = np.array(no_future_reuse_ind, dtype=np.bool)[np.newaxis].T
        self.reader.reset()
        return self.no_future_reuse_ind



    ######################## distance related #####################
    def get_frd(self):
        if self.future_reuse_dist is not None:
            return self.future_reuse_dist

        self.future_reuse_dist = CLRUProfiler(self.reader).get_future_reuse_distance()[np.newaxis].T.astype(np.float32)
        self.reader.reset()
        return self.future_reuse_dist

    def get_rd(self):
        if self.reuse_dist is not None:
            return self.reuse_dist

        self.reuse_dist = CLRUProfiler(self.reader).get_reuse_distance()[np.newaxis].T.astype(np.float32)
        self.reader.reset()
        return self.reuse_dist

    def get_dist(self):
        """
        last access dist
        :return:
        """

        if self.last_access_dist is not None:
            return self.last_access_dist

        last_access_vtime = {}
        last_access_dist  = []
        for ts, req in enumerate(self.reader):
            if req in last_access_vtime:
                last_access_dist.append(ts - last_access_vtime[req])
            else:
                last_access_dist.append(-1)
            last_access_vtime[req] = ts

        self.last_access_dist = np.array(last_access_dist, dtype=np.float32)[np.newaxis].T
        self.reader.reset()
        return self.last_access_dist

    def get_fdist(self):
        """
        future access distance, get dist (virtual time) to next access

        :return: a list of fdist, each element corresponds to one request
        """

        if self.next_access_dist is not None:
            return self.next_access_dist

        last_access_vtime = {}
        next_access_dist = [-1] * len(self.reader)
        for ts, req in enumerate(self.reader):
            if req in last_access_vtime:
                next_access_dist[last_access_vtime[req]] = ts - last_access_vtime[req]
            last_access_vtime[req] = ts

        self.next_access_dist = np.array(next_access_dist, dtype=np.int64)[np.newaxis].T
        self.reader.reset()
        return self.next_access_dist


    def get_hit_in_past_n_req(self):
        """
        for each of self.n_past_req_list, check whether current obj was accessed in the past n req 
        :return: [[], [], []... []]         len(reader) sublists * len(self.n_past_req_list)
        """ 
        
        hit_in_past_n_req = []

        dist_all = self.get_dist()
        for dist in dist_all:
            l = []
            for i in range(len(self.n_past_req_list)):
                if self.n_past_req_list[i] <= dist:
                    l.append(True)
                else:
                    l.append(False)
            hit_in_past_n_req.append(tuple(l))

        self.reader.reset()
        return np.array(hit_in_past_n_req, dtype=np.bool)


    def get_freq_in_past_n_req(self):
        """
        for each of self.n_past_req_list, check the freq of current obj in the past n req
        :return: [[], [], []... []]         len(reader) sublists * len(self.n_past_req_list)
        """

        freq_in_past_n_req = []
        freq_dicts = []

        for i in range(len(self.n_past_req_list)):
            freq_dicts.append(defaultdict(int))

        for n, req in enumerate(self.reader):
            l = []
            for d, n_past in zip(freq_dicts, self.n_past_req_list):
                d[req] += 1
                l.append(d[req])
                if n and n % n_past == 0:
                    d.clear()

            freq_in_past_n_req.append(tuple(l))

        self.reader.reset()
        return np.array(freq_in_past_n_req, dtype=np.float32)


    def get_freq_in_past_n_req_old(self):
        """
        for each of self.n_past_req_list, check the freq of current obj in the past n req
        :return: [[], [], []... []]         len(reader) sublists * len(self.n_past_req_list)
        """

        freq_in_past_n_req = []

        access_time = defaultdict(list)
        for n, req in enumerate(self.reader):
            access_time[req].append(n)
            ts_list = np.negative(np.array(access_time[req]) - n)

            l = []
            for i in range(len(self.n_past_req_list)):
                l.append(sum(ts_list < self.n_past_req_list[i]))
            freq_in_past_n_req.append(tuple(l))

        self.reader.reset()
        return np.array(freq_in_past_n_req, dtype=np.float)

    ############################# time related ###########################
    def get_frt(self):
        """
        future reuse time
        :param reader:
        :return:
        """
        if self.future_reuse_time:
           return self.future_reuse_time

        record = self.reader.read_time_req()  # t: (time, request)
        # WARNING: this can use huge amount of RAM
        ts_req_list = []
        future_reuse_time = []
        while record:
            ts_req_list.append(record)
            record = self.reader.read_time_req()  # t: (time, request)

        next_access_time = {}
        for record in reversed(ts_req_list):
            if record[1] in next_access_time:
                future_reuse_time.append(next_access_time[record[1]] - record[0])
            else:
                future_reuse_time.append(-1)
            next_access_time[record[1]] = record[0]
        future_reuse_time.reverse()

        self.future_reuse_time = np.array(future_reuse_time, dtype=np.float)[np.newaxis].T
        self.reader.reset()
        return self.future_reuse_time

    def get_rt(self):
        """
        reuse time

        :param reader:
        :return:
        """
        if self.reuse_time is not None:
            return self.reuse_time

        record = self.reader.read_time_req()  # t: (time, request)
        last_access_time = {}
        reuse_time = []
        while record:
            ts, req = record
            reuse_time.append(ts - last_access_time.get(req, ts+1))  # current time - last time seen it
            last_access_time[req] = ts
            record = self.reader.read_time_req()

        self.reuse_time = np.array(reuse_time, dtype=np.float32)[np.newaxis].T
        self.reader.reset()
        return self.reuse_time



    ############################ Other ############################
    def get_freq(self):
        """
        get frequency

        :param reader:
        :return:
        """
        if self.freq_list:
            return self.freq_list

        freq_list = []
        for req in self.reader:
            freq_list.append(self.freq_cnt.get(req, 0))
            self.freq_cnt[req] += 1

        self.freq_list = np.array(freq_list)[np.newaxis].T
        self.reader.reset()
        return self.freq_list

    def get_opt_rank(self):
        """
        if opt rank the obj given current cache state, what rank it will have

        :return:
        """

        if self.opt_rank:
            return self.opt_rank

        next_access_ts = [-1] * len(self.reader)
        last_access_vtime = {}

        for ts, req in enumerate(self.reader):
            if req in last_access_vtime:
                next_access_ts[last_access_vtime[req]] = ts
            last_access_vtime[req] = ts

        self.reader.reset()
        cache_list = []
        opt_rank = []
        cache_list_begin = 0


        for ts, next_ts in enumerate(next_access_ts):
            if cache_list_begin < len(cache_list) and cache_list[cache_list_begin] < ts:
                cache_list_begin += 1

            if next_ts == -1:
                opt_rank.append(-1)
            else:
                # in this way, we don't need to remove the expired next_ts, which is O(N) operation
                # but we will use a large amount of RAM
                pos = bisect.bisect_left(cache_list, next_ts, lo=cache_list_begin)
                if len(cache_list) and pos < len(cache_list):
                    assert cache_list[pos] != next_ts

                # TODO huge time cost, need binaryTree in C
                cache_list.insert(pos, next_ts)
                opt_rank.append(pos - cache_list_begin)

            # if ts % 2000 == 0:
            # if ts >= 113860:
            #     print("ts {}, begin {}, next_ts {}, rank {}, {}".format(ts, cache_list_begin, next_ts, opt_rank[-1], cache_list[cache_list_begin:]))

            # clean
            if cache_list_begin > 200000:
                cache_list_new = []
                for i in range(cache_list_begin, len(cache_list)):
                    cache_list_new.append(cache_list[i])
                cache_list_begin = 0
                del cache_list
                cache_list = cache_list_new

        self.opt_rank = np.array(opt_rank, dtype=np.int64)[np.newaxis].T
        return self.opt_rank



    ################## These require a larger footprint O(N) to track (num of req) ####################
    ############################ History of Obj ############################
    def get_rd_list(self):
        """
        return a list of reuse distance of the obj at current time (at most history_len)

        :param reader:
        :return:
        """
        if self.rd_list_per_ts is not None:
            return self.rd_list_per_ts

        all_rd = self.get_rd()
        rd_list_dict = {}
        rd_list_per_ts = []

        for n, req in enumerate(self.reader):
            if all_rd[n] == -1:
                assert req not in rd_list_dict, "ts {}, rd {}, obj {} has {} accesses".format(
                    n, all_rd[n], req, len(rd_list_dict[req])
                )
                rd_list_dict[req] = deque([-1] * self.history_len, maxlen=self.history_len)
            else:
                rd_list_dict[req].append(all_rd[n])
                # rd_list_dict[req].popleft()
            rd_list_per_ts.append(tuple(rd_list_dict[req]))

        self.rd_list_per_ts = np.array(rd_list_per_ts, dtype=np.float)
        self.rd_list_per_ts[self.rd_list_per_ts==-1] = np.nan

        self.reader.reset()
        return self.rd_list_per_ts

    def get_rt_list(self):
        """
        return a list of reuse time of the obj at current time, the first cold miss (-1) is omitted

        :param reader:
        :return:
        """
        if self.rt_list_per_ts is not None:
            return self.rt_list_per_ts

        all_rt = self.get_rt()
        rt_dict = {}
        rt_list_per_ts = []

        for n, req in enumerate(self.reader):
            if all_rt[n] == -1:
                assert req not in rt_dict
                # rt_dict[req] = deque()
                rt_dict[req] = deque([-1] * self.history_len, maxlen=self.history_len)
            else:
                rt_dict[req].append(all_rt[n])
                # if len(rt_dict[req]) > self.history_len:
                # rt_dict[req].popleft()

            rt_list_per_ts.append(tuple(rt_dict[req]))

        self.rt_list_per_ts = np.array(rt_list_per_ts, dtype=np.float)
        self.rt_list_per_ts[self.rt_list_per_ts == -1] = np.nan

        self.reader.reset()
        return self.rt_list_per_ts

    def get_rd_list_stat(self):
        """
        return a list of stat of reuse distance, mean, median, max, std, notice that nan needs to be excluded

        :param reader:
        :return:
        """

        self.rd_list_per_ts = self.get_rd_list()
        rd_list_per_ts = self.rd_list_per_ts[self.kept_rec_ind]
        self.rd_list_stat = np.column_stack((np.nanmean(rd_list_per_ts, axis=1), np.nanmedian(rd_list_per_ts, axis=1),
                                             np.nanmax(rd_list_per_ts, axis=1), np.nanstd(rd_list_per_ts, axis=1)))

        return self.rd_list_stat

    def get_rt_list_stat(self):
        """
        return a list of stat of reuse time, mean, median, max, std,

        :param reader:
        :return:
        """

        self.rt_list_per_ts = self.get_rt_list()

        rt_list_per_ts = self.rt_list_per_ts[self.kept_rec_ind]
        self.rt_list_stat = np.column_stack((np.nanmean(rt_list_per_ts, axis=1), np.nanmedian(rt_list_per_ts, axis=1),
                                             np.nanmax(rt_list_per_ts, axis=1), np.nanstd(rt_list_per_ts, axis=1)))

        return self.rt_list_stat


    ########################## statistics about neighbours #######################
    def get_left_neighbour_freq(self):
        """
        return the freq of self.n_neighbour left neighbours of each req
        :return: [None, None, None..., [], [], []...]
        """

        if self.left_neighbour_freq is not None:
            return self.left_neighbour_freq

        freq_list = self.get_freq()
        neighbour_freq_list = []
        for i in range(len(freq_list)):
            if i < self.n_neighbours:
                pass
                # neighbour_freq_list.append(None)
            else:
                neighbour_freq_list.append(tuple(freq_list[i-self.n_neighbours:i]))

        self.left_neighbour_freq = np.array(neighbour_freq_list, dtype=np.float)[:, :, 0]
        return self.left_neighbour_freq


    def get_left_neighbour_rd(self):
        if self.left_neighbour_rd is not None:
            return self.left_neighbour_rd

        rd_list = self.get_rd()
        neighbour_rd_list = []
        for i in range(len(rd_list)):
            if i < self.n_neighbours:
                pass
            else:
                neighbour_rd_list.append(tuple(rd_list[i - self.n_neighbours:i]))

        self.left_neighbour_rd = np.array(neighbour_rd_list, dtype=np.float)[:, :, 0]
        self.left_neighbour_rd[self.left_neighbour_rd == -1] = np.nan

        return self.left_neighbour_rd

    def get_left_neighbour_rt(self):
        if self.left_neighbour_rt is not None:
            return self.left_neighbour_rt

        rt_list = self.get_rt()
        neighbour_rt_list = []
        for i in range(len(rt_list)):
            if i < self.n_neighbours:
                pass
                # neighbour_rt_list.append(None)
            else:
                neighbour_rt_list.append(tuple(rt_list[i - self.n_neighbours:i]))

        self.left_neighbour_rt = np.array(neighbour_rt_list, dtype=np.float)[:, :, 0]
        self.left_neighbour_rt[self.left_neighbour_rt == -1] = np.nan

        return self.left_neighbour_rt


    def get_left_neighbour_stat(self):
        """
        number of cold miss, rd_mean (except cold miss), rd_median (except cold miss), rd_var (except cold_miss), freq_mean, freq_var
        :param n:
        :return:
        """

        # both neighbour_freq_list and neighbour_rd_list have already been truncated by history_len elements
        # neighbour_freq_list = self.get_left_neighbour_freq()
        neighbour_rd_list   = self.get_left_neighbour_rd()

        # assert len(neighbour_freq_list) == len(neighbour_rd_list)

        cold_miss = np.sum(np.isnan(neighbour_rd_list), axis=1)
        all_cold_miss = cold_miss == self.history_len
        neighbour_rd_list_copy = np.copy(neighbour_rd_list)
        neighbour_rd_list_copy[all_cold_miss, :] = 0

        left_neighbour_stat = np.column_stack((cold_miss,
                                              np.nanmean(neighbour_rd_list_copy, axis=1),
                                              np.nanmedian(neighbour_rd_list_copy, axis=1),
                                              np.nanstd(neighbour_rd_list_copy, axis=1),
                                              # np.mean(neighbour_freq_list, axis=1),
                                              # np.median(neighbour_freq_list, axis=1),
                                              # np.std(neighbour_freq_list, axis=1),
                                                ))


        # left_neighbour_stat = []
        #
        # for i in range(len(neighbour_freq_list)):
        #     rd_list = neighbour_rd_list[i]
        #     freq_list = neighbour_freq_list[i]
        #     cold_miss = np.sum(np.isnan(rd_list))
        #
        #     if cold_miss == self.history_len:
        #         left_neighbour_stat.append((cold_miss, 0, 0, 0,
        #                                     np.mean(freq_list),
        #                                     np.median(freq_list),
        #                                     np.std(freq_list)))
        #     else:
        #         assert np.sum(np.isnan(rd_list)) != rd_list.shape[0]
        #         left_neighbour_stat.append((cold_miss,
        #                                     np.nanmean(rd_list),
        #                                     np.nanmedian(rd_list),
        #                                     np.nanstd(rd_list),
        #                                     np.mean(freq_list),
        #                                     np.median(freq_list),
        #                                     np.std(freq_list),
        #                                     ))
        # left_neighbour_stat = np.array(left_neighbour_stat, dtype=np.float)

        return left_neighbour_stat


    ########################## statistics about stream #######################
    def get_request_rate(self):
        """
        the real time (in original unit) to have time_interval requests

        """

        record_deq = deque()
        request_rate_list = []

        # for obj before time_intervals, it does not have this stat
        for _ in range(self.time_interval-1):
            record = self.reader.read_time_req()
            record_deq.append(record)
            request_rate_list.append(None)

        record = self.reader.read_time_req()
        while record:
            request_rate_list.append(record[0] - record_deq[0][0])
            record_deq.popleft()
            record = self.reader.read_time_req()

        self.request_rate_list = np.array(request_rate_list, dtype=np.float)[np.newaxis].T
        self.reader.reset()
        return self.request_rate_list


    def get_cold_miss_count(self):
        cold_miss_count_list = []

        dist_all = self.get_dist()
        dist_deq = deque()
        cold_miss_count = 0
        for i in range(self.time_interval-1):
            if dist_all[i] == -1:
                cold_miss_count += 1
            dist_deq.append(dist_all[i])
            cold_miss_count_list.append(None)

        for i in range(self.time_interval, len(dist_all)):
            if dist_all[i] == -1:
                cold_miss_count += 1
            dist_deq.append(dist_all[i])
            cold_miss_count_list.append(cold_miss_count)
            poped_dist = dist_deq.popleft()
            if poped_dist == -1:
                cold_miss_count -= 1

        return np.array(cold_miss_count_list, dtype=np.float32)[np.newaxis].T

def helper(dat, dat_type, **kwargs):
    FeatureExtractor(dat=dat, dat_type=dat_type, **kwargs).extract(**kwargs)


def run_cphy_parallel(max_workers=os.cpu_count()):
    from PyMimircache.utils.jobRunning import run_parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed


    futures_dict = {}
    results_dict = {}
    n_finished = 0
    print("{} threads".format(48))

    with ProcessPoolExecutor(max_workers=max_workers) as ppe:
        for i in range(106, 10, -1):
            futures_dict[ppe.submit(helper, dat="w{}".format(i), dat_type="cphy", no_warning=True)] = i
        for futures in as_completed(futures_dict):
            results_dict[futures_dict[futures]] = futures.result()
            print("{} finished {}/96".format(futures_dict[futures], n_finished))


def run_akamai():
    FeatureExtractor(dat="/scratch/jason/akamai3/layer/1/185.232.99.68.anon.1", dat_type="akamai3").extract()


if __name__ == "__main__":
    import cProfile, io, pstats

    FeatureExtractor().extract(dat="small", dat_type="cphy")
    # FeatureExtractor().extract(dat="w106", dat_type="cphy")
    # FeatureExtractor().extract(dat="w92", dat_type="cphy")
    # FeatureExtractor().extract(dat="w78", dat_type="cphy")
    # FeatureExtractor().extract(dat="w60", dat_type="cphy")

    run_cphy_parallel(max_workers=4)
    sys.exit(1)


    pr = cProfile.Profile()
    pr.enable()
    FeatureExtractor().extract(dat="w106", dat_type="cphy", feature_types=("rd", "rt"), prediction_types=("opt_rank", ))
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    # FeatureExtractor().extract(dat="w106", dat_type="cphy", feature_types=("rd", "rt"), prediction_types=("opt_rank", ))

    # for i in range(106, 10, -1):
    #     print("dat " + str(i))
    #     FeatureExtractor().extract(dat="w{}".format(i), dat_type="cphy", no_warning=True)

    # run_akamai()
    # run_cphy_parallel(max_workers=4)



        # feature_types=("rd", "rt", "rd_list", "rt_list",
        #
        #                "left_neighbour_rd", "left_neighbour_rt",
        #
        #                "rd_list_stat", "rt_list_stat", "left_neighbour_stat",
        #
        #                ),
        # prediction_types=("opt_rank", "frd", "frt", "fdist", "no_future_reuse"))

    # FeatureExtractor("w92", "cphy").extract()
