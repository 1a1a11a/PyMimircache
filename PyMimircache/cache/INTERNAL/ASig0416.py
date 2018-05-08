# coding=utf-8


import sys
from collections import defaultdict
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cache.lru import LRU
from PyMimircache.cacheReader.requestItem import Req
from PyMimircache.cache.cacheLine import CacheLine
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import transform_dist_list_to_dist_count
from collections import OrderedDict
from heapdict import heapdict
# from randomdict import RandomDict
from PyMimircache.utils.randomdict import RandomDict
from numba import jit, double

from PyMimircache.profiler.utils.dist import *

import random


# frequency upbound should be related to cache size
# expected lifetime probability should be related to cache size


# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist


# known diff
# 1. freq_count for high-freq keeps adding in ASig0430, but not 0416
# 2. freq_count does not clear for ASig0430



class ASig0416(Cache):
    def __init__(self, cache_size, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.ts = 0

        self.cache_dict = RandomDict()
        # self.cache_dict = {}
        # for high-freq obj, pin them in memory and check regularly
        self.pin_dict = {}

        self.access_ts = {}
        self.sigmoid_params = {}
        self.evict_reasons = defaultdict(int)
        self.key_list = []
        self.key_list_accu_error = 0
        self.already_print = set()
        self.freq_count = defaultdict(int)
        self.n_evictions = 0

        # minimal number of ts needed for fitting
        self.freq_boundary = kwargs.get("freq_boundary", (12, 800))
        self.n_rnd_evict_samples = kwargs.get("n_rnd_evict_samples", 64)

        # max_remaining_dist MRD
        # expected distance ED
        # discrete distance DD
        # intergral INT
        assert "evict_type" in kwargs, "please specify evict_type"
        self.evict_type = kwargs.get("evict_type", "ED")

        self.max_rd = {}
        self.lifetime_prob = kwargs.get("lifetime_prob", 0.960)

        self.next_access_time = kwargs.get("next_access_time", [])
        self.next_access_time_dict = {}
        self.expected_dist = {}
        self.expected_dist_hm = [0, 0]
        self.low_freq_ts_update = defaultdict(int)

        self.opt_cmp_pos = [0]

        self.low_freq_ts_count = []
        for i in range(self.freq_boundary[0]):
            self.low_freq_ts_count.append(defaultdict(int))

        self.min_dist = kwargs.get("min_dist", -1)
        self.log_base = kwargs.get("log_base", 1.20)
        self.sigmoid_func = kwargs.get("sigmoid_func", "arctan")
        # self.predict_range = kwargs.get("predict_range", (0.05, 0.96))
        # self.check_n_in_eviction = kwargs.get("check_n_in_eviction", 20)

        # self.eviction_priority = heapdict()

        # non-ASig evict, ASig evict
        self.failed_fit_count = 0

        self.temp_dict = {}
        self.temp_dict_low = {}
        self.temp_dictTrue = {}
        self.temp_dictFalse = {}

        self.output_log = open("Asig0416", "w")

    def __len__(self):
        return self.get_size()

    def _fit(self, req_id):
        ts_list = self.access_ts[req_id]

        if len(ts_list) >= 2:
            rd = ts_list[-1] - ts_list[-2]
            self.max_rd[req_id] = max(self.max_rd.get(req_id, 0), rd)
        if len(ts_list) > self.freq_boundary[1]:
            try:
                del self.sigmoid_params[req_id]
            except:
                print("fail to delete {} from sigmoid_params, probably due to fitting failed".format(req_id))
            del self.access_ts[req_id]
            del self.cache_dict[req_id]
            self.pin_dict[req_id] = self.ts

        elif len(ts_list) > self.freq_boundary[0] and (len(ts_list) == self.freq_boundary[0]+1 or req_id not in self.sigmoid_params or len(ts_list) % (2 * self.freq_boundary[0]) == 1):
            dist_list = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
            dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_dist,
                                                                log_base=self.log_base)
            # print("ASig0416 {} fit {} count {}".format(self.ts, req_id, len(ts_list)))
            xdata = [self.log_base ** i for i in range(len(dist_count_list))]
            try:
                popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list, self.sigmoid_func)
                self.sigmoid_params[req_id] = (popt, sigmoid_func)
                # print("{} ASig0416 {} cnt {}, {}, {} {}".format(self.ts, req_id, len(ts_list), popt, xdata, dist_count_list))

                # t_l = [0] * 20
                # for dist in dist_list:
                #     t_l[(max(int(math.log(dist, self.log_base)), 0))] += 1
                # print("ASig0416 {} {}".format(sorted(dist_list), dist_count_list))
            except Exception as e:
                print("ASig0416 failed to fit")
                self.failed_fit_count += 1
                pass

    # # rewrite
    # def _get_rd_prediction(self, req_id, max_num_bin=2000):
    #     popt, func = self.sigmoid_params[req_id]
    #     x, x_min, x_max = -1, -1, -1
    #
    #     if func.__name__ == "arctan":
    #         x_min = arctan_inv(self.predict_range[0], *popt)
    #         x_max = arctan_inv(self.predict_range[1], *popt)
    #
    #     else:
    #         base = math.pow(self.cache_size*20, 1 / max_num_bin)
    #         for i in range(max_num_bin):
    #             lastx = x
    #             x = int(base ** i)
    #             if x == lastx:
    #                 continue
    #             y = func(x, *popt)
    #             if x_min == -1 and y > self.predict_range[0]:
    #                 x_min = x
    #             if x_max == -1 and y > self.predict_range[1]:
    #                 x_max = x
    #                 break
    #     return x_min, x_max

    def has(self, req_id, **kwargs):
        """

        :param req_id:
        :param kwargs:
        :return:
        """
        if req_id in self.cache_dict or req_id in self.pin_dict:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """
        add ts, if number of ts is larger than threshold, then fit sigmoid

        :param req_item:
        :param kwargs:
        :return:
        """

        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id

        if req_id in self.pin_dict:
            self.pin_dict[req_id] = self.ts
        else:
            self.cache_dict[req_id] = self.ts
            ts_list = self.access_ts.get(req_id, [])
            ts_list.append(self.ts)
            self.access_ts[req_id] = ts_list
            self._fit(req_id)

            # record dist for low-freq
            if 2 <= len(ts_list) < self.freq_boundary[0] + 2:
                cur_dist = ts_list[-1] - ts_list[-2]
                self.low_freq_ts_count[len(ts_list) - 1 - 1][cur_dist] += 1


    def _insert(self, req_item, **kwargs):
        """
        the given request is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return:
        """

        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id

        if req_id not in self.pin_dict:
            self.cache_dict[req_id] = self.ts
            ts_list = list(self.access_ts.get(req_id, []))
            ts_list.append(self.ts)
            self.access_ts[req_id] = ts_list
            self._fit(req_id)

            # this happens only when we have 2+ requests
            if 2 <= len(ts_list) < self.freq_boundary[0] + 2:
                cur_dist = ts_list[-1] - ts_list[-2]
                self.low_freq_ts_count[len(ts_list) - 1 - 1][cur_dist] += 1
        else:
            self.pin_dict[req_id] = self.ts


    def cal_LHD_score(self, req_id):

        cur_age = self.ts - self.cache_dict[req_id]
        if self.evict_type == "ED":
            ts_list = self.access_ts[req_id]
            if len(ts_list) <= self.freq_boundary[0]:
                if self.low_freq_ts_update.get(len(ts_list) - 1, 0) == 0 or \
                        self.ts - self.low_freq_ts_update[len(ts_list) - 1] > 20000:
                    d = self.low_freq_ts_count[len(ts_list)-1]
                    dist_count_list = []
                    dist_cdf_list = []
                    for dist, count in sorted(d.items(), key=lambda x: x[0]):
                        if len(dist_count_list) != 0:
                            dist_count_list.append((dist, count+dist_count_list[-1][1]))
                        else:
                            dist_count_list.append((dist, count))
                    for (dist, count) in dist_count_list:
                        dist_cdf_list.append((dist, count/dist_count_list[-1][1]))

                    self.low_freq_ts_update[len(ts_list) - 1] = self.ts
                    self.expected_dist["lowFreq{}".format(len(ts_list))] = dist_cdf_list


                else:
                    dist_cdf_list = self.expected_dist["lowFreq{}".format(len(ts_list))]

            else:
                if req_id in self.expected_dist and len(ts_list) - len(self.expected_dist[req_id]) <= \
                        self.freq_boundary[0]:
                    dist_cdf_list = self.expected_dist[req_id]
                else:
                    dist_list = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
                    d = defaultdict(int)
                    for dist in dist_list:
                        d[dist] += 1
                    dist_count_list = []
                    dist_cdf_list = []
                    for dist, count in sorted(d.items(), key=lambda x: x[0]):
                        if len(dist_count_list) != 0:
                            dist_count_list.append((dist, count+dist_count_list[-1][1]))
                        else:
                            dist_count_list.append((dist, count))
                    for (dist, count) in dist_count_list:
                        dist_cdf_list.append((dist, count/dist_count_list[-1][1]))

                    self.expected_dist[req_id] = dist_cdf_list
            prev_cdf = 0
            prev_dist = 0
            init_cdf = 0
            is_first = True
            ed = 0
            for dist, cdf in dist_cdf_list:
                if dist < cur_age:
                    prev_cdf = cdf
                    prev_dist = dist
                else:
                    if is_first:
                        init_cdf = (prev_cdf + cdf) / 2
                        init_dist = (prev_dist + dist) / 2
                        is_first = False
                    ed += (dist - init_dist) * (cdf - init_cdf)
            if init_cdf == 0:
                ed = self.cache_size * 2000
            else:
                ed /= (1 - init_cdf)


            # print("{} ed {} {}".format(len(dist_cdf_list), ed, dist_cdf_list))
            return -ed

        elif self.evict_type == "MRD":
            if req_id in self.sigmoid_params:
                popt, func = self.sigmoid_params[req_id]
                # self.evict_reasons["Sigmoid"] += 1
            else:
                if "lowFreq{}".format(len(self.access_ts[req_id])) not in self.sigmoid_params:
                    print("{} low freq {} cannot find".format(req_id, len(self.access_ts[req_id])))
                    popt, func = self.sigmoid_params["lowFreq{}".format(self.freq_boundary[0])]


                    # return 100000000
                else:
                    popt, func = self.sigmoid_params["lowFreq{}".format(len(self.access_ts[req_id]))]
                    # popt, func = self.sigmoid_params["lowFreq{}".format(len(self.access_ts[req_id])-1)]
                # self.evict_reasons["lowFreq"] += 1


            if func.__name__ == "arctan":
                b, c = popt
                P_hit = 1 - (1 / (math.pi / 2) * math.atan(b * (cur_age + c)))
                E_lt = 0
                E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age
                ret_val = P_hit / E_lt
                # self.output_log.write("ts {} {} {} freq {}, cur_age {} P {:.4f} E {} {:.2f} {}\n".format(
                #     self.ts, req_id[:16], req_id in self.sigmoid_params, len(self.access_ts[req_id]), cur_age, P_hit, E_lt, ret_val, popt))

                return ret_val
                # return P_hit * len(self.access_ts[req_id]) / E_lt
            else:
                raise RuntimeError("unknown func {}".format(func.__name__))


        # max_remaining_dist MRD

        # expected distance ED

        # discrete distance DD

        # intergral INT

        elif self.evict_type == "DD":
            raise RuntimeError("not supported")
        elif self.evict_type == "INT":
            raise RuntimeError("not supported")
        else:
            raise RuntimeError("unknown")

    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param kwargs:
        :return: content of evicted element
        """

        self.n_evictions += 1

        if self.evict_type == "MRD" and self.n_evictions % (self.cache_size) == 1:
            for i in range(self.freq_boundary[0]):
                dist_list = []
                to_remove = set()
                for k, v in self.low_freq_ts_count[i].items():
                    for _ in range(v):
                        dist_list.append(k)
                    if v // 2 != 0:
                        self.low_freq_ts_count[i][k] = v // 2
                    else:
                        to_remove.add(k)

                for k in to_remove:
                    del self.low_freq_ts_count[i][k]

                if not len(dist_list):
                    print("ASig0416 ts {}  lowFreq{} not ready, {}".format(self.ts, i, ", ".join([str(len(i)) for i in self.low_freq_ts_count])))
                    continue

                # dist_count_list_size = int(math.floor(math.log(max(self.low_freq_ts_count[i].keys()) + 1, self.log_base)) + 1)
                # dist_count_list_size = 120
                # dist_count_list = [0] * dist_count_list_size
                # # print("list size {} {}".format(dist_count_list_size, max(self.low_freq_ts_count[i].keys())))
                # for dist, ct in {20: 20, 12:12, 18:18, 36:36, 80:80}.items():
                # # for dist, ct in self.low_freq_ts_count[i].items():
                #     if dist != -1:
                #         dist = dist + 1
                #         if self.min_dist != -1 and dist < self.min_dist:
                #             dist = self.min_dist
                #         dist_count_list[int(math.floor(math.log(dist, self.log_base)))] += ct
                #
                # for i in range(1, len(dist_count_list)):
                #     dist_count_list[i] = dist_count_list[i - 1] + dist_count_list[i]
                #
                # last_value = dist_count_list[0]
                # for i in range(1, len(dist_count_list)):
                #     if dist_count_list[i] == 0:
                #         dist_count_list[i] = last_value
                #     else:
                #         last_value = dist_count_list[i]
                #
                # for i in range(0, len(dist_count_list)):
                #     dist_count_list[i] = dist_count_list[i] / dist_count_list[-1]
                #
                # # now change all points that are disconnected due to min_dist
                # pos = 0
                # for i in range(len(dist_count_list)):
                #     if dist_count_list[i] != 0:
                #         pos = i
                #         break
                # for i in range(0, pos):
                #     dist_count_list[i] = dist_count_list[pos]

                dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_dist,
                                                                    log_base=self.log_base)
                xdata = [self.log_base ** i for i in range(len(dist_count_list))]
                try:
                    popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list, self.sigmoid_func)
                    self.sigmoid_params["lowFreq{}".format(i + 1)] = (popt, sigmoid_func)
                except Exception as e:
                    self.failed_fit_count += 1
                    pass

        min_score = -1
        chosen_key = None

        # for comparing with OPT
        max_dist = 0
        opt_chosen_key_sets = set()
        opt_key_score = []


        for i in range(self.n_rnd_evict_samples):
            k, ts = self.cache_dict.random_item()
        # CHANGE A
        # for k, ts in self.cache_dict.items():

            # if mid-freq
            score = self.cal_LHD_score(k)
            # score = ts
            # self.output_log.write("{} {} {} {:.2g} {}\n".format(self.ts, k, ts, score, k in self.sigmoid_params))

            if score < min_score or min_score == -1:
                min_score = score
                chosen_key = k

            # comparing with OPT
            if len(self.next_access_time):

                dist = self.next_access_time_dict[k]
                opt_key_score.append((k, dist))

                if dist == max_dist:
                    opt_chosen_key_sets.add(k)
                elif dist > max_dist:
                    max_dist = dist
                    opt_chosen_key_sets.clear()
                    opt_chosen_key_sets.add(k)

                # chosen_key_freq = "low" if chosen_key not in self.sigmoid_params else "mid"
                # s = sum([i in self.sigmoid_params for i in opt_chosen_key_sets])
                # if s == 0:
                #     opt_key_freq = "low"
                # elif s == len(opt_chosen_key_sets):
                #     opt_key_freq = "mid"
                # else:
                #     opt_key_freq = "mix"
                #
                # if chosen_key not in opt_chosen_key_sets:
                #     self.temp_dict["False"] = self.temp_dict.get("False", 0) + 1
                #     self.temp_dictFalse["{}.{}".format(chosen_key_freq, opt_key_freq)] = \
                #         self.temp_dictFalse.get("{}.{}".format(chosen_key_freq, opt_key_freq), 0) + 1
                # else:
                #     self.temp_dict["True"] = self.temp_dict.get("True", 0) + 1
                #     self.temp_dictTrue["{}.{}".format(chosen_key_freq, opt_key_freq)] = \
                #         self.temp_dictTrue.get("{}.{}".format(chosen_key_freq, opt_key_freq), 0) + 1


        if len(self.next_access_time):
            opt_key_score.sort(reverse=True, key=lambda x:x[1])
            pos = -1

            grouped_opt_key_scores = []
            last_score = "x"
            for key, score in opt_key_score:
                if last_score == "x" or abs(last_score / score - 1) > 0.01:
                    grouped_opt_key_scores.append([key])
                    last_score = score
                else:
                    grouped_opt_key_scores[-1].append(key)
                if chosen_key == key:
                    pos = len(grouped_opt_key_scores)
                    self.opt_cmp_pos.append(pos)
                    break

            # pos = -1
            # for i in range(len(grouped_opt_key_scores)):
            #     for j in grouped_opt_key_scores[i]:
            #         if chosen_key == j:
            #             self.opt_cmp_pos.append(i)
            #             pos = i
            #             break


            self.output_log.write("ASig0416 ts {} evict {} freq {} pos {}/{}\n" # ", opt evicts freq {} age {}\n"
                .format(self.ts, chosen_key, len(self.access_ts[chosen_key]), pos, len(grouped_opt_key_scores),
                    set([len(self.access_ts[i]) for i in opt_chosen_key_sets]), set([self.ts - self.cache_dict[i] for i in opt_chosen_key_sets])))


        if chosen_key in self.sigmoid_params:
            self.evict_reasons["Sigmoid"] += 1
        else:
            self.evict_reasons["lowFreq"] += 1

        del self.cache_dict[chosen_key]


    def access(self, req_item, **kwargs):
        """
        :param kwargs:
        :param req_item: the element in the trace, it can be in the cache, or not
        :return: None
        """

        self.ts += 1
        self.freq_count[req_item] += 1
        if len(self.next_access_time):
            self.next_access_time_dict[req_item] = self.next_access_time[self.ts - 1]
            if self.next_access_time[self.ts - 1]  == -1:
                self.next_access_time_dict[req_item] = sys.maxsize

        if self.ts and self.ts % (self.cache_size * 20) == 0:
            to_remove = set()
            for k, v in self.pin_dict.items():
                if self.ts - v > self.cache_size * 2:
                    to_remove.add(k)
            for k in to_remove:
                del self.pin_dict[k]
            # if to_remove:
            #     print("ASig0416 {} remove {} left {}".format(self.ts, to_remove, self.pin_dict.items()))


        if self.ts and self.ts % (self.cache_size * 8) == 0:
            print("ASig0416 ts {} used size {} pin_dict {}, sigmoid {}, opt_pos {}, failed count {}, "
                  "{}, {}, True {}, False {}".format(
                self.ts, self.get_size(), len(self.pin_dict), len(self.sigmoid_params),
                sum(self.opt_cmp_pos)/len(self.opt_cmp_pos), self.failed_fit_count,
                ["{}: {}".format(k, v) for k, v in self.evict_reasons.items()],
                ["{}: {}({:.2f})".format(k, v, v / sum(self.temp_dict.values())) for k, v in self.temp_dict.items()],
                ["{}: {:.2f}".format(k, v / sum(self.temp_dictTrue.values())) for k, v in self.temp_dictTrue.items()],
                ["{}: {:.2f}".format(k, v / sum(self.temp_dictFalse.values())) for k, v in self.temp_dictFalse.items()]
            ))

        if self.has(req_item, ):
            self._update(req_item, )
            retval = True
        else:
            self._insert(req_item, )
            if self.get_size() > self.cache_size:
                self.evict()
            retval = False


        return retval

    def get_size(self):
        """
        return current used cache size
        :return:
        """
        return len(self.cache_dict) + len(self.pin_dict)


if __name__ == "__main__":
    ASig0414(2000)