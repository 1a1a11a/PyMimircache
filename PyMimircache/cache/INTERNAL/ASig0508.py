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
from PyMimircache.utils.randomdict import RandomDict


from PyMimircache.profiler.utils.dist import *

# import matplotlib.pyplot as plt


import random
from pprint import pprint

import numpy as np
np.seterr(all='warn')
import warnings
# warnings.filterwarnings('error')

CHANGE_CDF2 = False
USE_LOG = False


# frequency upbound should be related to cache size
# expected lifetime probability should be related to cache size


# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist


class ASig0508(Cache):
    def __init__(self, cache_size, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.ts = 0

        self.cache_dict = RandomDict()

        # self.access_ts = defaultdict(list)
        self.sigmoid_params = {}

        self.freq_count = defaultdict(int)

        # for high freq
        self.hf_dict = {}
        self.possible_hf_set = set()
        self.req_curr_interval = set()
        self.last_cs_access_list = deque()
        self.hf_n = kwargs.get("hf_n", 5)

        self.evict_reasons = defaultdict(int)
        self.already_print = set()
        self.n_evictions = 0
        self.failed_fit_count = 0


        self.n_rnd_evict_samples = kwargs.get("n_rnd_evict_samples", 64)
        self.lifetime_prob = kwargs.get("lifetime_prob", 0.9999)
        self.sigmoid_func_mid = kwargs.get("sigmoid_func_mid", "arctan")
        self.sigmoid_func_low = kwargs.get("sigmoid_func_low", "arctan")
        self.use_log = kwargs.get("use_log", USE_LOG)
        # if self.sigmoid_func == "arctan2" or self.sigmoid_func == "arctan3":
        #     global USE_LOG
        #     USE_LOG = True


        self.meta_space_limit_percentage = kwargs.get("meta_space_limit_percentage", 0.2)
        self.avg_obj_size = kwargs.get("avg_obj_size", 32)

        self.log_base = kwargs.get("log_base", 1.2)
        self.min_track_ts = kwargs.get("min_track_ts", 1)
        self.max_track_ts = kwargs.get("max_tract_ts", self.cache_size*120)
        self.last_access_time = {}


        self.ts_log_start = int(math.log(self.min_track_ts, self.log_base))

        self.track_ts_count = []
        i = self.ts_log_start
        while self.log_base ** i < self.max_track_ts:
            # self.tracked_ts_count[int(self.log_base ** i)] = defaultdict(int)
            self.track_ts_count.append(defaultdict(int))
            i += 1
        print("track ts {}".format(len(self.track_ts_count)))

        self.meta_space_limit = self.cache_size * self.avg_obj_size * 1024 * self.meta_space_limit_percentage
        self.effective_size = self.cache_size # - self.meta_space_limit


        # minimal number of ts needed for fitting
        self.freq_boundary = kwargs.get("freq_boundary", (12, 800))
        self.fit_interval = kwargs.get("fit_interval", self.freq_boundary[0] * 2)


        random.seed(20)
        self.fixed_numerator = kwargs.get("fixed_numerator", -1)
        self.options = kwargs.get("options", [])
        if kwargs.get("use_fifo", False):
            self.options.append("fifo")
        if kwargs.get("mimic_arctan", False):
            self.options.append("mimic_arctan")
        if kwargs.get("mimic_arctan3", False):
            self.options.append("mimic_arctan3")
        # self.use_fifo = kwargs.get("use_fifo", False)
        # self.mimic_arctan = kwargs.get("mimic_arctan", False)
        # self.mimic_arctan3 = kwargs.get("mimic_arctan3", False)


        self.next_access_time = kwargs.get("next_access_time", [])
        self.next_access_time_dict = {}
        self.expected_dist = {}
        self.expected_dist_hm = [0, 0]
        self.hit_cnt = [0, 0]

        self.opt_cmp_pos = [0]
        self.opt_cmp_pos_no_non_reuse = [0]

        self.low_freq_ts_count = []
        for i in range(self.freq_boundary[0]):
            self.low_freq_ts_count.append([0] * int(math.ceil(math.log(self.max_track_ts, self.log_base))))

        self.temp_list = []
        self.temp_set = set()

        self.output_log = open("ASig0508_{}_{}".format(self.sigmoid_func_low, self.sigmoid_func_mid), "w")
        self.hit_miss_cnt = [0, 0]

        s = "cache size {}, effective size {} fitInterval {}, freqBoundary {}, minTrackTs {}, maxTrackTs {}, " \
            "metaSpaceLimit {}, avgObjSize {}, func {} {}, nSamples {}, prob {} use_log {}, fixed_numerator {}, options {}\n".format(
            self.cache_size, self.effective_size, self.fit_interval, self.freq_boundary, self.min_track_ts, self.max_track_ts,
            self.meta_space_limit_percentage, self.avg_obj_size, self.sigmoid_func_low, self.sigmoid_func_mid , self.n_rnd_evict_samples,
            self.lifetime_prob, self.use_log, self.fixed_numerator, self.options
        )
        print(s)
        self.output_log.write(s)


    def __len__(self):
        return self.get_size()

    def _get_fit_params(self, dist_cnt_list, sig_func_name):
        # CDF
        for i in range(1, len(dist_cnt_list)):
            dist_cnt_list[i] += dist_cnt_list[i - 1]

        # Normalization
        for i in range(0, len(dist_cnt_list)):
            dist_cnt_list[i] = dist_cnt_list[i] / dist_cnt_list[-1]


        # bridge the beginning
        pos = 0
        for i in range(len(dist_cnt_list)):
            if dist_cnt_list[i] != 0:
                pos = i
                break
        for i in range(0, pos):
            dist_cnt_list[i] = dist_cnt_list[pos]

        # fit
        pos = 0
        if self.use_log:
            xdata = [(i + self.ts_log_start) for i in range(len(dist_cnt_list))]
        else:
            xdata = [self.log_base ** (i + self.ts_log_start + pos) for i in range(len(dist_cnt_list) - pos)]
            # xdata = [self.log_base ** i for i in range(len(dist_count_list))]
        try:
            popt, sigmoid_func = sigmoid_fit(xdata, dist_cnt_list[pos:], sig_func_name)
            # if sig_func_name == "tanh":
            #     if popt[1] < 0:
            #         return None
            # elif sig_func_name == "tanh2":
            #     if popt[0] < 0:
            #         print("fitting using {}".format(dist_cnt_list))
            #         return None
            return popt, sigmoid_func
        except Exception as e:
            print("ts {} {} func {} failed to fit {} {}".format(self.ts, e, sig_func_name, dist_cnt_list, xdata))
            self.failed_fit_count += 1
            return None

    def _fit(self, req_id, sig_func_name):
        if self.freq_count[req_id] > self.freq_boundary[0] and (req_id not in self.sigmoid_params or (self.freq_count[req_id]) % self.fit_interval == 0):
            dist_cnt_list = [d.get(req_id, 0) for d in self.track_ts_count]
            r = self._get_fit_params(dist_cnt_list, sig_func_name)
            if r:
                self.sigmoid_params[req_id] = r


    def has(self, req_id, **kwargs):
        """

        :param req_id:
        :param kwargs:
        :return:
        """

        if req_id in self.cache_dict or req_id in self.hf_dict:
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

        if req_id in self.hf_dict:
            self.hf_dict[req_id] = self.ts
            return

        else:
            self.cache_dict[req_id] = self.ts
            # CHANGE A
            last_age_bin = max(int(math.log(self.ts - self.last_access_time[req_id], self.log_base)) - self.ts_log_start, 0)

            if last_age_bin >= len(self.track_ts_count):
                last_age_bin = len(self.track_ts_count)-1
            self.track_ts_count[last_age_bin][req_id] += 1

            # high time complexity
            hf = False
            if len(self.last_cs_access_list) >= self.hf_n and req_id in self.possible_hf_set:
                hf = True

            if hf:
                self.hf_dict[req_id] = True
                del self.cache_dict[req_id]
                if req_id in self.sigmoid_params:
                    del self.sigmoid_params[req_id]
                # self.freq_count[req_id] = 0
            else:
                self._fit(req_id, self.sigmoid_func_mid)


            if 2 <= self.freq_count[req_id] <= self.freq_boundary[0] + 1:
                self.low_freq_ts_count[self.freq_count[req_id] - 2][last_age_bin] += 1


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

        if req_id not in self.hf_dict:
            self.cache_dict[req_id] = self.ts

            last_age_bin = -1
            if self.freq_count[req_id] > 1:
                last_age_bin = max(int(math.log(self.ts - self.last_access_time[req_id], self.log_base))-self.ts_log_start, 0)

                if last_age_bin >= len(self.track_ts_count):
                    last_age_bin = -1
                # CHANGE A
                if not CHANGE_CDF2:
                    self.track_ts_count[last_age_bin][req_id] += 1
            if CHANGE_CDF2:
                self.track_ts_count[last_age_bin][req_id] += 1
            self._fit(req_id, self.sigmoid_func_mid)

            # this happens only when we have 2+ requests, fitting begins at freq_boundary[0]+1 requests
            if 2 <= self.freq_count[req_id] <= self.freq_boundary[0] + 1:
                self.low_freq_ts_count[self.freq_count[req_id] - 2][last_age_bin] += 1

        else:
            self.hf_dict[req_id] = self.ts

    def cal_LHD_score(self, req_id):

        if req_id not in self.last_access_time:
            assert self.cache_dict[req_id] == self.ts

        cur_age = self.ts - self.last_access_time.get(req_id, self.ts)

        if req_id in self.sigmoid_params:
            popt, func = self.sigmoid_params[req_id]
        else:

            if "fifo" in self.options:
                if cur_age > self.cache_size:
                    return -(sys.maxsize//2)
                else:
                    return sys.maxsize

            else:
                if "lowFreq{}".format(self.freq_count[req_id]) not in self.sigmoid_params:
                    print("{} low freq {} cannot find".format(req_id, self.freq_count[req_id]))
                    popt, func = self.sigmoid_params["lowFreq{}".format(self.freq_boundary[0])]
                else:
                    popt, func = self.sigmoid_params["lowFreq{}".format(self.freq_count[req_id])]


        if func.__name__ == "arctan":
            b, c = popt
            if self.use_log:
                cur_age_log = 0
                if cur_age != 0:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)

                if self.fixed_numerator == -1:
                    P_hit = 1 - (1 / (math.pi / 2) * math.atan(b * (cur_age_log + c)))
                else:             # NEW
                    P_hit = self.fixed_numerator
                E_lt = self.log_base ** arctan_inv(self.lifetime_prob, *popt) - cur_age
                # E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age_log
                ret_val = P_hit / E_lt
            else:
                if self.fixed_numerator == -1:
                    P_hit = 1 - (1 / (math.pi / 2) * math.atan(b * (cur_age + c)))
                else:             # NEW
                    P_hit = self.fixed_numerator

                if "mimic_arctan3" in self.options:
                    E_lt = arctan_inv3(self.lifetime_prob, *popt) - cur_age
                else:
                    E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age
                ret_val = P_hit / E_lt
            return ret_val
        elif func.__name__ == "arctan2":
            # assert self.use_log == True
            b, c, d = popt
            if cur_age == 0:
                cur_age_log = 0
            else:
                cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)

            if self.use_log:
                P_hit = 1 - 1/(math.pi) * (math.atan(b * (cur_age_log + c)) + d)
                E_lt = (arctan_inv2(self.lifetime_prob, *popt)) - cur_age_log
            else:
                P_hit = 1 - 1/(math.pi) * (math.atan(b * (cur_age + c)) + d)
                E_lt = self.log_base ** (arctan_inv2(self.lifetime_prob, *popt)) - cur_age

            # CHANGE C
            if E_lt < 0:
                ret_val = E_lt
            else:
                ret_val = P_hit / E_lt

            # self.output_log.write("ts {} {} {} freq {}, cur_age {} P {:.4f} E {} {:.2f} {}\n".format(
            #     self.ts, req_id[:16], req_id in self.sigmoid_params, self.freq_count[req_id], cur_age, P_hit, E_lt, ret_val, popt))

            return ret_val

        elif func.__name__ == "arctan3":
            # assert self.use_log == True
            b, c = popt
            if cur_age == 0:
                cur_age_log = 0
            else:
                cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)

            if self.use_log:
                if self.fixed_numerator == -1:
                    P_hit = 1 - (1 / (math.pi) * (math.atan(b * (cur_age_log + c))) + 0.5)
                else:             # NEW
                    P_hit = self.fixed_numerator
                E_lt = (arctan_inv3(self.lifetime_prob, *popt)) - cur_age_log
                # E_lt = self.log_base ** E_lt
            else:
                if self.fixed_numerator == -1:
                    P_hit = 1 - (1 / (math.pi) * (math.atan(b * (cur_age + c))) + 0.5)
                else:             # NEW
                    P_hit = self.fixed_numerator
                try:
                    if "mimic_arctan" in self.options:
                        E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age
                    else:
                        E_lt = arctan_inv3(self.lifetime_prob, *popt) - cur_age
                except RuntimeWarning as w:
                    print("catch {}, popt {}, curage {}, E {}".format(w, popt, cur_age, 0))
                    E_lt = np.inf
            # CHANGE C
            if E_lt < 0:
                ret_val = E_lt
            else:
                ret_val = P_hit / E_lt
            return ret_val
        elif func.__name__ == "tanh":
            a, b, c = popt
            if self.use_log:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)

                if self.fixed_numerator == -1:
                    P_hit = 1 - a * math.tanh(b * (cur_age_log + c))
                else:             # NEW
                    P_hit = self.fixed_numerator

                E_lt = math.atanh(self.lifetime_prob / a) / b - c - cur_age_log
                # ret_val = P_hit / E_lt
                ret_val = P_hit / (self.log_base ** E_lt)
            else:
                if self.fixed_numerator == -1:
                    P_hit = 1 - a * math.tanh(b * (cur_age + c))
                else:             # NEW
                    P_hit = self.fixed_numerator

                E_lt = math.atanh(self.lifetime_prob / a) / b - c - cur_age
                ret_val = P_hit / E_lt
            return ret_val
        elif func.__name__ == "tanh2":
            a, b = popt
            if (a, b) not in self.temp_set:
                # print("{:.4g}, {:.4g}".format(a, b))
                self.temp_set.add((a, b))

            if self.use_log:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)
                # np.tanh(a * x + b) / 2 + 0.5
                P_hit = 1 - (math.tanh( a * cur_age_log + b ) + 0.5)
                E_lt = (math.atanh((self.lifetime_prob - 0.5)) - b) / a - cur_age_log
                ret_val = P_hit / (self.log_base ** E_lt)
                # ret_val = P_hit / E_lt
            else:
                P_hit = 1 - (math.tanh( a * cur_age + b ) + 0.5)
                E_lt = (math.atanh((self.lifetime_prob - 0.5)) - b) / a - cur_age
                ret_val = P_hit / E_lt
            return ret_val



        else:
            raise RuntimeError("unknown func {}".format(func.__name__))


    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param kwargs:
        :return: content of evicted element
        """

        self.n_evictions += 1

        if self.n_evictions % (self.cache_size) == 1:
            for i in range(self.freq_boundary[0]):
                dist_cnt_list = self.low_freq_ts_count[i]
                # print("{} {}".format(i, dist_cnt_list))
                if sum(dist_cnt_list) == 0:
                    continue

                r = self._get_fit_params(dist_cnt_list, self.sigmoid_func_low)
                if r:
                    popt, sigmoid_func = r
                    self.sigmoid_params["lowFreq{}".format(i+1)] = (popt, sigmoid_func)
                for j in range(len(self.low_freq_ts_count[i])):
                    self.low_freq_ts_count[i][j] = self.low_freq_ts_count[i][j] // 2

        min_score = -1
        chosen_key = None

        # for comparing with OPT
        max_dist = 0
        chosen_key_scores = []
        opt_chosen_key_sets = set()
        opt_key_scores = []

        for i in range(self.n_rnd_evict_samples):
            k = self.cache_dict.random_key()

        # for k in self.cache_dict.keys:
            # if mid-freq
            score = self.cal_LHD_score(k)
            # score = ts
            # self.output_log.write("{} {} {} {:.2g} {}\n".format(self.ts, k, ts, score, k in self.sigmoid_params))
            chosen_key_scores.append((k, score * 10 ** 16))

            if score < min_score or min_score == -1:
                min_score = score
                chosen_key = k


            # comparing with OPT
            if len(self.next_access_time):
                dist = self.next_access_time_dict[k]
                opt_key_scores.append((k, dist))

                if max_dist == 0:
                    opt_chosen_key_sets.add(k)
                    opt_chosen_key_sets_one = k
                    max_dist = dist
                elif abs(dist / max_dist - 1) < 0.0001:
                    opt_chosen_key_sets.add(k)
                elif dist > max_dist:
                    max_dist = dist
                    opt_chosen_key_sets.clear()
                    opt_chosen_key_sets.add(k)
                    opt_chosen_key_sets_one = k


        if len(self.next_access_time):
            opt_key_scores.sort(reverse=True, key=lambda x:x[1])
            chosen_key_scores.sort(reverse=True, key=lambda x:x[1])
            pos = -1

            grouped_opt_key_scores = []
            last_score = "x"
            for key, score in opt_key_scores:
                if last_score == "x" or abs(last_score / score - 1) > 0.01:
                    grouped_opt_key_scores.append([key])
                    last_score = score
                else:
                    grouped_opt_key_scores[-1].append(key)

                if chosen_key == key:
                    pos = len(grouped_opt_key_scores)-1
                    self.opt_cmp_pos.append(pos)
                    if max_dist != sys.maxsize:
                        self.opt_cmp_pos_no_non_reuse.append(pos)
                    break
                # NEW 0530
                # for n, (key2, score2) in (chosen_key_scores):
                #     if key2 == key:



            # self.output_log.write("ts {} opt {}\nts {} {}\n".format(self.ts, grouped_opt_key_scores, self.ts, chosen_key_scores))

            # s = ""
            # if chosen_key in opt_chosen_key_sets:
            #     s += "YES "
            #     if max_dist == sys.maxsize:
            #         s += "Max   "
            #     else:
            #         self.hit_cnt[0] += 1
            #         s += "evict freq {} should {}".format(self.freq_count[chosen_key], set([self.freq_count[i] for i in opt_chosen_key_sets]))
            # else:
            #     s += "NO  "
            #     if max_dist == sys.maxsize:
            #         s += "Max "
            #     else:
            #         self.hit_cnt[1] += 1
            #         s += "evict freq {} should {}".format(self.freq_count[chosen_key], set([self.freq_count[i] for i in opt_chosen_key_sets]))
            # if "Max" not in s:
            #     self.output_log.write("ts {} {}\n".format(self.ts, s))


            # self.output_log.write("ASig0508 ts {} evict {} freq {} pos {}/{}\n".format(
            #     self.ts, chosen_key, self.freq_count[chosen_key], pos, len(grouped_opt_key_scores)))
            if self.ts % 2000 == 0:
                self.output_log.flush()
            # pprint(grouped_opt_key_scores)


            # chosen_key_scores.sort(reverse=False, key=lambda x:x[1])
            # print("ts {} opt    {}".format(self.ts, opt_key_scores))
            # print("ts {} chosen {}".format(self.ts, chosen_key_scores))
            # self.output_log.write("ts {} opt {}\n".format(self.ts, [i[:16] for i in opt_chosen_key_sets]))

        if chosen_key in self.sigmoid_params:
            self.evict_reasons["Sigmoid"] += 1
        else:
            self.evict_reasons["lowFreq"] += 1
        del self.cache_dict[chosen_key]
        # del self.cache_dict[opt_chosen_key_sets_one]


    def access(self, req_item, **kwargs):
        """
        :param kwargs:
        :param req_item: the element in the trace, it can be in the cache, or not
        :return: None
        """

        self.ts += 1
        self.freq_count[req_item] += 1
        self.req_curr_interval.add(req_item)

        # CHANGE A
        if self.ts and self.ts % (self.cache_size // self.hf_n) == 0:
            self.last_cs_access_list.append(self.req_curr_interval)
            self.req_curr_interval = set()
            if len(self.last_cs_access_list) > self.hf_n:
                self.last_cs_access_list.popleft()
                self.possible_hf_set = self.last_cs_access_list[0].intersection(self.last_cs_access_list[1])
                for i in range(2, self.hf_n):
                    self.possible_hf_set = self.possible_hf_set.intersection(self.last_cs_access_list[i])

                to_del = []
                for i in self.hf_dict.keys():
                    if i not in self.possible_hf_set:
                        to_del.append(i)
                for i in to_del:
                    del self.hf_dict[i]

        if len(self.next_access_time):
            self.next_access_time_dict[req_item] = self.next_access_time[self.ts - 1]
            if self.next_access_time[self.ts - 1]  == -1:
                self.next_access_time_dict[req_item] = sys.maxsize


            # else:
            #     dist = self.next_access_time[self.ts - 1] - self.ts
            #     if req_item in self.sigmoid_params:
            #         popt, func = self.sigmoid_params[req_item]
            #         v = func(dist, *popt)
            #         self.temp_list.append(v)
            #         if self.ts % 20000 == 0:
            #             print("ts {} {} {:.4f} {:.4f}".format(self.ts, req_item, v, sum(self.temp_list)/len(self.temp_list)))
            #
            #             plt.plot([sum(self.temp_list[i*2000:(i+1)*2000]) / 2000 for i in range(len(self.temp_list)//2000) ])
            #             print([sum(self.temp_list[i*2000:(i+1)*2000]) / 2000 for i in range(len(self.temp_list)//2000) ])
            #             plt.savefig("test_{}_{}.png".format(self.sigmoid_func_low, self.sigmoid_func_mid))

        # print("time {} {} will apear at time {}".format(self.ts, req_item, self.next_access_time[self.ts-1]))


        if self.ts and self.ts % (self.cache_size * 8) == 0:
            print("ASig0508 ts {} func {:8} {:8}, {}, prob {:.4f}, hit ratio {:.2f}, used size {} hf_dict {}, sigmoid {}, opt_pos {:.2f} ({:.2f}, {:.2f}, {:.2f}, {:.2f}) {:.2f}, failed count {}, {}".format(
                self.ts,
                self.sigmoid_func_low, self.sigmoid_func_mid,
                "log" if self.use_log else "no_log",
                self.lifetime_prob,
                self.hit_miss_cnt[0] / sum(self.hit_miss_cnt),
                self.get_size(), len(self.hf_dict),
                # len(self.possible_hf_set),
                len(self.sigmoid_params),
                sum(self.opt_cmp_pos)/len(self.opt_cmp_pos),
                sum(self.opt_cmp_pos[-200:])/len(self.opt_cmp_pos[-200:]),
                sum(self.opt_cmp_pos[-2000:])/len(self.opt_cmp_pos[-2000:]),
                sum(self.opt_cmp_pos[-20000:])/len(self.opt_cmp_pos[-20000:]),
                sum(self.opt_cmp_pos[-100000:])/len(self.opt_cmp_pos[-100000:]) if len(self.opt_cmp_pos)>100000 else 0,
                sum(self.opt_cmp_pos_no_non_reuse) / len(self.opt_cmp_pos_no_non_reuse),
                self.failed_fit_count,
                ["{}: {}".format(k, v) for k, v in self.evict_reasons.items()],
            ))
            # print(sorted(self.freq_count.items(), key=lambda x:x[1], reverse=True)[:20])

        if self.has(req_item, ):
            self._update(req_item, )
            self.hit_miss_cnt[0] += 1
            retval = True
        else:
            self._insert(req_item, )
            self.hit_miss_cnt[1] += 1
            # eviction needs the up-to-date ts
            self.last_access_time[req_item] = self.ts
            if self.get_size() > self.cache_size:
                self.evict()
            retval = False

        self.last_access_time[req_item] = self.ts

        return retval

    def get_size(self):
        """
        return current used cache size
        :return:
        """
        return len(self.cache_dict) + len(self.hf_dict)


if __name__ == "__main__":
    from PyMimircache import CsvReader
    from PyMimircache.bin.conf import AKAMAI_CSV3
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    a = ASig0508(2000)
    for r in reader:
        a.access(r)