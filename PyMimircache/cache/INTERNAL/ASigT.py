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

import random
from pprint import pprint

CHANGE_HF = True
CHANGE_CDF1 = True
CHANGE_CDF2 = False
CHANGE_FUNC = True


# frequency upbound should be related to cache size
# expected lifetime probability should be related to cache size


# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist


class ASigTest(Cache):
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
        self.sigmoid_func = kwargs.get("sigmoid_func", "arctan")
        if self.sigmoid_func == "arctan2" or self.sigmoid_func == "arctan3":
            global CHANGE_FUNC
            CHANGE_FUNC = True
        else:
            CHANGE_FUNC = False


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



        self.next_access_time = kwargs.get("next_access_time", [])
        self.next_access_time_dict = {}
        self.expected_dist = {}
        self.expected_dist_hm = [0, 0]

        self.opt_cmp_pos = [0]

        self.low_freq_ts_count = []
        for i in range(self.freq_boundary[0]):
            self.low_freq_ts_count.append([0] * int(math.ceil(math.log(self.max_track_ts, self.log_base))))


        self.temp_dict = {}
        self.temp_dict_low = {}
        self.temp_dictTrue = {}
        self.temp_dictFalse = {}

        self.output_log = open("ASigTest", "w")


        print("cache size {}, effective size {} fitInterval {}, freqBoundary {}, minTrackTs {}, maxTrackTs {}, "
              "metaSpaceLimit {}, avgObjSize {}, func {}, nSamples {}, prob {}".format(
            self.cache_size, self.effective_size, self.fit_interval, self.freq_boundary, self.min_track_ts, self.max_track_ts,
        self.meta_space_limit_percentage, self.avg_obj_size, self.sigmoid_func, self.n_rnd_evict_samples, self.lifetime_prob))


    def __len__(self):
        return self.get_size()

    def _get_fit_params(self, dist_cnt_list):
        # CDF
        for i in range(1, len(dist_cnt_list)):
            dist_cnt_list[i] += dist_cnt_list[i - 1]

        # Normalization
        for i in range(0, len(dist_cnt_list)):
            dist_cnt_list[i] = dist_cnt_list[i] / dist_cnt_list[-1]
            if not CHANGE_FUNC and dist_cnt_list[i] == dist_cnt_list[-1]:
                dist_cnt_list = dist_cnt_list[:i + 1]
                break

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
        if CHANGE_FUNC:
            xdata = [(i + self.ts_log_start) for i in range(len(dist_cnt_list))]
        else:
            xdata = [self.log_base ** (i + self.ts_log_start + pos) for i in range(len(dist_cnt_list) - pos)]
        # xdata = [self.log_base ** i for i in range(len(dist_count_list))]
        try:
            popt, sigmoid_func = sigmoid_fit(xdata, dist_cnt_list[pos:], self.sigmoid_func)
            # print(popt)
            return popt, sigmoid_func
        except Exception as e:
            # print(e)
            print("ts {} failed to fit {} {}".format(self.ts, dist_cnt_list, xdata))
            self.failed_fit_count += 1
            return None

    def _fit(self, req_id, last_age_bin):
        if self.freq_count[req_id] > self.freq_boundary[0] and (req_id not in self.sigmoid_params or (self.freq_count[req_id]) % self.fit_interval == 0):
            dist_cnt_list = [d.get(req_id, 0) for d in self.track_ts_count]
            r = self._get_fit_params(dist_cnt_list)
            if r:
                self.sigmoid_params[req_id] = r


            # for i in range(1, len(dist_cnt_list)):
            #     dist_cnt_list[i] += dist_cnt_list[i-1]
            # for i in range(0, len(dist_cnt_list)):
            #     dist_cnt_list[i] = dist_cnt_list[i] / dist_cnt_list[-1]
            #     if not CHANGE_FUNC and dist_cnt_list[i] == dist_cnt_list[-1]:
            #         dist_cnt_list = dist_cnt_list[:i+1]
            #         break
            #
            # pos = 0
            # for i in range(len(dist_cnt_list)):
            #     if dist_cnt_list[i] != 0:
            #         pos = i
            #         break
            # for i in range(0, pos):
            #     dist_cnt_list[i] = dist_cnt_list[pos]
            # # print(dist_cnt_list)
            #
            # pos = 0
            # if CHANGE_FUNC:
            #     xdata = [(i+self.ts_log_start) for i in range(len(dist_cnt_list))]
            # else:
            #     xdata = [self.log_base ** (i+self.ts_log_start+pos) for i in range(len(dist_cnt_list)-pos)]
            # # xdata = [self.log_base ** i for i in range(len(dist_count_list))]
            # try:
            #     popt, sigmoid_func = sigmoid_fit(xdata, dist_cnt_list[pos:], self.sigmoid_func)
            #     self.sigmoid_params[req_id] = (popt, sigmoid_func)
            # except Exception as e:
            #     # print(e)
            #     print("ts {} failed to fit {} {}".format(self.ts, dist_cnt_list, xdata))
            #     self.failed_fit_count += 1
            #     pass


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
            if CHANGE_CDF1:
                last_age_bin = max(int(math.log(self.ts - self.last_access_time[req_id], self.log_base)) - self.ts_log_start, 0)
            else:
                last_age_bin = max(int(math.floor(math.log(self.ts - self.last_access_time[req_id] + 1, self.log_base))) - self.ts_log_start, 0)

            if last_age_bin >= len(self.track_ts_count):
                last_age_bin = len(self.track_ts_count)-1
            self.track_ts_count[last_age_bin][req_id] += 1

            # high time complexity
            hf = False
            # CHANGE A
            if CHANGE_HF:
                if len(self.last_cs_access_list) >= self.hf_n and req_id in self.possible_hf_set:
                    hf = True
            else:
                if self.freq_count[req_id] > self.freq_boundary[1]:
                    hf = True
            if hf:
                self.hf_dict[req_id] = True
                del self.cache_dict[req_id]
                if req_id in self.sigmoid_params:
                    del self.sigmoid_params[req_id]
                # self.freq_count[req_id] = 0
            else:
                self._fit(req_id, last_age_bin)


            if 2 <= self.freq_count[req_id] <= self.freq_boundary[0] + 1:
                cur_dist = self.ts - self.last_access_time[req_id]
                # CHANGE B
                # self.low_freq_ts_count[self.freq_count[req_id] - 2][cur_dist] += 1
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
                # CHANGE A
                if CHANGE_CDF1:
                    last_age_bin = max(int(math.log(self.ts - self.last_access_time[req_id], self.log_base))-self.ts_log_start, 0)
                else:
                    last_age_bin = max(int(math.floor(math.log(self.ts - self.last_access_time[req_id] + 1, self.log_base))) - self.ts_log_start, 0)

                if last_age_bin >= len(self.track_ts_count):
                    last_age_bin = -1
                # CHANGE A
                if not CHANGE_CDF2:
                    self.track_ts_count[last_age_bin][req_id] += 1
            if CHANGE_CDF2:
                self.track_ts_count[last_age_bin][req_id] += 1


            self._fit(req_id, last_age_bin)

            # this happens only when we have 2+ requests, fitting begins at freq_boundary[0]+1 requests
            if 2 <= self.freq_count[req_id] <= self.freq_boundary[0] + 1:
                cur_dist = self.ts - self.last_access_time[req_id]
                # CHANGE B
                # self.low_freq_ts_count[self.freq_count[req_id] - 2][cur_dist] += 1
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
            if "lowFreq{}".format(self.freq_count[req_id]) not in self.sigmoid_params:
                print("{} low freq {} cannot find".format(req_id, self.freq_count[req_id]))
                popt, func = self.sigmoid_params["lowFreq{}".format(self.freq_boundary[0])]
            else:
                popt, func = self.sigmoid_params["lowFreq{}".format(self.freq_count[req_id])]


        if func.__name__ == "arctan":
            assert CHANGE_FUNC == False
            b, c = popt
            P_hit = 1 - (1 / (math.pi / 2) * math.atan(b * (cur_age + c)))
            E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age
            ret_val = P_hit / E_lt
            return ret_val
        elif func.__name__ == "arctan2":
            assert CHANGE_FUNC == True
            b, c, d = popt
            if CHANGE_CDF1:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)
            else:
                cur_age_log = max(int(math.floor(math.log(cur_age + 1, self.log_base))) - self.ts_log_start, 0)

            P_hit = 1 - 1/(math.pi) * (math.atan(b * (cur_age_log + c)) + d)
            # CHANGE C
            # E_lt = self.log_base ** (arctan_inv2(self.lifetime_prob, *popt)) - cur_age
            E_lt = (arctan_inv2(self.lifetime_prob, *popt)) - cur_age_log
            if E_lt < 0:
                ret_val = E_lt
            else:
                ret_val = P_hit / E_lt

            # self.output_log.write("ts {} {} {} freq {}, cur_age {} P {:.4f} E {} {:.2f} {}\n".format(
            #     self.ts, req_id[:16], req_id in self.sigmoid_params, self.freq_count[req_id], cur_age, P_hit, E_lt, ret_val, popt))

            return ret_val

        elif func.__name__ == "arctan3":
            assert CHANGE_FUNC == True
            b, c = popt
            if CHANGE_CDF1:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) - self.ts_log_start, 0)
            else:
                cur_age_log = max(int(math.floor(math.log(cur_age + 1, self.log_base))) - self.ts_log_start, 0)

            P_hit = 1 - 1/(math.pi) * (math.atan(b * (cur_age_log + c)))+ 0.5
            # CHANGE C
            # E_lt = self.log_base ** (arctan_inv3(self.lifetime_prob, *popt)) - cur_age
            E_lt = (arctan_inv3(self.lifetime_prob, *popt)) - cur_age_log
            if E_lt < 0:
                ret_val = E_lt
            else:
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

                r = self._get_fit_params(dist_cnt_list)
                if r:
                    popt, sigmoid_func = r
                    self.sigmoid_params["lowFreq{}".format(i+1)] = (popt, sigmoid_func)
                for j in range(len(self.low_freq_ts_count[i])):
                    self.low_freq_ts_count[i][j] = self.low_freq_ts_count[i][j] // 2

                # dist_list = []
                # to_remove = set()
                # for dist, cnt in self.low_freq_ts_count[i].items():
                #     for _ in range(cnt):
                #         dist_list.append(dist)
                #     if cnt // 2 != 0:
                #         self.low_freq_ts_count[i][dist] = cnt // 2
                #     else:
                #         to_remove.add(dist)
                #
                # for k in to_remove:
                #     del self.low_freq_ts_count[i][k]
                #
                # if not len(dist_list):
                #     print("ASigTest ts {} lowFreq{} not ready, {}".format(self.ts, i, ", ".join([str(len(i)) for i in self.low_freq_ts_count])))
                #     continue
                #
                # dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_track_ts,
                #                                                     log_base=self.log_base)
                # xdata = [self.log_base ** i for i in range(len(dist_count_list))]
                # try:
                #     popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list, self.sigmoid_func)
                #     # i+1 because the first item is for the second access
                #     self.sigmoid_params["lowFreq{}".format(i+1)] = (popt, sigmoid_func)
                # except Exception as e:
                #     self.failed_fit_count += 1
                #     print("ASigTest ts {} lowFreq{} fitting failed".format(self.ts, i))
                #     pass

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
                    max_dist = dist
                elif abs(dist / max_dist - 1) < 0.0001:
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
            opt_key_scores.sort(reverse=True, key=lambda x:x[1])
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
            self.output_log.write("ASigTest ts {} evict {} freq {} pos {}/{}\n".format(self.ts, chosen_key, self.freq_count[chosen_key], pos, len(grouped_opt_key_scores)))
            if self.ts % 20 == 0:
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
        if CHANGE_HF and self.ts and self.ts % (self.cache_size // self.hf_n) == 0:
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

                    # not_in = 0
                    # for j in range(self.hf_n):
                    #     if i not in self.last_cs_access_list[j]:
                    #         not_in += 1
                    # if not_in >= self.hf_n//2:
                    #     del self.hf_dict[i]


        if len(self.next_access_time):
            self.next_access_time_dict[req_item] = self.next_access_time[self.ts - 1]
            if self.next_access_time[self.ts - 1]  == -1:
                self.next_access_time_dict[req_item] = sys.maxsize

        # print("time {} {} will apear at time {}".format(self.ts, req_item, self.next_access_time[self.ts-1]))

        # CHANGE A
        if not CHANGE_HF:
            if self.ts and self.ts % (self.cache_size * 20) == 0:
                to_remove = set()
                for k, v in self.hf_dict.items():
                    if self.ts - v > self.cache_size * 2:
                        to_remove.add(k)
                for k in to_remove:
                    del self.hf_dict[k]
                # if to_remove:
                #     print("ASigTest {} remove {} left {}".format(self.ts, to_remove, self.hf_dict.items()))


        if self.ts and self.ts % (self.cache_size * 8) == 0:
            print("ASigTest ts {} used size {} hf_dict {}, sigmoid {}, opt_pos {}, failed count {}, {}, {}, True {}, False {}".format(
                self.ts, self.get_size(), len(self.hf_dict),
                # len(self.possible_hf_set),
                len(self.sigmoid_params), sum(self.opt_cmp_pos)/len(self.opt_cmp_pos),
                self.failed_fit_count,
                ["{}: {}".format(k, v) for k, v in self.evict_reasons.items()],
                ["{}: {}({:.2f})".format(k, v, v / sum(self.temp_dict.values())) for k, v in self.temp_dict.items()],
                ["{}: {:.2f}".format(k, v / sum(self.temp_dictTrue.values())) for k, v in self.temp_dictTrue.items()],
                ["{}: {:.2f}".format(k, v / sum(self.temp_dictFalse.values())) for k, v in self.temp_dictFalse.items()]
            ))
            # print(sorted(self.freq_count.items(), key=lambda x:x[1], reverse=True)[:20])

        if self.has(req_item, ):
            self._update(req_item, )
            retval = True
        else:
            self._insert(req_item, )
            # self.freq_count[req_item] += 1
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
    a = ASigTest(2000)
    for r in reader:
        a.access(r)