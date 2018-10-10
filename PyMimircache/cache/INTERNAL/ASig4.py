# coding=utf-8

"""
    this uses priorityQueue and balance equation
"""


from collections import defaultdict
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cache.lru import LRU
from PyMimircache.cacheReader.requestItem import Req
from PyMimircache.cache.cacheLine import CacheLine
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import transform_dist_list_to_dist_count, add_one_rd_to_dist_list
from collections import OrderedDict
from heapdict import heapdict
import matplotlib.pyplot as plt

# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist

ENABLE_PRINT = False


class ASig4(Cache):
    def __init__(self, cache_size, **kwargs):

        super().__init__(cache_size, **kwargs)
        self.ts = 0
        # self.LRU_seg = OrderedDict()
        self.ASig_seg = heapdict()

        # self.cacheline_dict = OrderedDict()
        # self.cacheline_dict = {}

        self.access_ts = {}
        self.sigmoid_params = {}
        self.dist_count_list = {}

        self.enable_sigmoid_eviction = True

        # minimal number of ts needed for fitting
        self.fit_period_init = kwargs.get("fit_period_init", 12)
        self.fit_period = kwargs.get("fit_period_init", 6)

        self.high_freq_threshold = kwargs.get("high_freq_threshold", 20000)
        self.high_freq_id = {}
        self.min_dist = kwargs.get("min_dist", -1)
        self.min_age = kwargs.get("min_age", self.cache_size//10)
        self.log_base = kwargs.get("log_base", 1.08)
        self.decay_coefficient_dist_count = kwargs.get("decay_coefficient_dist_count", 0.5)
        self.decay_coefficient_dist_count_overall = kwargs.get("decay_coefficient_dist_count_overall", 0.5)
        self.decay_coefficient_hr = kwargs.get("decay_coefficient_hr", 0.95)
        self.sigmoid_func = kwargs.get("sigmoid_func", "arctan")
        # self.check_n_in_eviction = kwargs.get("check_n_in_eviction", 20)

        # self.eviction_priority = heapdict()

        # non-ASig evict, ASig evict
        # self.evict_reason = [0, 0]
        self.evict_reason = defaultdict(int)
        self.failed_fit_count = 0


        # ASig2 specific
        self.last_cal_ts = 0
        self.expected_dist_sum = sum(range(1, self.cache_size+1))
        self.expected_dist = self.cache_size

        self.current_hr = 0
        self.current_interval_hc = 0        # hit count of current interval

        self.current_dist_sum = 0
        self.false_eviction = 0
        self.recent_dist_list_dict = defaultdict(list)
        self.recent_dist_count_list_dict = defaultdict(list)
        # self.expected_dist_for_obj_should_be_evicted = [200, self.cache_size//2]
        self.prob_list = []

        self.fitting_of_freq = {}


    def __len__(self):
        return self.get_size()


    def cal_expected_dist(self):
        # if len(self.ASig_seg) < self.cache_size:
        #     return

        count = 0
        current_sum = 0
        overflow_count = 0
        # for k, v in self.ASig_seg:
        for obj in self.ASig_seg:
            v = self.ASig_seg[obj]
            if v[1] == "ASig":
                dist = v[0]
                if dist >= self.ts + self.cache_size:
                    dist = self.ts +  self.cache_size
                    overflow_count += 1
                current_sum += (dist - self.ts)
                count += 1
        # current_sum += (self.cache_size - count) * self.ts

        # expected_sum = self.expected_dist_sum + self.cache_size * self.ts
        expected_sum = self.expected_dist_sum
        sum_diff  = (expected_sum - current_sum)
        # self.expected_dist += sum_diff // self.cache_size


        # 1 + 2 + 3 + ... + d = sum_diff
        # d = int(math.sqrt(2 * sum_diff + 1/4))
        # a + (a+1) + (a+2) + ... + (a+cache_size-count-1) = sum_diff
        # (a+a+cache_size-count+1)*(cache_size - count)/2 = sum_diff

        num_non_ASig = (self.cache_size - count)
        supposed_sum_diff = (1 + num_non_ASig) * num_non_ASig // 2
        self.expected_dist += (sum_diff - supposed_sum_diff) // num_non_ASig


        # if ENABLE_PRINT:
        print("calculate {} {} ({} {}) {}".format(sum_diff - supposed_sum_diff,
                                                  (sum_diff - supposed_sum_diff) // num_non_ASig,
                                               count, overflow_count,
                                          self.expected_dist))



    def _fit(self, req_id):

        ts_list = self.access_ts[req_id]
        if len(ts_list) > self.high_freq_threshold:
            del self.sigmoid_params[req_id]
            del self.access_ts[req_id]
            # del self.eviction_priority[req_id]
            # print("del {}".format(req_id))
            self.high_freq_id[req_id] = True

        if len(ts_list) >= self.fit_period_init and len(ts_list) % self.fit_period == 1:
            dist_list = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
            if req_id not in self.dist_count_list:
                dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_dist,
                                                                    log_base=self.log_base, cdf=True, normalization=False)
                self.dist_count_list[req_id] = dist_count_list
            else:
                dist_count_list = self.dist_count_list[req_id]
                self.dist_count_list[req_id] = [i * self.decay_coefficient_dist_count for i in dist_count_list]
                for dist in dist_list:
                    add_one_rd_to_dist_list(dist, dist_count_list, 1 - self.decay_coefficient_dist_count, base=self.log_base)

            dist_count_list_normalized = [i/dist_count_list[-1] for i in dist_count_list]
            xdata = [self.log_base ** i for i in range(len(dist_count_list_normalized))]
            try:
                popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list_normalized, self.sigmoid_func)
                self.sigmoid_params[req_id] = (popt, sigmoid_func)
            except Exception as e:
                self.failed_fit_count += 1
                # print("{} {}".format(dist_count_list, e))
                pass

    def _fit_overall(self):
        if len(self.recent_dist_count_list_dict) == 0:
            for freq, recent_dist_list in self.recent_dist_list_dict.items():
                self.recent_dist_count_list_dict[freq] = transform_dist_list_to_dist_count(
                    recent_dist_list, min_dist=self.min_dist, log_base=self.log_base, cdf=True, normalization=False)
        else:
            for freq, recent_dist_count_list in self.recent_dist_count_list_dict.items():
                self.recent_dist_count_list_dict[freq] = [i * self.decay_coefficient_dist_count_overall for i in recent_dist_count_list]
                for dist in self.recent_dist_list_dict[freq]:
                    add_one_rd_to_dist_list(dist, self.recent_dist_count_list_dict[freq],
                                            1 - self.decay_coefficient_dist_count_overall, base=self.log_base)

        for freq, recent_dist_count_list in self.recent_dist_count_list_dict.items():
            dist_count_list_normalized = [i / recent_dist_count_list[-1] for i in recent_dist_count_list]

            xdata = [self.log_base ** i for i in range(len(dist_count_list_normalized))]
            try:
                popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list_normalized, self.sigmoid_func)
                if self.sigmoid_func == "arctan":
                    expected_dist = int(arctan_inv(self.current_hr, *popt) * 1.2 )
                    # if expected_dist < 2000:
                    #     print("freq {}, expected dist {}".format(freq, expected_dist))
                    #     expected_dist = 2000

                    self.fitting_of_freq[freq] = (expected_dist, popt, sigmoid_func)

                    # prob = arctan(self.cache_size, *popt)
                    # self.prob_list.append(prob)
                    # plt.plot(self.prob_list, label="Prob")
                    # plt.savefig("prob_{}.png".format(self.cache_size))
                else:
                    raise RuntimeError("other function not supported")
            except Exception as e:
                self.failed_fit_count += 1
                # print("{} {}".format(dist_count_list, e))
                pass

        self.recent_dist_list_dict.clear()


    # rewrite
    def _get_rd_prediction(self, req_id, max_num_bin=2000):
        popt, func = self.sigmoid_params[req_id]
        if func.__name__ == "arctan":
            x = int(arctan_inv(0.95, *popt))
        else:
            raise RuntimeError("unexpected func")

        return x


    def has(self, req_id, **kwargs):
        """

        :param req_id:
        :param kwargs:
        :return:
        """
        if req_id in self.ASig_seg:
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


        freq = 0
        if req_id not in self.high_freq_id:
            ts_list = list(self.access_ts.get(req_id, ()))
            ts_list.append(self.ts)
            freq = len(ts_list)
            self.access_ts[req_id] = tuple(ts_list)
            self.current_dist_sum += self.ts - self.access_ts[req_id][-2]

            self._fit(req_id)



        if req_id in self.sigmoid_params and self.enable_sigmoid_eviction:
            x = self._get_rd_prediction(req_id)
            if x > self.expected_dist * 2:
                x = self.min_age
            self.ASig_seg[req_id] = (self.ts + x, "ASig")

        elif req_id in self.high_freq_id:
            self.ASig_seg[req_id] = (self.ts + self.expected_dist, "highFreq")

        elif freq <= self.fit_period_init:
            if self.ts <= 20000:
                self.ASig_seg[req_id] = (self.ts + self.expected_dist, "freq")
            else:
                x = self.fitting_of_freq[freq][0]
                if x > self.expected_dist:
                    x = self.expected_dist // self.fit_period_init * freq
                elif x < self.min_age:
                    x = self.min_age
                self.ASig_seg[req_id] = (self.ts + x, "freq")

        else:
            print("failed freq {}, fitting period init {}".format(freq, self.fit_period_init))
            self.ASig_seg[req_id] = (self.ts + self.expected_dist, "failed")


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

        freq = 0
        if req_id not in self.high_freq_id:
            ts_list = list(self.access_ts.get(req_id, ()))
            ts_list.append(self.ts)
            freq = len(ts_list)
            self.access_ts[req_id] = tuple(ts_list)
            self._fit(req_id)

        # if req_id in self.sigmoid_params and self.enable_sigmoid_eviction:
        #     # x_min, x_max = self._get_rd_prediction(req_id)
        #     x = self._get_rd_prediction(req_id)
        #     # self.ASig_seg[req_id] = (self.ts + x_max, "ASig")
        #     self.ASig_seg[req_id] = (self.ts + x, "ASig")
        # else:
        #     self.ASig_seg[req_id] = (self.ts + self.expected_dist, "LRU")


        if req_id in self.sigmoid_params and self.enable_sigmoid_eviction:
            # x_min, x_max = self._get_rd_prediction(req_id)
            x = self._get_rd_prediction(req_id)
            # self.ASig_seg[req_id] = (self.ts + x_max, "ASig")]
            if x > self.expected_dist * 2:
                x = self.expected_dist
            self.ASig_seg[req_id] = (self.ts + x, "ASig")

        elif req_id in self.high_freq_id:
            self.ASig_seg[req_id] = (self.ts + self.expected_dist, "highFreq")

        elif freq <= self.fit_period_init:
            if self.ts <= 20000:
                self.ASig_seg[req_id] = (self.ts + self.expected_dist, "freq")
            else:
                x = self.fitting_of_freq[freq][0]
                if x > self.expected_dist:
                    x = self.expected_dist // self.fit_period_init * freq
                elif x < 2000:
                    x = 2000
                self.ASig_seg[req_id] = (self.ts + x, "freq")

        else:
            print("failed freq {}".format(freq))
            self.ASig_seg[req_id] = (self.ts + self.expected_dist, "failed")




    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: content of evicted element
        """

        false_eviction_list = []
        evict_id, (exp_time, reason) = self.ASig_seg.popitem()

        while reason == "ASig" and exp_time > self.ts:
            false_eviction_list.append((evict_id, (exp_time, reason)))
            self.false_eviction += 1
            evict_id, (exp_time, reason) = self.ASig_seg.popitem()

        for i in false_eviction_list:
            self.ASig_seg[i[0]] = i[1]

        self.evict_reason[reason] += 1

        # if reason == "LRU":
        #     self.evict_reason[0] += 1
        # else:
        #     # now check whether we should evict this one
        #     self.evict_reason[1] += 1

        # if self.ts - self.last_cal_ts > 20000 or self.false_eviction > 2000:
        #     if ENABLE_PRINT:
        #         print("need to cal {} {}".format(self.ts, exp_time), end="\t, ")
        #     self.false_eviction = 0
        #     self.cal_expected_dist()
        #     self.last_cal_ts = self.ts


        # if self.ts - self.last_cal_ts > 20000 and abs(self.ts - exp_time) > self.cache_size//10:
        #     if ENABLE_PRINT:
        #         print("need to cal {} {}".format(self.ts, exp_time), end="\t, ")
        #     self.cal_expected_dist()
        #     self.last_cal_ts = self.ts

        return evict_id


    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param req_item: the element in the trace, it can be in the cache, or not
        :return: None
        """
        if self.ts == 0:
            self.cal_expected_dist()


        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id

        if req_id in self.access_ts and len(self.access_ts[req_id]) <= self.fit_period_init:
            # self.recent_dist_list_dict.append(self.ts - self.access_ts[req_id][-1])
            self.recent_dist_list_dict[len(self.access_ts[req_id])].append(self.ts - self.access_ts[req_id][-1])

        if self.ts and self.ts % 20000 == 0:
            interval_hr = self.current_interval_hc / 20000
            if self.current_hr != 0:
                self.current_hr = self.decay_coefficient_hr * self.current_hr + (1-self.decay_coefficient_hr) * interval_hr
            else:
                self.current_hr = interval_hr
            self.current_interval_hc = 0

            self._fit_overall()
            self.cal_expected_dist()

        self.ts += 1
        if self.ts % 100000 == 0:
            print("{} availParams {} failedFitting {}, size {}, expected dist {}, "\
                  "evict_reason {}, current hr {}, param for freq {}".
                  format(self.ts, len(self.sigmoid_params), self.failed_fit_count,
                         len(self.ASig_seg), self.expected_dist,
                         list(self.evict_reason.items()), self.current_hr, [(k, v[0]) for k,v in self.fitting_of_freq.items()]))

        if self.has(req_item, ):
            self._update(req_item, )
            self.current_interval_hc +=1
            return True
        else:
            self._insert(req_item, )
            if self.get_size() > self.cache_size:
                self.evict()
            return False

    def get_size(self):
        """
        return current used cache size
        :return:
        """
        return len(self.ASig_seg)

    def __repr__(self):
        return "ASig cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_dict))
