# coding=utf-8

"""
    this uses priorityQueue and balance equation
"""


from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cache.lru import LRU
from PyMimircache.cacheReader.requestItem import Req
from PyMimircache.cache.cacheLine import CacheLine
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import transform_dist_list_to_dist_count, add_one_rd_to_dist_list
from collections import OrderedDict
from heapdict import heapdict


# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist

ENABLE_PRINT = False


class ASig2(Cache):
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
        self.log_base = kwargs.get("log_base", 1.20)
        self.decay_coefficient = kwargs.get("decay_coefficient", 0.5)
        self.decay_coefficient2 = kwargs.get("decay_coefficient2", 0.8)
        self.sigmoid_func = kwargs.get("sigmoid_func", "arctan")
        self.predict_range = kwargs.get("predict_range", (0.05, 0.90))
        self.check_n_in_eviction = kwargs.get("check_n_in_eviction", 20)

        # self.eviction_priority = heapdict()

        # non-ASig evict, ASig evict
        self.evict_reason = [0, 0]
        self.failed_fit_count = 0


        # ASig2 specific
        self.last_cal_ts = 0
        self.expected_dist_sum = sum(range(self.cache_size))
        self.expected_dist = self.cache_size

        self.current_dist_sum = 0
        self.false_eviction = 0 
        self.recent_dist_list = []
        self.recent_dist_count_list = []
        self.expected_dist_for_obj_should_be_evicted = [200, self.cache_size//2]


    def __len__(self):
        return self.get_size()


    def cal_expected_dist(self):
        count = 0
        current_sum = 0
        overflow_count = 0
        # for k, v in self.ASig_seg:
        for obj in self.ASig_seg:
            v = self.ASig_seg[obj]
            if v[1] == "ASig":
                dist = v[0]
                if dist > self.ts + self.cache_size:
                    dist = self.ts +  self.cache_size
                    overflow_count += 1
                current_sum += dist
                count += 1
        current_sum += (self.cache_size - count) * self.ts

        expected_sum = self.expected_dist_sum + self.cache_size * self.ts
        sum_diff  = (expected_sum - current_sum)
        # 1 + 2 + 3 + ... + d = sum_diff
        # self.expected_dist = int(math.sqrt(2 * sum_diff + 1/4))
        # a + (a+1) + (a+2) + ... + (a+cache_size-count-1) = sum_diff
        # (a+a+cache_size-count+1)*(cache_size - count)/2 = sum_diff
        num_terms = (self.cache_size - count)
        self.expected_dist = int((sum_diff * 2 / num_terms - (num_terms + 1))/2) + num_terms

        # if ENABLE_PRINT:
        print("calculate {} {} ({}, {}) {}".format(expected_sum - self.cache_size * self.ts,
                                          current_sum - self.cache_size * self.ts,
                                               num_terms, overflow_count,
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
                self.dist_count_list[req_id] = [i * self.decay_coefficient for i in dist_count_list]
                for dist in dist_list:
                    add_one_rd_to_dist_list(dist, dist_count_list, 1 - self.decay_coefficient, base=self.log_base)

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
        if len(self.recent_dist_count_list) == 0:
            self.recent_dist_count_list = transform_dist_list_to_dist_count(self.recent_dist_list, min_dist=self.min_dist,
                                                                log_base=self.log_base, cdf=True, normalization=False)
        else:
            self.recent_dist_count_list = [i * self.decay_coefficient2 for i in self.recent_dist_count_list]
            for dist in self.recent_dist_list:
                add_one_rd_to_dist_list(dist, self.recent_dist_count_list, 1 - self.decay_coefficient2, base=self.log_base)

        dist_count_list_normalized = [i/self.recent_dist_count_list[-1] for i in self.recent_dist_count_list]
        xdata = [self.log_base ** i for i in range(len(dist_count_list_normalized))]
        try:
            popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list_normalized, self.sigmoid_func)
            if self.sigmoid_func == "arctan": 
                x_min = int(arctan_inv(self.predict_range[0], *popt))
                x_max = int(arctan_inv(self.predict_range[1], *popt))
                if x_min < 0: 
                    x_min = 0 
                if x_max < 0: 
                    x_max = 2000
                self.expected_dist_for_obj_should_be_evicted = (x_min, x_max)
            else: 
                raise RuntimeError("other function not supported")            
        except Exception as e:
            self.failed_fit_count += 1
            # print("{} {}".format(dist_count_list, e))
            pass


        self.recent_dist_list.clear() 
        print("overall fitting dist {}".format(self.expected_dist_for_obj_should_be_evicted))


    # rewrite
    def _get_rd_prediction(self, req_id, max_num_bin=2000):
        popt, func = self.sigmoid_params[req_id]
        x, x_min, x_max = -1, -1, -1

        if func.__name__ == "arctan":
            x_min = int(arctan_inv(self.predict_range[0], *popt))
            x_max = int(arctan_inv(self.predict_range[1], *popt))

        else:
            base = math.pow(self.cache_size*20, 1 / max_num_bin)
            for i in range(max_num_bin):
                lastx = x
                x = int(base ** i)
                if x == lastx:
                    continue
                y = func(x, *popt)
                if x_min == -1 and y > self.predict_range[0]:
                    x_min = x
                if x_max == -1 and y > self.predict_range[1]:
                    x_max = x
                    break

        if x_min < 0:
            x_min = 0
        if x_max < 0:
            print("max {} {}".format(x_max, popt))
            x_max = 0

        if x_max < self.expected_dist_for_obj_should_be_evicted[1]:
            x_max = self.expected_dist_for_obj_should_be_evicted[1]

        if math.sqrt(x_min * x_max) > self.expected_dist:
            # Jason: I should give expected_dist to it, THIS IS x_max 
            x_max = self.expected_dist_for_obj_should_be_evicted[1]

        # if x_max < 200:
        #     # print("update max from {} to {}".format(x_max, 2000))
        #     x_max = 200
        # if x_max > self.cache_size * 2:
            # we should give a general score, which should be obtained by fitting over all obj 
            # ts_list = self.access_ts[req_id]
            # print("too large {} {}".format(x_max, [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]))
        # if x_max > self.cache_size:
            # max_ts = max(self.access_ts[req_id])
            # if x_max > max_ts:
            #     x_max = max_ts
        # if x_max > self.cache_size * 2:
        #     x_max = self.cache_size * 2

        # print("predict {} (ts len {}) {}".format(req_id, len(self.access_ts[req_id]), (x_min, x_max)))
        return x_min, x_max


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



        if req_id not in self.high_freq_id:
            ts_list = list(self.access_ts.get(req_id, ()))
            ts_list.append(self.ts)
            self.access_ts[req_id] = tuple(ts_list)
            self.current_dist_sum += self.ts - self.access_ts[req_id][-2]

            self._fit(req_id)



        if req_id in self.sigmoid_params and self.enable_sigmoid_eviction:
            x_min, x_max = self._get_rd_prediction(req_id)
            # if x_max < self.current_dist_sum // self.ts and self.ts % 1000 < 2:
            #     print("update: force update x_max from {} to {} ({} {})".
            #           format(x_max, self.current_dist_sum//self.ts, self.current_dist_sum, self.ts))
            #     x_max = self.current_dist_sum // self.ts
            self.ASig_seg[req_id] = (self.ts + x_max, "ASig")

        else:
            self.ASig_seg[req_id] = (self.ts + self.expected_dist, "LRU")


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

        if req_id not in self.high_freq_id:
            ts_list = list(self.access_ts.get(req_id, ()))
            ts_list.append(self.ts)
            self.access_ts[req_id] = tuple(ts_list)
            self._fit(req_id)

        if req_id in self.sigmoid_params and self.enable_sigmoid_eviction:
            x_min, x_max = self._get_rd_prediction(req_id)
            # if x_max < self.current_dist_sum // self.ts:
            #     print("insert: force update x_max from {} to {} ({} {})".
            #           format(x_max, self.current_dist_sum//self.ts, self.current_dist_sum, self.ts))
            #     x_max = self.current_dist_sum // self.ts
            self.ASig_seg[req_id] = (self.ts + x_max, "ASig")
        else:
            self.ASig_seg[req_id] = (self.ts + self.expected_dist, "LRU")




    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: content of evicted element
        """

        false_eviction_list = []
        evict_id, (exp_time, reason) = self.ASig_seg.popitem()

        while reason != "LRU" and exp_time > self.ts: 
            false_eviction_list.append((evict_id, (exp_time, reason)))
            self.false_eviction += 1
            print("here")
            evict_id, (exp_time, reason) = self.ASig_seg.popitem()

        for i in false_eviction_list:
            self.ASig_seg[i[0]] = i[1] 

        if reason == "LRU":
            self.evict_reason[0] += 1
        else:
            # now check whether we should evict this one 
            self.evict_reason[1] += 1

        if self.ts - self.last_cal_ts > 20000 or self.false_eviction > 2000:
            if ENABLE_PRINT:
                print("need to cal {} {}".format(self.ts, exp_time), end="\t, ")
            self.false_eviction = 0
            self.cal_expected_dist()
            self.last_cal_ts = self.ts


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
        if req_id in self.access_ts: 
            self.recent_dist_list.append(self.ts - self.access_ts[req_id][-1])
        if self.ts and self.ts % 20000 == 0:
            self._fit_overall()

        self.ts += 1
        if self.ts % 100000 == 0:
            print("{} availParams {} failedFitting {}, size {}, evict from {}, expected dist {}, "\
                  "expected_dist_for_obj_should_be_evicted {}, false_eviction {}".
                  format(self.ts, len(self.sigmoid_params), self.failed_fit_count,
                         len(self.ASig_seg), self.evict_reason, self.expected_dist,
                         self.expected_dist_for_obj_should_be_evicted, self.false_eviction))

        if self.has(req_item, ):
            self._update(req_item, )
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
