# coding=utf-8

"""
    use hitProbability/ExpectedLifetimeAtProbability0.88


"""

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


# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist


class ASig0414(Cache):
    def __init__(self, cache_size, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.ts = 0

        self.cache_dict = RandomDict()
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
        self.freq_boundary = kwargs.get("freq_boundary", (12, 2000))
        self.n_rnd_evict_samples = kwargs.get("n_rnd_evict_samples", 64)
        self.lifetime_prob = kwargs.get("lifetime_prob", 0.88)

        self.next_access_time = kwargs.get("next_access_time", [])
        self.next_access_time_dict = {}

        self.low_freq_ts_count = []
        for i in range(self.freq_boundary[0]):
            self.low_freq_ts_count.append(defaultdict(int))


        self.min_dist = kwargs.get("min_dist", -1)
        self.log_base = kwargs.get("log_base", 1.20)
        self.sigmoid_func = kwargs.get("sigmoid_func", "arctan")


        # non-ASig evict, ASig evict
        self.failed_fit_count = 0


        self.temp_dict = {}
        self.temp_dict_low = {}
        self.temp_dictTrue = {}
        self.temp_dictFalse = {}


    def __len__(self):
        return self.get_size()


    def _fit(self, req_id):
        ts_list = self.access_ts[req_id]

        if len(ts_list) > self.freq_boundary[1]:
            try:
                del self.sigmoid_params[req_id]
            except:
                print("fail to delete {} from sigmoid_params, probably due to fitting failed".format(req_id))
            del self.access_ts[req_id]
            del self.cache_dict[req_id]
            self.pin_dict[req_id] = self.ts

        if len(ts_list) > self.freq_boundary[0] and len(ts_list) % self.freq_boundary[0] == 1:
            dist_list = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
            dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_dist,
                                                                log_base=self.log_base)
            xdata = [self.log_base ** i for i in range(len(dist_count_list))]
            try:
                popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list, self.sigmoid_func)
                self.sigmoid_params[req_id] = (popt, sigmoid_func)
            except Exception as e:
                self.failed_fit_count += 1
                pass


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
            if 2 <= len(ts_list) < self.freq_boundary[0]+2:
                cur_dist = ts_list[-1] - ts_list[-2]
                self.low_freq_ts_count[len(ts_list)-1 -1][cur_dist] += 1


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

            if 2 <= len(ts_list) < self.freq_boundary[0]+2:
                cur_dist = ts_list[-1] - ts_list[-2]
                self.low_freq_ts_count[len(ts_list)-1 -1][cur_dist] += 1
        else:
            self.pin_dict[req_id] = self.ts

    # @jit(double(string))
    # @jit
    def cal_LHD_score(self, req_id):

        if req_id in self.sigmoid_params:
            popt, func = self.sigmoid_params[req_id]
            self.evict_reasons["Sigmoid"] += 1
        else:
            if "lowFreq{}".format(len(self.access_ts[req_id])) not in self.sigmoid_params:
                self.evict_reasons["lowFreqNotFound"] +=1
                return 100000000
            popt, func = self.sigmoid_params["lowFreq{}".format(len(self.access_ts[req_id]))]
            self.evict_reasons["lowFreq"] += 1


        cur_age = self.ts - self.cache_dict[req_id]

        if func.__name__ == "arctan":
            b, c = popt
            P_hit = 1 - (1/(math.pi/2) * math.atan(b * (cur_age + c)))
            E_lt  = arctan_inv(self.lifetime_prob, *popt) - cur_age
            return P_hit/E_lt
        else:
            raise RuntimeError("unknown func {}".format(func.__name__))


    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param kwargs:
        :return: content of evicted element
        """

        self.n_evictions += 1

        # update low-freq curve
        if self.n_evictions % (self.cache_size) == 1:
            for i in range(self.freq_boundary[0]):
                dist_list = []
                to_remove = set()
                for k, v in self.low_freq_ts_count[i].items():
                    for _ in range(v):
                        dist_list.append(k)
                    if v//2 != 0:
                        self.low_freq_ts_count[i][k] = v//2
                    else:
                        to_remove.add(k)

                for k in to_remove:
                    del self.low_freq_ts_count[i][k]

                if not len(dist_list):
                    print("{} not ready, {}".format(i, ", ".join([str(len(i)) for i in self.low_freq_ts_count])))
                    continue

                dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_dist,
                                                                    log_base=self.log_base)
                xdata = [self.log_base ** i for i in range(len(dist_count_list))]
                try:
                    popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list, self.sigmoid_func)
                    self.sigmoid_params["lowFreq{}".format(i+1)] = (popt, sigmoid_func)
                except Exception as e:
                    self.failed_fit_count += 1
                    pass




        min_score = -1
        chosen_key = None

        # for comparing with OPT
        max_dist = 0
        opt_chosen_key_sets = set()

        key_set = set()


        for i in range(self.n_rnd_evict_samples):
            k, ts = self.cache_dict.random_item()
            # if ASig
            score = self.cal_LHD_score(k)
            # if LRU
            # score = ts

            if score < min_score or min_score == -1:
                min_score= score
                chosen_key = k


            # comparing with OPT
            if len(self.next_access_time):
                key_set.add(k)
                dist = self.next_access_time_dict[k]
                if dist == -1:
                    if max_dist != -1:
                        opt_chosen_key_sets.clear()
                        max_dist = -1
                    opt_chosen_key_sets.add(k)
                elif dist == max_dist:
                    opt_chosen_key_sets.add(k)
                elif dist > max_dist:
                    opt_chosen_key_sets.clear()
                    opt_chosen_key_sets.add(k)
                else:
                    raise RuntimeError("dist {}, maxDist {}".format(dist, max_dist))

                chosen_key_freq = "low" if chosen_key not in self.sigmoid_params else "mid"
                s = sum([i in self.sigmoid_params for i in opt_chosen_key_sets])
                if s == 0:
                    opt_key_freq = "low"
                elif s == len(opt_chosen_key_sets):
                    opt_key_freq = "mid"
                else:
                    opt_key_freq = "mix"

                if chosen_key not in opt_chosen_key_sets:
                    self.temp_dict["False"] = self.temp_dict.get("False", 0) + 1
                    self.temp_dictFalse["{}.{}".format(chosen_key_freq, opt_key_freq)] = \
                        self.temp_dictFalse.get("{}.{}".format(chosen_key_freq, opt_key_freq), 0) + 1
                else:
                    self.temp_dict["True"] = self.temp_dict.get("True", 0) + 1
                    self.temp_dictTrue["{}.{}".format(chosen_key_freq, opt_key_freq)] = \
                        self.temp_dictTrue.get("{}.{}".format(chosen_key_freq, opt_key_freq), 0) + 1

        # print("{} ASig chooses {} vs {}(next access {}) from {}".format(self.ts, chosen_key, opt_chosen_key_sets, max_dist, key_set))
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
            self.next_access_time_dict[req_item] = self.next_access_time[self.ts-1]
        # print("time {} {} will apear at time {}".format(self.ts, req_item, self.next_access_time[self.ts-1]))

        if self.ts and self.ts % (self.cache_size * 20) == 0:
            to_remove = set()
            for k, v in self.pin_dict.items():
                if self.ts - v > self.cache_size * 2:
                    to_remove.add(k)
            for k in to_remove:
                del self.pin_dict[k]
        if self.ts and self.ts % (self.cache_size * 1) == 0:
            print("ts {} used size {} sigmoid {} pin_dict {}, {}, {}, True {}, False {}".format(
                self.ts, self.get_size(), len(self.sigmoid_params), len(self.pin_dict),
                ["{}: {}".format(k, v) for k, v in self.evict_reasons.items()],
                ["{}: {}({:.2f})".format(k, v, v/sum(self.temp_dict.values())) for k, v in self.temp_dict.items()],
                ["{}: {:.2f}".format(k, v/sum(self.temp_dictTrue.values())) for k, v in self.temp_dictTrue.items()],
                ["{}: {:.2f}".format(k, v/sum(self.temp_dictFalse.values())) for k, v in self.temp_dictFalse.items()]
            ))


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
        return len(self.cache_dict) + len(self.pin_dict)



if __name__ == "__main__":
    ASig0414(2000)