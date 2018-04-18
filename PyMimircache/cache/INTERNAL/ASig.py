# coding=utf-8


from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cache.lru import LRU
from PyMimircache.cacheReader.requestItem import Req
from PyMimircache.cache.cacheLine import CacheLine
from PyMimircache.A1a1a11a.script_diary.sigmoid import *
from PyMimircache.A1a1a11a.script_diary.sigmoidUtils import transform_dist_list_to_dist_count
from collections import OrderedDict
from heapdict import heapdict


# low-freq obj either have
#   1. all small reuse distances
#   2. most small reuse distances + ocassional large reuse distance
#   3. all large reuse
# => if it is miss, then it is not valuable

# assume everyone has a small reuse dist


class ASig(Cache):
    def __init__(self, cache_size, **kwargs):

        super().__init__(cache_size, **kwargs)
        self.ts = 0
        self.LRU_seg = OrderedDict()
        self.ASig_seg = heapdict()

        # self.cacheline_dict = OrderedDict()
        # self.cacheline_dict = {}

        self.access_ts = {}
        self.sigmoid_params = {}

        self.enable_sigmoid_eviction = False

        # minimal number of ts needed for fitting
        self.fit_period = kwargs.get("fit_period", 20)
        self.high_freq_threshold = kwargs.get("high_freq_threshold", 20000)
        self.high_freq_id = {}
        self.min_dist = kwargs.get("min_dist", -1)
        self.log_base = kwargs.get("log_base", 1.20)
        self.sigmoid_func = kwargs.get("sigmoid_func", "arctan")
        self.predict_range = kwargs.get("predict_range", (0.05, 0.96))
        self.check_n_in_eviction = kwargs.get("check_n_in_eviction", 20)

        # self.eviction_priority = heapdict()

        # non-ASig evict, ASig evict
        self.evict_reason = [0, 0]
        self.failed_fit_count = 0


    def __len__(self):
        return self.get_size()


    def _fit(self, req_id):

        ts_list = self.access_ts[req_id]
        if len(ts_list) > self.high_freq_threshold:
            del self.sigmoid_params[req_id]
            del self.access_ts[req_id]
            # del self.eviction_priority[req_id]
            # print("del {}".format(req_id))
            self.high_freq_id[req_id] = True

        if len(ts_list) > self.fit_period and len(ts_list) % self.fit_period == 1:
            dist_list = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
            dist_count_list = transform_dist_list_to_dist_count(dist_list, min_dist=self.min_dist,
                                                                log_base=self.log_base)
            xdata = [self.log_base ** i for i in range(len(dist_count_list))]
            try:
                popt, sigmoid_func = sigmoid_fit(xdata, dist_count_list, self.sigmoid_func)
                self.sigmoid_params[req_id] = (popt, sigmoid_func)
            except Exception as e:
                self.failed_fit_count += 1
                # print("{} {}".format(dist_count_list, e))
                pass

    # rewrite
    def _get_rd_prediction(self, req_id, max_num_bin=2000):
        popt, func = self.sigmoid_params[req_id]
        x, x_min, x_max = -1, -1, -1

        if func.__name__ == "arctan":
            x_min = arctan_inv(self.predict_range[0], *popt)
            x_max = arctan_inv(self.predict_range[1], *popt)

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
        return x_min, x_max


    def has(self, req_id, **kwargs):
        """

        :param req_id:
        :param kwargs:
        :return:
        """
        if req_id in self.LRU_seg or req_id in self.ASig_seg:
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
            self._fit(req_id)

        if req_id in self.sigmoid_params and self.enable_sigmoid_eviction:
            x_min, x_max = self._get_rd_prediction(req_id)
            # print("predict in {} req".format(x_max))
            self.ASig_seg[req_id] = self.ts + x_max
            if req_id in self.LRU_seg:
                del self.LRU_seg[req_id]

        else:
            if req_id in self.LRU_seg:
                self.LRU_seg.move_to_end(req_id)

            # nothing need to be done for ASig, it has already been updated
            # elif req_id in self.ASig_seg:
            #     self.ASig_seg[req_id] = self.ts + x_max


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
            # print("predict in {} req".format(x_max))
            self.ASig_seg[req_id] = self.ts + x_max
        else:
            self.LRU_seg[req_id] = True


    def evict0(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: content of evicted element
        """

        min_confidence = 1
        evict_id = None
        if self.enable_sigmoid_eviction:
            if len(self.eviction_priority):
                item = self.eviction_priority.peekitem()
                if -item[1] > self.cache_size * 2:
                    evict_id = item[0]
                    print("evict due to too long")


                # check expired ones
                else:
                    for req_id, dist in self.eviction_priority.items():
                        if req_id in self.access_ts:
                            if self.access_ts[req_id][-1] + (-dist) < self.ts:
                                evict_id = req_id
                                print("evict due to expire {} {} {}".format(self.access_ts[req_id][-1], -dist, self.ts))
                                break

            if evict_id:
                print("del {} ".format(evict_id))
                del self.cacheline_dict[evict_id]
                if evict_id in self.eviction_priority:
                    del self.eviction_priority[evict_id]
            else:
                # LRU evict
                req = self.cacheline_dict.popitem(last=False)
                evict_id = req[0]
                if evict_id in self.eviction_priority:
                    del self.eviction_priority[evict_id]
                # print("LRU eviction")
                # del self.cacheline_dict[evict_id]
                return evict_id


            # test_list = []
            # for i in range(self.check_n_in_eviction):
            #     if isinstance(self.cacheline_dict, OrderedDict):
            #         req = self.cacheline_dict.popitem(last=False)
            #     else:
            #         req = self.cacheline_dict.popitem()
            #     req_id = req[0]
            #     if req_id in self.high_freq_id:
            #         test_list.append(req)
            #         continue
            #
            #     if req_id not in self.sigmoid_params:
            #         evict_id = req_id
            #         break
            #
            #     # if req has info
            #     popt, func = self.sigmoid_params[req_id]
            #     age = self.ts - self.access_ts[req_id][-1]
            #     time_to_lru = self.cache_size - age
            #     confidence = func(self.cache_size, *popt)
            #     if confidence < min_confidence:
            #         min_confidence = confidence
            #         evict_id = req_id
            #
            #     test_list.append(req)
            # for req in test_list:
            #     if req[0] != evict_id:
            #         self.cacheline_dict[req[0]] = req[1]
            # return evict_id
        else:
            if isinstance(self.cacheline_dict, OrderedDict):
                return self.cacheline_dict.popitem(last=False)[0]
            else:
                return self.cacheline_dict.popitem()[0]


    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: content of evicted element
        """

        evict_id = None
        if self.enable_sigmoid_eviction and len(self.ASig_seg):
            obj, exp_time = self.ASig_seg.peekitem()
            # if exp_time + len(self.LRU_seg) < self.ts:
            # temp fix
            while exp_time + 2000 < self.ts :
            # if exp_time + 2000 < self.ts :
                evict_id = obj
                self.ASig_seg.popitem()
                self.evict_reason[1] += 1
                obj, exp_time = self.ASig_seg.peekitem()

                # print("evict from ASig")
            # else:
            #     print("fail to evict from ASig {} {}".format(exp_time, self.ts))

        if evict_id is None:
            obj, _ = self.LRU_seg.popitem(last=False)
            evict_id = obj
            self.evict_reason[0] += 1
        return evict_id


    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param req_item: the element in the trace, it can be in the cache, or not
        :return: None
        """

        self.ts += 1
        if self.ts % 100000 == 0:
            print("{} availParams {} failedFitting {}, size {} {}, evict from {}".
                  format(self.ts, len(self.sigmoid_params), self.failed_fit_count,
                         len(self.LRU_seg), len(self.ASig_seg), self.evict_reason))
        # if self.ts == 20000000:
        if len(self.sigmoid_params) >= self.cache_size//8:
            print("enable ASig")
            self.enable_sigmoid_eviction = True

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
        return len(self.LRU_seg) + len(self.ASig_seg)

    def __repr__(self):
        return "ASig cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_dict))
