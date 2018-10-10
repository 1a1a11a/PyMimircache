# coding=utf-8

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


class ASigOPTNew(Cache):
    def __init__(self, cache_size, func_name, params, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.ts = 0

        self.cache_dict = RandomDict()
        self.func_name = func_name
        self.sigmoid_params = params


        self.n_rnd_evict_samples = kwargs.get("n_rnd_evict_samples", 64)
        self.lifetime_prob = kwargs.get("lifetime_prob", 0.9999)
        self.use_log = kwargs.get("use_log", False)

        self.log_base = kwargs.get("log_base", 1.2)

        self.last_access_time = {}


        self.hit_miss_cnt = [0, 0]
        self.opt_cmp_pos = [0]
        self.opt_cmp_pos_no_non_reuse = [0]
        self.next_access_time = kwargs.get("next_access_time", [])
        self.next_access_time_dict = {}


    def __len__(self):
        return self.get_size()


    def has(self, req_id, **kwargs):
        """

        :param req_id:
        :param kwargs:
        :return:
        """

        if req_id in self.cache_dict:
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
        self.cache_dict[req_id] = self.ts


    def _insert(self, req_item, **kwargs):
        """
        the given request is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return:
        """

        req_id = req_item
        self.cache_dict[req_id] = self.ts


    def cal_LHD_score(self, req_id):

        cur_age = self.ts - self.last_access_time.get(req_id, self.ts)
        if cur_age == 0:
            return sys.maxsize

        if req_id not in self.sigmoid_params:
            if cur_age > self.cache_size:
                return -sys.maxsize
            else:
                return sys.maxsize

        popt = self.sigmoid_params[req_id]
        
        if self.func_name == "arctan":
            b, c = popt
            if self.use_log:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) , 0)
                P_hit = 1 - (1 / (math.pi / 2) * math.atan(b * (cur_age_log + c)))
                E_lt = self.log_base ** arctan_inv(self.lifetime_prob, *popt) - cur_age
                # E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age
                ret_val = P_hit / E_lt
            else:
                P_hit = 1 - (1 / (math.pi / 2) * math.atan(b * (cur_age + c)))
                E_lt = arctan_inv(self.lifetime_prob, *popt) - cur_age
                ret_val = P_hit / E_lt
            return ret_val
        elif self.func_name == "arctan2":
            # assert self.use_log == True
            b, c, d = popt
            if cur_age == 0:
                cur_age_log = 0
            else:
                cur_age_log = max(int(math.log(cur_age, self.log_base)) , 0)

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

        elif self.func_name == "arctan3":
            # assert self.use_log == True
            b, c = popt
            if cur_age == 0:
                cur_age_log = 0
            else:
                cur_age_log = max(int(math.log(cur_age, self.log_base)) , 0)

            if self.use_log:
                P_hit = 1 - (1/(math.pi) * (math.atan(b * (cur_age_log + c)))+ 0.5)
                E_lt = (arctan_inv3(self.lifetime_prob, *popt)) - cur_age_log
                # E_lt = self.log_base ** E_lt
            else:
                P_hit = 1 - (1/(math.pi) * (math.atan(b * (cur_age + c)))+ 0.5)
                try:
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
        elif self.func_name == "tanh":
            a, b, c = popt
            # if (a, b, c) not in self.temp_set:
            #     print("{:.4g}, {:.4g}, {:.4g}".format(a, b, c))
            #     self.temp_set.add((a, b, c))

            if self.use_log:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) , 0)

                P_hit = 1 - a * math.tanh( b * (cur_age_log + c))
                E_lt = math.atanh(self.lifetime_prob / a) / b - c - cur_age_log
                # ret_val = P_hit / E_lt
                ret_val = P_hit / (self.log_base ** E_lt)
            else:
                P_hit = 1 - a * math.tanh( b * (cur_age + c))
                E_lt = math.atanh(self.lifetime_prob / a) / b - c - cur_age
                ret_val = P_hit / E_lt
            return ret_val
        elif self.func_name == "tanh2":
            a, b = popt
            if self.use_log:
                if cur_age == 0:
                    cur_age_log = 0
                else:
                    cur_age_log = max(int(math.log(cur_age, self.log_base)) , 0)
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

        min_score = -1
        chosen_key = None

        # for comparing with OPT
        max_dist = 0
        chosen_key_scores = []
        opt_chosen_key_sets = set()
        opt_key_scores = []

        for i in range(self.n_rnd_evict_samples):
            k = self.cache_dict.random_key()
            score = self.cal_LHD_score(k)
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
                    if max_dist != sys.maxsize:
                        self.opt_cmp_pos_no_non_reuse.append(pos)
                    break
        del self.cache_dict[chosen_key]


    def access(self, req_item, **kwargs):
        """
        :param kwargs:
        :param req_item: the element in the trace, it can be in the cache, or not
        :return: None
        """

        self.ts += 1

        if len(self.next_access_time):
            self.next_access_time_dict[req_item] = self.next_access_time[self.ts - 1]
            if self.next_access_time[self.ts - 1]  == -1:
                self.next_access_time_dict[req_item] = sys.maxsize


        if self.ts and self.ts % (self.cache_size * 8) == 0:
            print("ASigOPTNew ts {} func {}, prob {:.4f}, hit ratio {:.2f}, used size {} opt_pos {:.2f} ({:.2f}, {:.2f}, {:.2f}, {:.2f}) {:.2f}".format(
                self.ts,
                self.func_name,
                self.lifetime_prob,
                self.hit_miss_cnt[0] / sum(self.hit_miss_cnt),
                self.get_size(),
                sum(self.opt_cmp_pos)/len(self.opt_cmp_pos),
                sum(self.opt_cmp_pos[-200:])/len(self.opt_cmp_pos[-200:]),
                sum(self.opt_cmp_pos[-2000:])/len(self.opt_cmp_pos[-2000:]),
                sum(self.opt_cmp_pos[-20000:])/len(self.opt_cmp_pos[-20000:]),
                sum(self.opt_cmp_pos[-100000:])/len(self.opt_cmp_pos[-100000:]) if len(self.opt_cmp_pos)>100000 else 0,
                sum(self.opt_cmp_pos_no_non_reuse) / len(self.opt_cmp_pos_no_non_reuse),
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
        return len(self.cache_dict)


if __name__ == "__main__":
    from PyMimircache import CsvReader
    from PyMimircache.bin.conf import AKAMAI_CSV3
    reader = CsvReader("/home/jason/ALL_DATA/akamai3/layer/1/185.232.99.68.anon.1", init_params=AKAMAI_CSV3)
    a = ASigOPTNew(2000)
    for r in reader:
        a.access(r)