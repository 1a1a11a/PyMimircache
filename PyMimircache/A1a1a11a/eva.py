# coding=utf-8
"""
Jason <peter.waynechina@gmail.com>  2017/11/08

EVA implementation


Expensive in fully-associative cache N^2logN
expensive without hardware support
magic number comes from the drop pos in EVA curve
age updating complicated off 1
when there is no EVA ready, need to random evict, cannot use LRU (otherwise converge problem)
    random will cause some problem in classification, it can be trapped into local minimal, for example,
    if an element from small array gets evicted then it becomes non-reused class and gets evicted every time comes in

self.reused_class.hit_probability[0] can be 1 in classification
"""


import sys
import socket
import math
from collections import deque
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.utils.linkedList import LinkedList
from heapdict import heapdict
# from numba import jit
from functools import reduce

from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import matplotlib.pyplot as plt
from PyMimircache.utils.timer import MyTimer
from PyMimircache.cache.lru import LRU
from PyMimircache import CLRUProfiler
from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler

from PyMimircache.profiler.cGeneralProfiler import CGeneralProfiler
from PyMimircache.cacheReader.vscsiReader import VscsiReader
from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache import PlainReader


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def argsort_dict(d):
    return sorted(d.keys(), key=d.__getitem__)


INTERNAL = True
DEBUG = True
PLOT_EVA = True
EPSILON = 0.00001


if socket.gethostname() == "node":
    DEBUG = False
    

class EVA(Cache):
    class Class:
        """
        Class should be agnostic to age_scaling factor 
        """ 

        def __init__(self, name, max_age, ewma_decay=0.8, dat_name="", cache_size=-1):
            # class name
            self.name = name
            self.max_age = max_age
            # decay multiplier on history
            self.ewma_decay = ewma_decay

            self.interval_hit   = [0] * self.max_age
            self.interval_evict = [0] * self.max_age

            self.hit_probability = [0] * self.max_age
            # self.expected_lifetime = [0] * (self.max_age + 1)
            self.expected_lifetime = [0] * (self.max_age)
            self.ewma_events = [0] * self.max_age

            self.ewma_hits = [0] * self.max_age
            self.ewma_evicts = [0] * self.max_age

            self.eva = [0] * self.max_age

            # set to 0 to enable debug plot
            self.eva_plot_count = 0 if PLOT_EVA else -1

            # dat_name and cache_size are used for plotting
            self.dat_name = dat_name
            self.cache_size = cache_size


        def update(self):
            self.ewma_events = [0] * self.max_age
            for i in range(self.max_age):
                self.ewma_hits[i] *= self.ewma_decay
                self.ewma_hits[i] += (1 - self.ewma_decay) * self.interval_hit[i]
                self.ewma_evicts[i] *= self.ewma_decay
                self.ewma_evicts[i] += (1 - self.ewma_decay) * self.interval_evict[i]
                self.ewma_events[i] = self.ewma_hits[i] + self.ewma_evicts[i]






        # @jit
        def reconfigure(self, per_access_cost):
            hits_up_to_now, events_up_to_now = 0, 0

            self.eva, self.hit_probability, self.expected_lifetime = \
                [0] * self.max_age, [0] * (self.max_age), [0] * (self.max_age)
            # self.expected_lifetime = [0] * (self.max_age + 1)

            expected_lifetime_unconditioned = 0

            # first calculate hit_probability, then expected lifetime
            for i in range(self.max_age-1, -1, -1):
                # self.expected_lifetime[i] = expected_lifetime_unconditioned

                # # this can also be put below else
                # events_up_to_now += self.ewma_events[i]
                # expected_lifetime_unconditioned += events_up_to_now
                #

                # this can also be put above
                events_up_to_now += self.ewma_events[i]
                hits_up_to_now += self.ewma_hits[i]
                expected_lifetime_unconditioned += events_up_to_now

                if events_up_to_now >= EPSILON:
                    self.hit_probability[i] = hits_up_to_now / events_up_to_now
                    self.expected_lifetime[i] = expected_lifetime_unconditioned / events_up_to_now
                    self.eva[i] = (self.hit_probability[i] - per_access_cost * self.expected_lifetime[i])

                else:
                    # MAYBE NEED CHANGE TO EPSILON 
                    self.hit_probability[i] = 0
                    self.expected_lifetime[i] = 0
                    self.eva[i] = 0



            # if INTERNAL and DEBUG:
            #     print("{} line gain: {}".format(self.name, per_access_cost))
            #
            #     print(", ".join(["{:8.2f}".format(i) for i in self.hits_per_interval]))
            #     print(", ".join(["{:8.2f}".format(i) for i in self.interval_evict]))
            #     # print(", ".join(["{:8.2f}".format(i) for i in self.ewma_events]))
            #     print(", ".join(["{:8.2f}".format(i) for i in self.hit_probability]))
            #     print(", ".join(["{:8.2f}".format(i) for i in self.expected_lifetime]))
            #     print(", ".join(["{:8.2f}".format(i*100) for i in self.eva]))

            # self.plot_eva(self.dat_name, self.cache_size)
            self.reset()


        def reset(self):
            self.interval_hit   = [0] * self.max_age
            self.interval_evict = [0] * self.max_age

        # @jit
        def hit(self, age):
            # if age >= len(self.hits_per_interval):
            #     raise RuntimeError("ERROR on class hit: age {} {}".format(age, len(self.hits_per_interval)))
            self.interval_hit[age] += 1

        # @jit
        def evict(self, age):
            # print("evict at age {}".format(age))
            # if age >= len(self.interval_evict):
            #     raise RuntimeError("ERROR on class evict: age {} {}".format(age, len(self.interval_evict)))
            self.interval_evict[age] += 1


        def plot_eva(self, dat_name, cache_size=None):
            if PLOT_EVA and self.eva_plot_count >= 0:
                figname = "1127EVA_Value/{}/eva_{}_{}_{}.png".format(dat_name, self.name, cache_size, self.eva_plot_count)
                if not os.path.exists(os.path.dirname(figname)):
                    os.makedirs(os.path.dirname(figname))

                # print("plot {}".format(self.eva))
                plt.plot(self.eva, label="{}_{}_{}".format(self.name, cache_size, self.eva_plot_count))
                plt.ylabel("EVA")
                plt.xlabel("Age")
                plt.savefig(figname)
                print("eva value plot {} done".format(figname))
                plt.clf()
                self.eva_plot_count += 1


    def __init__(self, cache_size=1000, update_interval=1280, max_age=20000, ewma_decay=0.8,
                 age_scaling=100, enable_classification=False, enable_stat=True, dat_name=None):
        super().__init__(cache_size)

        self.cache_size = cache_size
        self.update_interval = update_interval
        self.max_age = max_age // age_scaling + 1
        self.ewma_decay = ewma_decay
        self.age_scaling = age_scaling
        self.enable_classification = enable_classification
        self.enable_stat = enable_stat
        self.dat_name = dat_name
        self.debugging_mode = DEBUG

        print("cache size {}, update interval {}, age_scaling {}, classification {}, decay coefficient {}, max_age {}".format(
                    self.cache_size, self.update_interval, self.age_scaling,
                    self.enable_classification, self.ewma_decay, self.max_age), end="\n")

        self.eva_sorted_age = []
        self.access_time_d = {}

        # index using scaled_age, access O(N)
        self.age_to_items_deque = deque()

        self.current_ts = 0
        self.cache_content = {}

        # this is the oldest ts of elements in the frist element_list in age_to_items_deque
        # the most complicated internal data structure is like this: 
        # a deque named age_to_items_deque, the nth element of the deque is a deque of items of scaled_age = N-n (reversed order), 
        # in short, it looks like deque(an, an-1 ... a1) where a1, a2, a3... an each is a deque containing items of scaled_age n
        # cur_scaled_age_oldest_ts is describing the oldest ts of current age deque ai, it is used to determine whether  
        # current deque is full and a new deque is needed 
        self.cur_scaled_age_oldest_ts = 1

        if self.enable_classification:
            self.classes = [self.Class("nonReused", self.max_age, self.ewma_decay, self.dat_name, self.cache_size),
                            self.Class("reused", self.max_age, self.ewma_decay, self.dat_name, self.cache_size)]
            self.non_reused_class = self.classes[0]
            self.reused_class = self.classes[1]
            self.classID = {}
            if self.enable_stat:
                self.hit_stat = [[0]*self.max_age, [0]*self.max_age]
                self.evict_stat = [[0]*self.max_age, [0]*self.max_age]
        else:
            self.classes = [self.Class("general", self.max_age, self.ewma_decay, self.dat_name, self.cache_size)]


        self.overage_item_deque = deque()
        self.overage_item_set = set()

    # @jit
    def reconfigure(self):
        # get class stat ready for calculation
        for cl in self.classes:
            cl.update()

        # calculate per_access_cost
        ewma_hits_all = sum([sum(cl.ewma_hits) for cl in self.classes])
        ewma_evicts_all = sum([sum(cl.ewma_evicts) for cl in self.classes])

        if ewma_hits_all + ewma_evicts_all != 0:
            # this can happen when update_interval is very small
            # per_access_cost = ewma_hits_all / (ewma_hits_all + ewma_evicts_all) / self.cache_size

            # change
            per_access_cost = ewma_hits_all / (ewma_hits_all + ewma_evicts_all) / self.cache_size * self.age_scaling
        else:
            raise RuntimeError("ema hits and evicts sum 0")
            per_access_cost = 0

        # with per_access_cost, reconfigure each class, calculate EVA with classification
        for cl in self.classes:
            cl.reconfigure(per_access_cost)


        if self.enable_classification:
            # if self.reused_class.hit_probability[0] == 1:
                # should not happen often
            #     print("self.reused_class.hit_probability[0] = {}, reused eva0 = {}".format(
            #                 self.reused_class.hit_probability[0], self.reused_class.eva[0]))
            #     self.reused_class.hit_probability[0] = 0.99
            # print("self.reused_class.hit_probability[0] = {}, reused eva0 = {}".format(
            #     self.reused_class.hit_probability[0], self.reused_class.eva[0]))


            # eva_reused = self.reused_class.eva[0] / (1 - self.reused_class.hit_probability[0])
            if self.non_reused_class.hit_probability[0] == 0:
                print("size {} h 0".format(self.cache_size))
            eva_non_reused = self.non_reused_class.eva[0] / (self.non_reused_class.hit_probability[0])

            if ewma_hits_all + ewma_evicts_all == 0:
                hit_ratio_overall = 0
            else:
                hit_ratio_overall = ewma_hits_all / (ewma_hits_all + ewma_evicts_all)
            for cl in self.classes:
                for i in range(self.max_age-1, -1, -1):
                    # cl.eva[i] += (cl.hit_probability[i] - hit_ratio_overall) * eva_reused
                    cl.eva[i] += (hit_ratio_overall - cl.hit_probability[i]) * eva_non_reused
            eva_combined = {}
            for cl in self.classes:
                for n, eva in enumerate(cl.eva):
                    eva_combined[(cl.name, n)] = eva
            self.eva_sorted_age = argsort_dict(eva_combined)


        else:
            self.eva_sorted_age = argsort(self.classes[0].eva)


        # print("sorted age {}".format(self.eva_sorted_age))


    def _verify(self, msg=""):
        assert len(self.age_to_items_deque) <= self.max_age
        size = sum( [len(i) for i in self.age_to_items_deque] )
        if msg == "d" or msg == "e" or msg == "f" or \
                msg == "ub" or msg == "uc":
            assert size == len(self.access_time_d) - 1, \
                "msg {}, size different by more than one {} {}".format(
                msg, size, len(self.access_time_d)
            )

        else:
            assert size == len(self.access_time_d), \
                "msg {}, size different {} {}, age list size {}".format(
                msg, size, len(self.access_time_d), len(self.age_to_items_deque)
            )

        if msg != "e":
            assert len(self.access_time_d) + len(self.overage_item_set) <= self.cache_size, \
                "msg {}, number of items larger than cache size {} {}".format(
                    msg, len(self.access_time_d) + len(self.overage_item_set), self.cache_size
                )
        else:
            assert len(self.access_time_d) + len(self.overage_item_set) -1 <= self.cache_size, \
                "msg {}, number of items larger than cache size {} {}".format(
                    msg, len(self.access_time_d) + len(self.overage_item_set), self.cache_size
                )

        for n, element_list in enumerate(self.age_to_items_deque):
            if element_list:
                for element in element_list:
                    assert element in self.access_time_d, \
                        "cache size {}, msg {}, cannot find {}".format(self.cache_size, msg, element)
                    assert self.current_ts - self.access_time_d[element] - 1 == n, \
                        "age verification failed"


    def _verify2(self, msg=""):
        if self.current_ts == 0: return
        for e, t in self.access_time_d.items():
            current_ts = self.current_ts
            # if msg == "g" or msg == "a":
            #     current_ts = current_ts - 1
            age = self._get_age(e)
            # print("age {}/{} {}/{}/{} {}: {}".format(age, len(self.age_to_items_deque),
            #                                          self.cur_scaled_age_oldest_ts,
            #                                       current_ts, t, e, self.age_to_items_deque))
            assert e in self.age_to_items_deque[age], "{}: {} not in, age {} ({}/{}) {}".\
                format(msg, e, age, current_ts, t, self.age_to_items_deque)

    def _reset(self):
        for c in self.classes:
            c.reset()

    def _get_age(self, element):
        assert element in self.access_time_d, "ts {}, {} not in access_time_d".format(self.current_ts, element)

        ts_diff = self.cur_scaled_age_oldest_ts - self.access_time_d[element]
        if ts_diff < 0: 
            ts_diff = 0 
        scaled_age = int(math.ceil(ts_diff / self.age_scaling))

        if scaled_age < 0:
            raise RuntimeError("scaled age {}".format(scaled_age))


        # if left_over_age >= len(self.age_to_items_deque[0]):
        #     scaled_age += 1
        return scaled_age

        # # prev_ts = self.timestamps[ self._get_id(element) ]
        # prev_ts = self._get_timestamp(element)
        # exact_time = self.current_ts - prev_ts
        # coarse_time = (float(exact_time) // self.age_scaling) % self.max_age
        # return coarse_time


    # @jit
    def _update_age(self, element):

        # NOTICE the time order in element list (old->new as left->right) is different
        # from the time order of age_to_items_deque (old->new as right->left)
        # this is for easy traverse the old element in eviction

        # system age begins from 1
        self.current_ts += 1

        if (self.current_ts - self.cur_scaled_age_oldest_ts) % self.age_scaling == 0:
            # last scaled_age deque is full, add new one and update cur_scaled_age_oldest_ts 
            # notice the add op is appendLeft, in other words, items with the largest age is at beginning 
            self.age_to_items_deque.appendleft(deque([element, ]))
            self.cur_scaled_age_oldest_ts = self.current_ts
        else:
            self.age_to_items_deque[0].append(element)

        if self.debugging_mode:
            assert len(self.age_to_items_deque[0]) <= self.age_scaling, \
                "error first element size {} age scaling {}".format(len(self.age_to_items_deque[0]), self.age_scaling)

        if self.enable_stat:
            pass

        # Jason: age wrap up should be avoided
        # Jason: age wrap up is impossible to avoid
        if len(self.age_to_items_deque) > self.max_age:

            # calculate how many items in age_to_items_deque 
            # sum_size = sum([len(i) for i in self.age_to_items_deque])
            # print("sum size {}".format(sum_size))

            # s = " ".join([str(len(self.age_to_items_deque[i])) for i in range(12)])
            # s2 = " ".join([str(len(self.age_to_items_deque[-i])) for i in range(12)])
            # print("wrap up, age_to_item_deque len {}, max age {}, cur_ts {}, cache size {}\n{}\n{}".format(
            #     len(self.age_to_items_deque), self.max_age, self.current_ts, self.cache_size, s, s2))

            # if this is avoided, maybe use list to replace deque is better given sequential access is O(1) 
            # raise RuntimeError("wrap up should be avoided")

            # element_max_age can be None, this is the situation when the element has been hit
            element_list_max_age = self.age_to_items_deque.pop()
            for element_max_age in element_list_max_age:
                # del self.access_time_d[element_max_age]
                if element_max_age not in self.cache_content: 
                    print("in {}".format(element_max_age in self.overage_item_set))
                    raise RuntimeError("ts {} element {} cachecontent size {} overageset size {}".format(self.current_ts, element_max_age, 
                    len(self.cache_content), len(self.overage_item_set)))
                del self.cache_content[element_max_age]
                self.overage_item_deque.append(element_max_age)
                self.overage_item_set.add(element_max_age)

        self.access_time_d[element] = self.current_ts


    # @jit
    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """

        if req_id in self.cache_content or req_id in self.overage_item_set:
            return True
        else:
            return False


    # @jit
    def _update(self, req_item, **kwargs):
        """ the given req_item is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        """

        # Jason: corner case, how wrap up is handled
        in_top_eviction  = False
        if req_item in self.overage_item_set:
            # raise RuntimeError("Age wrap up")
            in_top_eviction = True
            scaled_age = self._get_age(req_item)
            # if scaled_age == -1: 
            #     scaled_age = self.max_age - 1
        else:
            # age = (self.current_ts - self.access_time_d[req_item]) // self.age_scaling
            scaled_age = self._get_age(req_item)

        if scaled_age >= self.max_age:
            raise RuntimeError("cache_size {}, scaled age larger than max age, {}/{}".\
                               format(self.cache_size, scaled_age, self.max_age))

        # Jason: corner case if req_item in top_eviction, what is hit age?
        if self.enable_classification:
            self.reused_class.hit(scaled_age)
        else:
            self.classes[0].hit(scaled_age)

        if not in_top_eviction:
            # req_item should be in age_to_items_deque
            # no need to add a new one to the left to age_to_element_list, which is done in update_age

            # HIGH TIME COMPLEXITY
            element_list = self.age_to_items_deque[scaled_age]
            # if self.debugging_mode:
            assert req_item in element_list, \
                    "update req_item error, req_item {} not in {}, prev {}, next {}, current ts {}, last access time {}, scaled_age {}".\
                    format(req_item, element_list, self.age_to_items_deque[scaled_age - 1], self.age_to_items_deque[scaled_age + 1],
                               self.current_ts, self.access_time_d[req_item], scaled_age)
            # HIGH TIME COMPLEXITY 
            element_list.remove(req_item)
            # self.age_to_items_deque[age] = element_list

        else:
            self.overage_item_set.remove(req_item)
            self.overage_item_deque.remove(req_item)

        # self.access_time_d[req_item] = self.current_ts


    # @jit
    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        ATTENTION: after insert, number of items in age_to_element_list and number of items in access_time_d is different
        adding to age_to_element_list is updated in update_age
        :param **kwargs:
        :param req_item:
        :return: evicted element or None
        """
        # self.access_time_d[req_item] = self.current_ts

        self.cache_content[req_item] = True
        scaled_age = self._get_age(req_item)
        if scaled_age >= self.max_age: 
            print("insert scaled age {} maxage {}".format(scaled_age, self.max_age))
            scaled_age = self.max_age - 1

        if self.enable_classification:
            # Jason: what will be the age of non-reused-class get hit
            self.non_reused_class.hit(scaled_age)


    # @jit
    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: content of evicted element
        """
        # raise RuntimeError("ts {} need to evict, current stack size {} stack {}".format(
        #                     self.current_ts, len(self.access_time_d), self.access_time_d))

        evict_age = -1
        element_to_evict = None
        evict_from = ""
        evict_element_class = None

        # first find the ts of the element that's going to be evicted with EVA
        if len(self.overage_item_deque):
            # if there are wrap up
            # raise RuntimeError("age wrapup should be avoided")
            element_to_evict = self.overage_item_deque.popleft()
            self.overage_item_set.remove(element_to_evict)
            evict_age = self.max_age -1
            evict_from = "top_eviction_queue"
            if self.enable_classification:
                evict_element_class = self.classID[element_to_evict]
                del self.classID[element_to_evict]
        else:
            if len(self.eva_sorted_age):
                # EVA info is ready
                if self.enable_classification:
                    for (class_name, scaled_age) in self.eva_sorted_age:
                        if scaled_age >= len(self.age_to_items_deque):
                            continue
                        evict_classID = True if class_name == "reused" else False
                        # HIGH TIME COMPLEXITY
                        element_list = self.age_to_items_deque[scaled_age]
                        for element in element_list: # old->new
                            if self.classID[element] == evict_classID:
                                element_to_evict = element
                                evict_age = scaled_age
                                element_list.remove(element_to_evict)
                                break

                        if evict_age != -1:
                            # break outer loop
                            evict_from = "sorted age"
                            break

                else:
                    # no classification
                    for scaled_age in self.eva_sorted_age:
                        if scaled_age >= len(self.age_to_items_deque):
                            continue
                        # HIGH TIME COMPLEXITY
                        element_list = self.age_to_items_deque[scaled_age]
                        if len(element_list):
                            element_to_evict = element_list.popleft()
                            evict_age = scaled_age
                            evict_from = "sorted age"
                            break

            else:
                # Jason: corner case, handling choice?
                # print("eva_sorted_age have nothing {}".format(self.eva_sorted_age),
                #       file=sys.stderr)

                # use LRU to evict when no info avail
                item_list = deque()
                while evict_age == -1: 
                    while len(item_list) == 0:
                        item_list = self.age_to_items_deque.pop()
                    element_to_evict = item_list.popleft() 
                    evict_age = len(self.age_to_items_deque)
                    evict_from = "no info"
                    if len(item_list): 
                        self.age_to_items_deque.append(item_list)


                # random eviction when no info avail
                ### for e, t in self.access_time_d.items():
                # for e in self.cache_content.keys():
                #     element_to_evict = e
                #     # evict_age = (self.current_ts - t)//self.age_scaling
                #     evict_age = self._get_age(e)
                #     evict_from = "no info avail"

                #     # HIGH COMPLEXITY
                #     item_deque = self.age_to_items_deque[evict_age]
                #     assert element_to_evict in item_deque, \
                #         "ts {} in eviction, {} not in element list {}, last access ts {}, scaled_age {}".format(
                #             self.current_ts, element_to_evict, item_deque, self.access_time_d[element_to_evict], evict_age)
                #     item_deque.remove(element_to_evict)
                #     break



        if element_to_evict is None:
            raise RuntimeError("cannot find element to evict {} {}".format(self.age_to_items_deque,
                                                                           self.eva_sorted_age))
        # if socket.gethostname() != "node":
        #     print("evict age {} {} ({}) from {}, age_to_element_list size {}, top_eviction size {}".format(
        #                 evict_age, element_to_evict, self.classID[element_to_evict],
        #                 evict_from, len(self.age_to_items_deque), len(self.overage_item_deque)))

        try:
            if evict_from != "top_eviction_queue":
                # del self.access_time_d[element_to_evict]
                del self.cache_content[element_to_evict]
                if self.enable_classification:
                    evict_element_class = self.classID[element_to_evict]
                    del self.classID[element_to_evict]
            # self._verify("y {}".format(evict_from))
        except Exception as e:
            print("evict from {} error: {}".format(evict_from, e))
            raise RuntimeError("{} not in {}".format(element_to_evict, self.access_time_d))

        if self.enable_classification:
            if evict_element_class:
                # reused
                self.reused_class.evict(evict_age)
            else:
                self.non_reused_class.evict(evict_age)
        else:
            self.classes[0].evict(evict_age)


    # @jit
    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: None
        """

        self._update_age(req_item)
        # self._verify2(msg = "a")

        if self.current_ts > self.cache_size and self.current_ts % self.update_interval == 0:
            print("ts {} age_to_items_deque size {}, overage item deque size {}".format(
                self.current_ts, len(self.age_to_items_deque), len(self.overage_item_set)))
            self.reconfigure()
        # self._verify2(msg = "b")

        found = False
        if self.has(req_item, ):
            # self._verify2(msg="c")
            found = True
            if self.enable_classification:
                self.classID[req_item] = found
            self._update(req_item, )
            # self._verify2(msg="d")
        else:
            if self.enable_classification:
                self.classID[req_item] = found
            self._insert(req_item, )
            # self._verify2(msg="e")
            if len(self.cache_content) + len(self.overage_item_set) > self.cache_size:
                self.evict()
            # self._verify2(msg = "f")


        # self._update_age(req_item)
        # self._verify2(msg = "g")


        if self.current_ts % 20000 == 0:
            for cl in self.classes:
                cl.plot_eva(self.dat_name, cache_size=self.cache_size)


        return found


    def __len__(self):
        return len(self.access_time_d)


    def __repr__(self):
        return "evaJ, cache size: {}, current size: {}, {}".format(
                    self.cache_size, len(self.access_time_d), super().__repr__())


def mytest1():
    import socket
    CACHE_SIZE = 40000
    NUM_OF_THREADS = os.cpu_count()

    BIN_SIZE = CACHE_SIZE // NUM_OF_THREADS // 4 + 1
    # BIN_SIZE = CACHE_SIZE // NUM_OF_THREADS + 1
    BIN_SIZE = CACHE_SIZE
    MAX_AGE = 200000
    figname = "1127vscsi/vscsiSmall.png"
    figname = "vscsiSmall.png"
    classification = False

    mt = MyTimer()
    reader = VscsiReader("../data/trace.vscsi")
    reader = PlainReader("../data/EVA_test1")
    # reader = plainReader("../data/trace.txt2")
    # eva = EVA(cache_size=2000)

    if "EVA_test" in reader.file_loc and "node" not in socket.gethostname():
        if "EVA_test1" in reader.file_loc:
            CACHE_SIZE = 64
            # CACHE_SIZE = 8
            MAX_AGE = 300
            # MAX_AGE = 600000
            BIN_SIZE = 64
        elif "EVA_test2" in reader.file_loc:
            CACHE_SIZE = 8
            MAX_AGE = 40
            BIN_SIZE = 1

        # BIN_SIZE = CACHE_SIZE
        figname = "eva_HRC.png"

    else:
    #     reader = vscsiReader("../data/trace.vscsi")
        reader = BinaryReader("../data/trace.vscsi",
                              init_params={"label": 6, "fmt": "<3I2H2Q"})
    p0 = PyGeneralProfiler(reader, LRU, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS)
    p0.plotHRC(figname="LRUSmall.png", no_clear=True, no_save=True, label="LRU")
    mt.tick()

    print(BIN_SIZE)
    print(reader.file_loc)
    p = PyGeneralProfiler(reader, EVA, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS,
                        cache_params={"update_interval": 2000,
                                      "max_age":MAX_AGE,
                                      "enable_classification": classification,
                                      "age_scaling":1,
                                      "dat_name": "smallTest"})
    p.plotHRC(figname=figname, label="EVA")
    mt.tick()


def run_data(dat, LRU_hr=None, cache_size=-1, update_interval=2000, max_age=20000, age_scaling=1, num_of_threads=os.cpu_count()):
    from PyMimircache.bin.conf import get_reader
    # reader = get_reader(dat, dat_type)
    reader = BinaryReader("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(dat),
                          init_params={"label":6, "fmt":"<3I2H2Q"})

    CACHE_SIZE = cache_size
    NUM_OF_THREADS = num_of_threads
    # BIN_SIZE = CACHE_SIZE // NUM_OF_THREADS + 1
    BIN_SIZE = CACHE_SIZE // 40 + 1

    if os.path.exists("1127vscsi/{}/EVA_{}_ma{}_ui{}_as{}.png".format(dat, dat, max_age, update_interval, age_scaling)):
        return

    mt = MyTimer()

    if LRU_hr is None:
        profiler_LRU = PyGeneralProfiler(reader, LRU, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS)
        profiler_LRU.plotHRC(figname="1127vscsi/{}/LRU_{}.png".format(dat, dat), no_clear=True, no_save=False, label="LRU")
    else:
        plt.plot(LRU_hr[0], LRU_hr[1], label="LRU")
        plt.legend(loc="best")
    mt.tick()

    p = PyGeneralProfiler(reader, EVA, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS,
                        cache_params={"update_interval": update_interval,
                                      "max_age":max_age,
                                      "enable_classification": False,
                                      "age_scaling":age_scaling,
                                      "dat_name": "noclass_{}_{}_{}".format(dat, update_interval, max_age)})
    p.plotHRC(no_clear=True, figname="1127vscsi/{}/EVA_{}_ma{}_ui{}_as{}_no_class.png".
              format(dat, dat, max_age, update_interval, age_scaling), label="EVA_noclass")
    mt.tick()

    p = PyGeneralProfiler(reader, EVA, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS,
                        cache_params={"update_interval": update_interval,
                                      "max_age":max_age,
                                      "enable_classification": True,
                                      "age_scaling":age_scaling,
                                      "dat_name": "class_{}_{}_{}".format(dat, update_interval, max_age)})
    p.plotHRC(figname="1127vscsi/{}/EVA_{}_ma{}_ui{}_as{}_classification.png".
                format(dat, dat, max_age, update_interval, age_scaling), label="EVA_class")
    mt.tick()




def run2(parallel=True):
    CACHE_SIZE = 80000
    BIN_SIZE = CACHE_SIZE//os.cpu_count()+1
    LRU_HR_dict = {}

    for dat in ["w78", "w92", "w106"]:
        reader = BinaryReader("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(dat),
                              init_params={"label":6, "fmt":"<3I2H2Q"})
        profiler_LRU = PyGeneralProfiler(reader, LRU, cache_size=CACHE_SIZE, bin_size=BIN_SIZE,
                                       num_of_threads=os.cpu_count())
        # hr = [1]
        hr = profiler_LRU.get_hit_ratio()
        LRU_HR_dict[dat] = ([i*BIN_SIZE for i in range(len(hr))], hr)

    if not parallel:
        for ma in [20000, 2000, 200000, 2000000]:
            for ui in [2000, 20000, 200000]:
                for age_scaling in [1, 2, 5, 10, 20, 100, 200]:
                    for dat in ["w92", "w106", "w78"]:
                        run_data(dat, LRU_HR_dict[dat], CACHE_SIZE,
                                 update_interval=ui, max_age=ma, age_scaling=age_scaling, num_of_threads=os.cpu_count())
    else:
        max_workers = 12
        with ProcessPoolExecutor(max_workers=max_workers) as ppe:
            futures_to_params = {}
            for ma in [20000, 2000, 200000, 2000000]:
                for ui in [2000, 20000, 200000]:
                    for age_scaling in [1, 2, 5, 10, 20, 100, 200]:
                        for dat in ["w92", "w106", "w78"]:
                            futures_to_params[ppe.submit(run_data, dat, LRU_HR_dict[dat], CACHE_SIZE,
                                        ui, ma, age_scaling, os.cpu_count()//max_workers)] = (ma, ui, age_scaling, dat)

            count = 0
            for i in as_completed(futures_to_params):
                result = i.result()
                count += 1
                print("{}/{}".format(count, len(futures_to_params)), end="\n")



def run_small():
    CACHE_SIZE = 40000
    BIN_SIZE = CACHE_SIZE // os.cpu_count() + 1
    NUM_OF_THREADS = os.cpu_count()

    # reader = vscsiReader("../data/trace.vscsi")
    reader = BinaryReader("../data/trace.vscsi",
                          init_params={"label": 6, "fmt": "<3I2H2Q"}, open_c_reader=False)
    p0 = PyGeneralProfiler(reader, LRU, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS)
    p0.plotHRC(no_clear=True, no_save=True, label="LRU")

    for ma in [2000, 5000, 20000, 30000, 40000]:
        for ui in [1000, 2000, 5000, 20000]:
            for age_scaling in [1, 2, 5, 10, 20, 100]:
                for cls in [True, False]:
                    figname = "1127vscsi0/small/small_ma{}_ui{}_as{}_{}.png".format(ma, ui, age_scaling, cls)
                    if os.path.exists(figname):
                        print(figname)
                        continue
                    p0.plotHRC(no_clear=True, no_save=True, label="LRU")
                    p = PyGeneralProfiler(reader, EVA, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS,
                                        cache_params={"update_interval": ui,
                                                      "max_age": ma,
                                                      "enable_classification": cls,
                                                      "age_scaling": age_scaling,
                                                      "dat_name": "smallTest"})
                    plt.legend(loc="best")
                    p.plotHRC(figname=figname, label="EVA")
                    reader.reset()



if __name__ == "__main__":

    mytest1()

    # run_small()
    # run2()

    # when update interval
