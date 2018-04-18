# coding=utf-8
"""
Jason <peter.waynechina@gmail.com>  2018/01/18

LHD implementation


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
import random
from collections import deque
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.utils.linkedList import LinkedList
from heapdict import heapdict
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


class LHD(Cache):
    class LHDClass:
        def __init__(self, max_age, name="name", ema_decay=1, dat_name="dat_name", cache_size=-1):
            self.name = name
            self.max_age = max_age
            self.ema_decay = ema_decay
            self.dat_name = dat_name
            self.cache_size = cache_size


            self.total_hit_interval = 0
            self.total_evict_interval = 0
            self.total_hit = 0
            self.total_evict = 0


            self.hits_interval = [0] * (self.max_age + 1)
            self.evicts_interval = [0] * (self.max_age + 1)
            self.hit_density = [0] * (self.max_age + 1)

            self.reset()

            # self.hit_probability   = [0] * self.max_age
            # self.expected_lifetime = [0] * (self.max_age + 1)
            # self.events = [0] * (self.max_age + 1)

            # self.hits_interval = [0] * self.max_age
            # self.evicts_interval = [0] * self.max_age

            # self._ewma_hits = [0] * self.max_age
            # self._ewma_evicts = [0] * self.max_age
            #
            # self.hit_density = [0] * self.max_age
            # self.eva_plot_count = -1


        def update(self):
            self.total_hit_interval = 0
            self.total_evict_interval = 0

            for i in range(self.max_age):
                self.hits_interval[i] *= self.ema_decay
                self.evicts_interval[i] *= self.ema_decay

            self.total_hit_interval = sum(self.hits_interval)
            self.total_evict_interval = sum(self.evicts_interval)

            self.total_hit *= (1 - self.ema_decay)
            self.total_evict *= (1 - self.ema_decay)
            self.total_hit += self.total_hit_interval
            self.total_evict += self.total_evict_interval


        def fit(self):
            self.update()

            hits_up_to_now = 0
            evicts_up_to_now = 0
            unconditioned_events = 0

            # print("class {} totalHit {} totalEvict {} "
            #       "hit0 {} hit1 {} evict0 {} evixt1 {}".format(self.name, self.total_hit_interval, self.total_evict_interval,
            #                                  self.hits_interval[0], self.hits_interval[1],
            #                                  self.evicts_interval[0], self.evicts_interval[1]))

            for i in range(self.max_age+1):
                hits_up_to_now += self.hits_interval[i]
                evicts_up_to_now += self.evicts_interval[i]
                unconditioned_events += hits_up_to_now + evicts_up_to_now
                if unconditioned_events > 1e5:
                    self.hit_density[i] = hits_up_to_now / unconditioned_events
                else:
                    self.hit_density[i] =  0

            self.reset()

        def reset(self):
            self.hits_interval = [0] * (self.max_age + 1)
            self.evicts_interval = [0] * (self.max_age + 1)


        def plot_eva(self, dat_name, cache_size=None):
            if self.eva_plot_count >= 0:
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

    class CacheItem:
        __slots__ = "req_id", "last_access_ts", "last_age", "last_last_age"


    def __init__(self, cache_size=1000, n_classes=16,
                 update_interval=-1,
                 evict_rand_num=32,
                 coarsen_age_shift=10,
                 ema_decay=1,
                 enable_stat=True, dat_name="dat_name", **kwargs):
        super().__init__(cache_size)

        self.cache_size = cache_size
        self.n_classes = n_classes

        self.ema_decay = ema_decay
        self.evict_rand_num = evict_rand_num

        if update_interval == -1:
            self.update_interval = min(self.cache_size * 20, 200000)
        else:
            self.update_interval = update_interval


        self.coarsen_age_shift = coarsen_age_shift

        if kwargs.get("max_coarsen_age", -1) == -1:
            self.max_coarsen_age = self.cache_size
            # else:
            #     self.max_age = kwargs.get("max_age")
            #     self.max_coarsen_age = int(math.ceil(self.max_age / (2 ** self.coarsen_age_shift)))
        else:
            self.max_coarsen_age = kwargs.get("max_coarsen_age")
            # if kwargs.get("max_age", -1) == -1:
            #     self.max_age = int(self.max_coarsen_age * (2 ** self.coarsen_age_shift))
            # else:
            #     raise RuntimeError("can only specify one of the two, max_coarsen_age, max_age")


        self.enable_stat = enable_stat
        self.dat_name = dat_name
        self.num_overflow = 0
        self.debugging_mode = True

        print("cache size {}, update interval {}, decay coefficient {}, max_age {}, max_coarsen_age {}, age shift {}".format(
                    self.cache_size, self.update_interval, self.ema_decay, self.max_coarsen_age * (2 ** self.coarsen_age_shift), self.max_coarsen_age, self.coarsen_age_shift), end="\n")

        self.current_ts = 0
        self.classes = [self.LHDClass(max_age=self.max_coarsen_age, name=str(i),
                                      ema_decay=self.ema_decay, dat_name=self.dat_name, cache_size=self.cache_size)
                        for i in range(self.n_classes)]


        self.cache_items = []
        self.cache_item_index = {}


        # self.last_access_ts = {}
        # self.last_last_age = {}


    def get_cur_age(self, req_id):
        """
        return coarsened age
        :param req:
        :return:
        """

        cache_item = self.cache_items[self.cache_item_index[req_id]]
        last_access_ts = cache_item.last_access_ts
        cur_age = self.current_ts - last_access_ts
        coarsened_age = int(math.floor(cur_age / (2 ** self.coarsen_age_shift)))

        if coarsened_age >= self.max_coarsen_age:
            self.num_overflow += 1
            return self.max_coarsen_age
        else:
            return coarsened_age


    def get_class_id(self, req_id):
        cache_item = self.cache_items[self.cache_item_index[req_id]]
        # last_age = self.get_cur_age(req_id)
        last_age = cache_item.last_age
        last_last_age = cache_item.last_last_age
        # print("{} {} {} {}".format(last_age, last_last_age, self.max_coarsen_age, last_age + last_last_age == 0 or self.max_coarsen_age - (last_age + last_last_age) <= 0 ))
        if last_age + last_last_age == 0 or self.max_coarsen_age - (last_age + last_last_age) <= 0:
            return self.n_classes - 1
        else:
            # print(int(math.floor(math.log(self.max_coarsen_age - (last_age + last_last_age), 2))))
            return min(self.n_classes-1, int(math.floor(math.log(self.max_coarsen_age - (last_age + last_last_age), 2))))


    # @jit
    def reconfigure(self):
        for cl in self.classes:
            cl.fit()

        # self.num_overflow = 0


    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """

        if req_id in self.cache_item_index:
            return True
        else:
            return False


    def _update(self, req_id, **kwargs):
        """ the given req_item is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        """

        cache_item = self.cache_items[self.cache_item_index[req_id]]

        cur_age = self.get_cur_age(req_id)
        cache_item.last_last_age = cache_item.last_age 
        cache_item.last_age = cur_age
        cache_item.last_access_ts = self.current_ts

        class_id = self.get_class_id(req_id)

        self.classes[class_id].hits_interval[cur_age] += 1

        # print("hit age {}".format(cur_age * (2 ** self.coarsen_age_shift)))


    # @jit
    def _insert(self, req_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        ATTENTION: after insert, number of items in age_to_element_list and number of items in access_time_d is different
        adding to age_to_element_list is updated in update_age
        :param **kwargs:
        :param req_item:
        :return: evicted element or None
        """

        cache_item = self.CacheItem()
        cache_item.last_access_ts = self.current_ts
        cache_item.last_age = 0
        cache_item.last_last_age = 0
        cache_item.req_id = req_id
        self.cache_item_index[req_id] = len(self.cache_items)
        self.cache_items.append(cache_item)


    # @jit
    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: content of evicted element
        """

        min_hit_density = sys.maxsize
        min_hd_ind = -1

        for i in range(min(self.evict_rand_num, len(self.cache_items))):
            ind = random.randrange(0, len(self.cache_item_index))
            cache_item = self.cache_items[ind]
            coarsened_age = self.get_cur_age(cache_item.req_id)
            class_id = self.get_class_id(cache_item.req_id)
            if coarsened_age == self.max_coarsen_age - 1:
                min_hd_ind = ind
                break
            else:
                hd = self.classes[class_id].hit_density[coarsened_age]
                if hd < min_hit_density:
                    min_hd_ind = ind

        cache_item = self.cache_items[min_hd_ind]
        coarsened_age = self.get_cur_age(cache_item.req_id)
        class_id = self.get_class_id(cache_item.req_id)
        self.classes[class_id].evicts_interval[coarsened_age] += 1
        # print("evict age {}".format(coarsened_age * (2**self.coarsen_age_shift)))

        if min_hd_ind != len(self.cache_items) - 1:
            self.cache_items[min_hd_ind] = self.cache_items.pop()
            self.cache_item_index[self.cache_items[min_hd_ind].req_id] = min_hd_ind
        else:
            self.cache_items.pop()
        del self.cache_item_index[cache_item.req_id]


    # @jit
    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: None
        """

        self.current_ts += 1


        if self.current_ts > self.cache_size and self.current_ts % self.update_interval == 0:
            self.reconfigure()

        found = False
        if self.has(req_item, ):
            found = True
            self._update(req_item, )
        else:
            self._insert(req_item, )
            if len(self.cache_items) > self.cache_size:
                self.evict()


        if self.current_ts % 200000 == 0:        #     for cl in self.classes:
                # cl.plot_eva(self.dat_name, cache_size=self.cache_size)
            print("cache {} overflow {}".format(self.cache_size, self.num_overflow))

        return found


    def __len__(self):
        return len(self.cache_items)


    def __repr__(self):
        return "LHD, cache size: {}, current size: {}, {}".format(
                    self.cache_size, len(self.cache_items), super().__repr__())



def mytest1():
    import socket
    CACHE_SIZE = 40000
    NUM_OF_THREADS = os.cpu_count()

    BIN_SIZE = CACHE_SIZE // NUM_OF_THREADS // 4 + 1
    BIN_SIZE = CACHE_SIZE
    MAX_COARSEN_AGE = CACHE_SIZE * 8
    figname = "vscsiSmall.png"
    classification = False

    mt = MyTimer()
    reader = VscsiReader("../../data/trace.vscsi")
    reader = PlainReader("../../data/EVA_test1")
    # reader = plainReader("../data/trace.txt2")
    # eva = EVA(cache_size=2000)

    # if "EVA_test" in reader.file_loc and "node" not in socket.gethostname():
    if "EVA_test" in reader.file_loc:
        if "EVA_test1" in reader.file_loc:
            CACHE_SIZE = 64
            # CACHE_SIZE = 1
            MAX_COARSEN_AGE = 80
            BIN_SIZE = 1
            # BIN_SIZE = 64
        elif "EVA_test2" in reader.file_loc:
            CACHE_SIZE = 8
            MAX_COARSEN_AGE = 40

        # BIN_SIZE = CACHE_SIZE
        figname = "eva_HRC.png"

    else:
    # p0 = LRUProfiler(reader, cache_size=20000)
    # p0.plotHRC("LRU.png")
    #     reader = vscsiReader("../data/trace.vscsi")
        reader = BinaryReader("../../data/trace.vscsi",
                              init_params={"label": 6, "fmt": "<3I2H2Q"})
    p0 = PyGeneralProfiler(reader, LRU, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS)
    p0.plotHRC(figname="LRUSmall.png", no_clear=True, no_save=False, label="LRU")
    mt.tick()

    p = PyGeneralProfiler(reader, LHD, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS,
                        cache_params={"update_interval": -1,
                                      "coarsen_age_shift": 2,
                                      "n_classes":16,
                                      # "max_coarsen_age":MAX_COARSEN_AGE,
                                      "dat_name": "smallTest"})
    p.plotHRC(figname=figname, label="LHD")
    mt.tick()


def run_data(dat, cache_size, LRU_hr=None, num_of_threads=os.cpu_count()):
    reader = BinaryReader("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(dat),
                          init_params={"label":6, "fmt":"<3I2H2Q"})

    CACHE_SIZE = cache_size
    NUM_OF_THREADS = num_of_threads
    BIN_SIZE = CACHE_SIZE // NUM_OF_THREADS + 1
    # BIN_SIZE = CACHE_SIZE // 40 + 1

    if os.path.exists("0402LHD/{}/LHD_{}.png".format(dat, dat)):
        return

    mt = MyTimer()

    if LRU_hr is None:
        profiler_LRU = PyGeneralProfiler(reader, LRU, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS)
        profiler_LRU.plotHRC(figname="0402LHD/{}/LRU_{}.png".format(dat, dat), no_clear=True, no_save=False, label="LRU")
    else:
        plt.plot(LRU_hr[0], LRU_hr[1], label="LRU")
        plt.legend(loc="best")
    mt.tick()

    p = PyGeneralProfiler(reader, LHD, cache_size=CACHE_SIZE, bin_size=BIN_SIZE, num_of_threads=NUM_OF_THREADS,
                        cache_params={"update_interval": -1,
                                      "coarsen_age_shift": 8,
                                      "n_classes":16,
                                      "max_coarsen_age":-1,
                                      "dat_name": "{}".format(dat)})
    p.plotHRC(no_clear=True, figname="0402LHD/{}/LHD_{}.png".
              format(dat, dat), label="LHD")
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

    # run_data("w92", 80000)

    # run_small()
    # run2()

    # when update interval