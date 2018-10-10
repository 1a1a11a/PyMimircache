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

from collections import defaultdict

import os


DEBUG_MODE = False

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def argsort_dict(d):
    return sorted(d.keys(), key=d.__getitem__)


class LHD(Cache):
    class LHDClass:
        def __init__(self, max_age, name="name", ema_decay=1, dat_name="dat_name", cache_size=-1, coarsen_age_shift=0):
            self.name = name
            self.max_age = max_age
            self.ema_decay = ema_decay
            self.dat_name = dat_name
            self.cache_size = cache_size
            self.coarsen_age_shift = coarsen_age_shift


            self.total_hit_interval = 0
            self.total_evict_interval = 0
            self.total_hit = 0
            self.total_evict = 0


            self.hits_interval = [0] * (self.max_age + 1)
            self.evicts_interval = [0] * (self.max_age + 1)
            self.hit_density = [0] * (self.max_age + 1)

            self.reset()


            self.eva_plot_count = 1 if DEBUG_MODE else -1


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

            for i in range(self.max_age, -1, -1):
                hits_up_to_now += self.hits_interval[i]
                evicts_up_to_now += self.evicts_interval[i]
                unconditioned_events += hits_up_to_now + evicts_up_to_now
                if unconditioned_events > 1e-5:
                    self.hit_density[i] = hits_up_to_now / unconditioned_events
                else:
                    self.hit_density[i] =  0
                # print("{} hd{} hi{} ei{} hitToNow{} evictToNow{} uncon{}".format(
                #     i, self.hit_density[i], self.hits_interval[i], self.evicts_interval[i],
                #     hits_up_to_now, evicts_up_to_now, unconditioned_events))
            pos = []
            for i in range(len(self.hits_interval)):
                if self.hits_interval[i] == 10000:
                    pos.append(i)
            self.plot_eva()
            self.reset()


        def reset(self):
            self.hits_interval = [0] * (self.max_age + 1)
            self.evicts_interval = [0] * (self.max_age + 1)


        def plot_eva(self):
            if self.eva_plot_count >= 0:
                self.eva_plot_count += 1

                # if self.eva_plot_count % 8 == 0:
                if True:
                    figname = "0512LHD_Value/{}/LHD_{}_size{}_plot{}.png".format(
                        self.dat_name, self.name, self.cache_size, self.eva_plot_count)
                    if not os.path.exists(os.path.dirname(figname)):
                        os.makedirs(os.path.dirname(figname))

                    plt.clf()
                    plt.plot([2**self.coarsen_age_shift*i for i in range(len(self.hits_interval))], self.hits_interval, label="hit")
                    plt.plot([2**self.coarsen_age_shift*i for i in range(len(self.hits_interval))], self.evicts_interval, label="evict")
                    plt.ylabel("count")
                    plt.xlabel("Age")
                    plt.legend(loc="best")
                    plt.savefig("0512LHD_Value/{}/hitEvicts_{}_size{}_plot{}.png".format(
                        self.dat_name, self.name, self.cache_size, self.eva_plot_count))
                    plt.clf()

                    plt.plot([2**self.coarsen_age_shift*i for i in range(len(self.hits_interval))], self.hit_density,
                             label="{}_{}_{}".format(self.name, self.cache_size, self.eva_plot_count))
                    plt.ylabel("LHD")
                    plt.xlabel("Age")
                    plt.legend(loc="best")
                    plt.savefig(figname)
                    print("LHD value plot {} done".format(figname))
                    plt.clf()


    class CacheItem:
        __slots__ = "req_id", "last_access_ts", "last_age", "last_last_age", "explorer"


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


        self.explore_inverse_prob = kwargs.get("explore_inverse_prob", 32)

        self.coarsen_age_shift = coarsen_age_shift
        self.num_reconfig = 0


        if kwargs.get("max_coarsen_age", -1) == -1:
            self.max_coarsen_age = self.cache_size
        else:
            self.max_coarsen_age = kwargs.get("max_coarsen_age")
            # if kwargs.get("max_age", -1) == -1:
            #     self.max_age = int(self.max_coarsen_age * (2 ** self.coarsen_age_shift))
            # else:
            #     raise RuntimeError("can only specify one of the two, max_coarsen_age, max_age")
        self.explore_budget = int(kwargs.get("explore_budget", 0.01) * self.cache_size)
        random.seed()

        self.enable_stat = enable_stat
        self.dat_name = dat_name
        self.num_overflow = 0


        print("cache size {}, update interval {}, decay coefficient {}, max_age {}, "
              "max_coarsen_age {}, age shift {}".format(
            self.cache_size, self.update_interval, self.ema_decay,
            self.max_coarsen_age * (2 ** self.coarsen_age_shift), self.max_coarsen_age,
            self.coarsen_age_shift), end="\n")

        self.current_ts = 0
        self.classes = [self.LHDClass(max_age=self.max_coarsen_age, name="class{}".format(i),
                                      ema_decay=self.ema_decay, dat_name=self.dat_name,
                                      cache_size=self.cache_size,
                                      coarsen_age_shift=self.coarsen_age_shift)
                        for i in range(self.n_classes)]


        self.cache_items = []
        self.cache_item_index = {}
        self.class_hits = [0] * self.n_classes
        self.freq_count = defaultdict(int)

    def get_cur_age(self, req_id):
        """
        return coarsened age
        :param req:
        :return:
        """

        cache_item = self.cache_items[self.cache_item_index[req_id]]
        # if req_id in self.cache_item_index:
        #     cache_item = self.cache_items[self.cache_item_index[req_id]]
        # elif req_id in self.ghost_cache_items:
        #     cache_item = self.ghost_cache_items[req_id]
        # else:
        #     raise RuntimeError("req not in either cacheDict nor in ghost")

        last_access_ts = cache_item.last_access_ts
        cur_age = self.current_ts - last_access_ts
        coarsened_age = int(math.floor(cur_age / (2 ** self.coarsen_age_shift)))

        if coarsened_age >= self.max_coarsen_age:
            self.num_overflow += 1
            return self.max_coarsen_age
        else:
            return coarsened_age


    def get_class_id0(self, req_id):
        cache_item = self.cache_items[self.cache_item_index[req_id]]


        last_age = cache_item.last_age
        last_last_age = cache_item.last_last_age
        base_age_shift = int(math.ceil(math.log(self.max_coarsen_age, 2))) - self.n_classes
        # print("base {}, last age {}, {} {}".format(2**base_age_shift, last_age,
        #                                         math.log( (last_age) / (2 ** base_age_shift), 2),
        #                                         int(math.log((self.max_coarsen_age - last_age), 2))))

        # if last_age + last_last_age == 0:
        if last_age == 0:
            class_id = self.n_classes - 1
        elif last_age < 2 ** base_age_shift:
            class_id = 0
        else:
            # class_id = max(int(math.log( (last_age + last_last_age) / (2 ** base_age_shift), 2)), self.n_classes - 1)
            # class_id = min(int(math.log( (self.max_coarsen_age - last_age), 2)), self.n_classes - 1)
            class_id = min(int(math.log( (last_age) / (2 ** base_age_shift), 2)), self.n_classes - 1)
        # print("lastAge {}, {}, class id {}".format(last_age, int(math.log( (self.max_coarsen_age - last_age), 2)), class_id))


        # last_age = cache_item.last_age
        # last_last_age = cache_item.last_last_age
        # if last_age + last_last_age == 0 or self.max_coarsen_age - (last_age + last_last_age) <= 0:
        #     class_id = self.n_classes - 1
        # else:
        #     # return min(self.n_classes-2, int(math.floor(math.log(self.max_coarsen_age - (last_age + last_last_age), 2))))
        #     class_id = min(self.n_classes-1, int(math.floor(math.log(self.max_coarsen_age - (last_age + last_last_age), 2))))

        self.class_hits[class_id] += 1
        return class_id

    def get_class_id(self, req_id):
        return min(self.freq_count[req_id], self.n_classes-1)

    # @jit
    def reconfigure(self):
        for cl in self.classes:
            cl.fit()
        self.adjust_age_coarsen_shift()
        self.num_reconfig += 1


    def adjust_age_coarsen_shift(self):
        if self.num_overflow > self.cache_size:
            print("overflow {}, age shift {}".format(self.num_overflow, self.coarsen_age_shift))
            self.coarsen_age_shift += 1
            for c in self.classes:
                c.coarsen_age_shift += 1
                hits   = [0] * (self.max_coarsen_age + 1)
                evicts = [0] * (self.max_coarsen_age + 1)
                for i in range(0, self.max_coarsen_age+1):
                    hits[i//2] += c.hits_interval[i]
                    evicts[i//2] += c.evicts_interval[i]
                c.hits_interval = hits
                c.evicts_interval = evicts

        self.num_overflow = 0

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
        if cache_item.explorer:
            if random.randrange(2) < 1:
                cache_item.explorer = True
            else:
                cache_item.explorer = False
                self.explore_budget += 1

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
        :param **kwargs:
        :param req_item:
        :return: evicted element or None
        """

        cache_item = self.CacheItem()
        cache_item.last_access_ts = self.current_ts
        cache_item.last_age = 0
        cache_item.last_last_age = self.max_coarsen_age-1
        cache_item.req_id = req_id

        if self.explore_budget >0 and random.randrange(2) < 1:
            cache_item.explorer = True
            self.explore_budget -= 1
        else:
            cache_item.explorer = False

        self.cache_item_index[req_id] = len(self.cache_items)
        self.cache_items.append(cache_item)


    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: content of evicted element
        """

        min_hit_density = sys.maxsize
        min_hit_density_ind = -1
        ca_list = []

        if self.num_reconfig == 0:
            min_hit_density_ind = random.randrange(0, len(self.cache_item_index))
        else:
            for i in range(min(self.evict_rand_num, len(self.cache_items))):
                ind = random.randrange(0, len(self.cache_item_index))
                cache_item = self.cache_items[ind]
                coarsened_age = self.get_cur_age(cache_item.req_id)
                ca_list.append(coarsened_age)
                class_id = self.get_class_id(cache_item.req_id)
                if cache_item.explorer and coarsened_age < self.max_coarsen_age // 2:
                    continue
                if coarsened_age == self.max_coarsen_age - 1:
                    min_hit_density_ind = ind
                    break
                else:
                    if True:
                        hd = self.classes[class_id].hit_density[coarsened_age]
                    else:
                        # LRU
                        hd = -coarsened_age
                    if hd < min_hit_density:
                        min_hit_density = hd
                        min_hit_density_ind = ind

        cache_item = self.cache_items[min_hit_density_ind]
        coarsened_age = self.get_cur_age(cache_item.req_id)
        class_id = self.get_class_id(cache_item.req_id)
        self.classes[class_id].evicts_interval[coarsened_age] += 1
        # if len(ca_list):
        #     print("evict age {} {}".format(coarsened_age * (2**self.coarsen_age_shift), sorted(ca_list)))

        if min_hit_density_ind != len(self.cache_items) - 1:
            item_to_move = self.cache_items.pop()
            self.cache_items[min_hit_density_ind] = item_to_move
            self.cache_item_index[item_to_move.req_id] = min_hit_density_ind
        else:
            self.cache_items.pop()

        if cache_item.explorer:
            self.explore_budget += 1
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


        # if self.current_ts % 200000 == 0:
        #     print("cache {} {} overflow {} class hits {}".format(self.cache_size, self.current_ts, self.num_overflow, self.class_hits))

        self.freq_count[req_item] += 1
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
            CACHE_SIZE = 160
            # CACHE_SIZE = 48
            MAX_COARSEN_AGE = 300
            BIN_SIZE = 2
            # BIN_SIZE = 160
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
                        cache_params={"update_interval": 20000,
                                      "coarsen_age_shift": 0,
                                      "n_classes":2,
                                      # "ema_decay": 0,
                                      "max_coarsen_age":MAX_COARSEN_AGE,
                                      "dat_name": "smallTest"})
    p.plotHRC(figname=figname, label="LHD")
    mt.tick()


def run_data(dat, cache_size, update_interval, coarsen_age_shift, n_classes, max_coarsen_age, LRU_hr=None, num_of_threads=os.cpu_count()):
    reader = BinaryReader("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(dat),
                          init_params={"label":6, "fmt":"<3I2H2Q"})
    # reader = CsvReader("/home/jason/ALL_DATA/{}.csv".format(dat), init_params={"header": False, "delimiter": ",", "label": 5, "real_time": 1, "size": 6})
    # reader = CsvReader("/home/jason/temp/dat", init_params={"header": False, "delimiter": ",", "label": 5, "real_time": 1, "size": 6})

    bin_size = cache_size // num_of_threads // 4 + 1
    # BIN_SIZE = CACHE_SIZE // 40 + 1

    if os.path.exists("0512LHD/{}/LHD_{}.png".format(dat, dat)):
        return
    else:
        if not os.path.exists("0512LHD/{}".format(dat)):
            os.makedirs("0512LHD/{}".format(dat))

    mt = MyTimer()

    if LRU_hr is None:
        profiler_LRU = PyGeneralProfiler(reader, LRU, cache_size=cache_size, bin_size=bin_size, num_of_threads=num_of_threads)
        profiler_LRU.plotHRC(figname="0512LHD/{}/LRU_{}.png".format(dat, dat), no_clear=True, no_save=False, label="LRU")
    else:
        plt.plot(LRU_hr[0], LRU_hr[1], label="LRU")
        plt.legend(loc="best")
    mt.tick()

    p = PyGeneralProfiler(reader, LHD, cache_size=cache_size, bin_size=bin_size, num_of_threads=num_of_threads,
                        cache_params={"update_interval": update_interval,
                                      "coarsen_age_shift": coarsen_age_shift,
                                      "n_classes":n_classes,
                                      "max_coarsen_age":max_coarsen_age,
                                      "dat_name": "{}".format(dat)})
    p.plotHRC(no_clear=True, figname="0512LHD/{}/LHD_{}.png".
              format(dat, dat), label="LHD")
    plt.clf()
    mt.tick()


def run2(parallel=False):
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
        for ma in [200000, 2000000]:
            for ui in [2000, 20000, 200000]:
                for coarsen_age_shift in [0, 2, 4, 8]:
                    for dat in ["w92", "w106", "w78"]:
                        run_data(dat, cache_size=CACHE_SIZE, LRU_hr=LRU_HR_dict[dat],
                                 update_interval=ui, max_coarsen_age=ma//(2**coarsen_age_shift)+1,
                                 n_classes=1,
                                 coarsen_age_shift=coarsen_age_shift, num_of_threads=os.cpu_count())
    else:
        max_workers = 12
        n_classes = 1
        with ProcessPoolExecutor(max_workers=max_workers) as ppe:
            futures_to_params = {}
            for max_age in [200000, 2000000]:
                for n_classes in [1]:
                    for update_interval in [2000, 20000, 200000]:
                        for coarsen_age_shift in [0, 2, 4, 8]:
                            for dat in ["w92", "w106", "w78"]:
                                futures_to_params[ppe.submit(run_data, dat,
                                                         CACHE_SIZE, update_interval, coarsen_age_shift, n_classes,
                                                         max_age//(2**coarsen_age_shift)+1, LRU_HR_dict[dat])] = (max_age, update_interval, coarsen_age_shift, dat)

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


def test(dat="w92"):
    from PyMimircache.profiler.cLRUProfiler import CLRUProfiler
    reader = BinaryReader("/home/cloudphysics/traces/{}_vscsi1.vscsitrace".format(dat),
                          init_params={"label":6, "fmt":"<3I2H2Q"})
    rd = CLRUProfiler(reader).get_reuse_distance()
    rd_count_list = [0] * (int(math.log(max(rd), 2)+2))
    for r in rd:
        if r > 0:
            rd_count_list[int(math.log(r, 2))] += 1
    print(rd_count_list[:200])
    plt.plot([2**(i+1) for i in range(len(rd_count_list))], rd_count_list)
    plt.savefig("a.png")
    plt.clf()
    plt.hist([i for i in rd if i!=-1], log=False)
    plt.savefig("b.png")


if __name__ == "__main__":
    from PyMimircache.profiler.pyGeneralProfiler import PyGeneralProfiler
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from PyMimircache.utils.timer import MyTimer
    from PyMimircache.cache.lru import LRU

    from PyMimircache.cacheReader.vscsiReader import VscsiReader
    from PyMimircache.cacheReader.binaryReader import BinaryReader
    from PyMimircache.cacheReader.csvReader import CsvReader
    from PyMimircache.cacheReader.plainReader import PlainReader

    import matplotlib.pyplot as plt


    # mytest1()

    # test()
    # run_data("src1_0", cache_size=1600000, update_interval=200000, coarsen_age_shift=3, n_classes=1, max_coarsen_age=2000000)
    run_data("w92", cache_size=800000, update_interval=200000, coarsen_age_shift=3, n_classes=24, max_coarsen_age=200000)

    # for i in range(106, 0, -1):
    #     try:
    #         run_data("w{}".format(i), cache_size=16000, update_interval=20000, coarsen_age_shift=0, n_classes=1,
    #                  max_coarsen_age=200000)
    #     except Exception as e:
    #         print(e)


    # run_small()
    # run2()

    # when update interval