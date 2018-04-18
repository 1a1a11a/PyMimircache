# coding=utf-8

"""
    This module uses LRU for low-freq and ASig for mid-freq and high-freq

"""

from collections import defaultdict
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.const import ALLOW_C_MIMIRCACHE, INSTALL_PHASE
if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.LRUProfiler as c_LRUProfiler
    import PyMimircache.CMimircache.Heatmap as c_heatmap
from heapdict import heapdict
from collections import OrderedDict

class ASigOPT(Cache):
    def __init__(self, cache_size, reader, **kwargs):
        super().__init__(cache_size, **kwargs)
        # reader.reset()
        self.reader = reader
        self.reader.lock.acquire()
        self.next_access = c_heatmap.get_next_access_dist(self.reader.c_reader)
        self.reader.lock.release()

        self.LRU_queue = OrderedDict()
        self.pq = heapdict()
        self.pq_reverse = heapdict()

        self.freq_threshold = kwargs.get("freq_threshold", 12)

        self.pq_ready = False
        self.access_time = self.get_access_time()
        self.last_expected_dist = {}
        self.mid_freq_obj = set()
        self.high_freq_obj = set()
        self.evict_reason = defaultdict(int)

        self.ts = 0


    def get_access_time(self):
        access_time = defaultdict(list)
        for ts, req in enumerate(self.reader):
            access_time[req].append(ts)
        return access_time

    def get_reversed_reuse_dist(self):
        return c_LRUProfiler.get_reversed_reuse_dist(self.reader.cReader)

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        if req_id in self.pq or req_id in self.LRU_queue:
            return True
        else:
            return False

    def _update_metadata(self, req_item, **kwargs):
        # mid-freq or high-freq
        add_to_LRU = False
        if req_item in self.mid_freq_obj:
            if self.next_access[self.ts] != -1:
                self.pq[req_item] = self.ts + self.next_access[self.ts]         # multiply by 2 affects little
                self.last_expected_dist[req_item] = self.next_access[self.ts]   # or just use cache size, little impact
            else:
                self.pq[req_item] = self.ts + self.last_expected_dist.get(req_item, self.cache_size)

        else:
            # low-freq
            add_to_LRU = True
            if req_item not in self.high_freq_obj and \
                        len(self.access_time[req_item]) >= self.freq_threshold:
                # check whether req can upgrade to mid-freq
                ind = -1
                for i in range(len(self.access_time[req_item])):
                    if self.access_time[req_item][i] > self.ts or i > self.freq_threshold:
                        ind = i
                        break
                # print("{} {}, {} {} {}".format(ind, self.freq_threshold, len(self.access_time[req_item]),
                #                                         self.ts, self.access_time[req_item]))
                # you have free upgrade
                if ind >= self.freq_threshold:
                    self.mid_freq_obj.add(req_item)
                    if self.next_access[self.ts] != -1:
                        self.pq[req_item] = self.ts + self.next_access[self.ts]
                        self.last_expected_dist[req_item] = self.next_access[self.ts]
                    else:
                        self.pq[req_item] = self.ts + self.last_expected_dist.get(req_item, self.cache_size)
                    add_to_LRU = False
            if add_to_LRU:
                self.LRU_queue[req_item] = self.ts

        if not add_to_LRU:
            self.pq_reverse[req_item] = -self.pq[req_item]



    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        :return: None
        """
        if req_item in self.LRU_queue:
            del self.LRU_queue[req_item]
        elif req_item in self.pq:
            del self.pq[req_item]
        self._update_metadata(req_item)


    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        self._update_metadata(req_item)


    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """

        if len(self.pq) >= self.cache_size//10 * 9:
            self.pq_ready = True
        else:
            self.pq_ready = False
            # new_pq = heapdict()
            # for k, v in self.pq.items():
            #     new_pq[k] = -v
            # print("PQ ready")

        if self.pq_ready:
            self.evict_pq_ready(**kwargs)
        else:
            self.evict_no_pq_ready(**kwargs)


    def evict_pq_ready(self, **kwargs):
        element, _ = self.pq_reverse.popitem()
        del self.pq[element]
        self.evict_reason["PQReady"] += 1
        return element


    def evict_no_pq_ready(self, **kwargs):
        evict_element = None
        if len(self.pq):
            element, expiration_time = self.pq.peekitem()
            if expiration_time < self.ts:
                evict_element = element
                self.pq.popitem()
                del self.pq_reverse[element]
                self.evict_reason["ASigExpire"] += 1

        if evict_element is None:
            # this must meets
            # if len(self.LRU_queue):
            element, ts = self.LRU_queue.popitem(last=False)
            # need better technique here to replace the magic number
            if self.ts - ts > self.cache_size // 10:
                evict_element = element
                self.evict_reason["LRU"] += 1
            else:
                self.LRU_queue[element] = ts
                self.LRU_queue.move_to_end(element, last=False)

            # else:
            #     # everything in pq, but none is expired
            #     element, expiration_time = self.pq.popitem()
            #     evict_element = element
            #     del self.pq_reverse[element]
            #     self.evict_reason["ASigFull"] += 1

        return evict_element


    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param req_item: the element in the reference, it can be in the cache, or not,
                        !!! Attention, for optimal, the element is a tuple of
                        (timestamp, real_request)
        :return: True if element in the cache
        """

        # if self.ts % 20000 == 0:
        #     print("{} size {}+{} {}, evict reason {}".format(self.ts, len(self.pq), len(self.LRU_queue),
        #             self.pq_ready, ", ".join("{}: {}".format(k, v) for k, v in self.evict_reason.items())))

        if self.has(req_item, ):
            self._update(req_item, )
            self.ts += 1
            return True
        else:
            self._insert(req_item, )
            if len(self.pq) + len(self.LRU_queue) > self.cache_size:
                self.evict()
            self.ts += 1
            return False

    def __repr__(self):
        return "Optimal Cache, current size: {}".\
            format(self.cache_size, super().__repr__())


