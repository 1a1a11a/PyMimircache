# coding=utf-8

"""
    a cache simulating optimal with oracle predicting next access


    Author: Juncheng Yang <peter.waynechina@gmail.com> 2016/08
"""

from PyMimircache.cache.abstractCache import Cache
from PyMimircache.profiler.utils.dist import get_next_access_dist
from PyMimircache.cacheReader.requestItem import Req
from PyMimircache.const import INSTALL_PHASE
if not INSTALL_PHASE:
    try:
        from heapdict import heapdict
    except:
        print("heapdict is not installed")


class Optimal(Cache):
    def __init__(self, cache_size, reader, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.reader = reader
        self.reader.lock.acquire()
        self.next_access = get_next_access_dist(self.reader)
        self.reader.lock.release()
        self.pq = heapdict()
        self.ts = 0

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        return req_id in self.pq


    def set_init_ts(self, ts):
        """ this is used for synchronizing ts in profiler if the
        trace is not ran from the beginning of the reader"""
        self.ts = ts

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param obj_id:
        :return: None
        """

        if self.next_access[self.ts] != -1:
            self.pq[obj_id] = -(self.ts + self.next_access[self.ts])
        else:
            del self.pq[obj_id]

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id:
        :return: True on success, False on failure
        """

        if self.next_access[self.ts] == -1:
            pass
        else:
            self.pq[obj_id] = -self.next_access[self.ts] - self.ts

    def _print_cache_line(self):
        print("size %d" % len(self.pq))
        for i in self.pq:
            print(i, end='\t')
        print('')

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """

        element = self.pq.popitem()[0]
        return element

    def access(self, req_item, **kwargs):
        """
        :param **kwargs: 
        :param req_item: the element in the reference, it can be in the cache, or not,
                        !!! Attention, for optimal, the element is a tuple of
                        (timestamp, real_request)
        :return: True if element in the cache
        """

        obj_id = req_item
        if isinstance(req_item, Req):
            obj_id = req_item.obj_id

        if self.has(obj_id, ):
            self._update(obj_id, )
            self.ts += 1
            return True
        else:
            self._insert(obj_id, )
            if len(self.pq) > self.cache_size:
                evict_item = self.evict()
                if "evict_item_list" in kwargs:
                    kwargs["evict_item_list"].append(evict_item)
            self.ts += 1
            return False




    def __contains__(self, obj_id):
        return obj_id in self.pq

    def __len__(self):
        return len(self.pq)

    def get_size(self, **kwargs):
        return len(self.pq)

    def __repr__(self):
        return "Optimal Cache of size {}, current size: {}, {}".format(self.cache_size, self.get_size(), super().__repr__())
