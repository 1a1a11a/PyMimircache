# coding=utf-8


from PyMimircache.cache.abstractCache import Cache
from PyMimircache.const import ALLOW_C_MIMIRCACHE
if ALLOW_C_MIMIRCACHE:
    import PyMimircache.CMimircache.LRUProfiler as c_LRUProfiler
    import PyMimircache.CMimircache.Heatmap as c_heatmap
from heapdict import heapdict


class Optimal(Cache):
    def __init__(self, cache_size, reader, **kwargs):
        super().__init__(cache_size, **kwargs)
        # reader.reset()
        self.reader = reader
        self.reader.lock.acquire()
        self.next_access = c_heatmap.get_next_access_dist(self.reader.cReader)
        self.reader.lock.release()
        self.pq = heapdict()


        self.ts = 0

    def get_reversed_reuse_dist(self):
        return c_LRUProfiler.get_reversed_reuse_dist(self.reader.cReader)

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        if req_id in self.pq:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        :return: None
        """

        if self.next_access[self.ts] != -1:
            self.pq[req_item] = -(self.ts + self.next_access[self.ts])
        else:
            del self.pq[req_item]


    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        if self.next_access[self.ts] == -1:
            pass
        else:
            self.pq[req_item] = -self.next_access[self.ts] - self.ts
            # self.seenset.add(req_item)

    def _printCacheLine(self):
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
        # self.seenset.remove(element)
        # print("evicting "+str(element))
        return element


    def access(self, req_item, **kwargs):
        """
        :param **kwargs: 
        :param req_item: the element in the reference, it can be in the cache, or not,
                        !!! Attention, for optimal, the element is a tuple of
                        (timestamp, real_request)
        :return: True if element in the cache
        """
        if self.has(req_item, ):
            self._update(req_item, )
            self.ts += 1
            return True
        else:
            self._insert(req_item, )
            if len(self.pq) > self.cache_size:
                self.evict()
            self.ts += 1
            return False

    def __repr__(self):
        return "Optimal Cache, current size: {}".\
            format(self.cache_size, super().__repr__())


