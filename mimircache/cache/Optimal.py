import sys
from queue import PriorityQueue

from mimircache.cache.abstractCache import cache
import mimircache.c_LRUProfiler as c_LRUProfiler
import mimircache.c_heatmap as c_heatmap

import time
from heapdict import heapdict


class Optimal(cache):
    def __init__(self, cache_size, reader):
        super().__init__(cache_size)
        reader.reset()
        self.reader = reader
        self.reader.lock.acquire()
        self.next_access = c_heatmap.get_next_access_dist(self.reader.cReader)
        self.reader.lock.release()
        self.pq = heapdict()
        # self.seenset = set()

        self.ts = 0

    def get_reversed_reuse_dist(self):
        return c_LRUProfiler.get_reversed_reuse_dist(self.reader.cReader)

    def checkElement(self, element):
        """
        :param element:
        :return: whether the given element is in the cache
        """
        if element in self.pq:
            return True
        else:
            return False

    def _updateElement(self, element):
        """ the given element is in the cache, now update it to new location
        :param element:
        :return: None
        """

        if self.next_access[self.ts] != -1:
            self.pq[element] = -(self.ts + self.next_access[self.ts])
        else:
            del self.pq[element]


    def _insertElement(self, element):
        """
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        """
        if self.next_access[self.ts] == -1:
            pass
        else:
            self.pq[element] = -self.next_access[self.ts] - self.ts
            # self.seenset.add(element)

    def _printCacheLine(self):
        print("size %d" % len(self.pq))
        for i in self.pq:
            print(i, end='\t')
        print('')

    def _evictOneElement(self):
        """
        evict one element from the cache line
        :return: True on success, False on failure
        """

        element = self.pq.popitem()[0]
        # self.seenset.remove(element)
        # print("evicting "+str(element))
        return element


    def addElement(self, element):
        """
        :param element: the element in the reference, it can be in the cache, or not,
                        !!! Attention, for optimal, the element is a tuple of
                        (timestamp, real_request)
        :return: True if element in the cache
        """
        # if self.pq.qsize() != len(self.seenset):
        #     print("ERROR: %d: %d" % (self.pq.qsize(), len(self.seenset)))
        #     print(self.seenset)
        #     sys.exit(1)
        if self.checkElement(element):
            self._updateElement(element)
            self.ts += 1
            return True
        else:
            self._insertElement(element)
            if len(self.pq) > self.cache_size:
                self._evictOneElement()
            self.ts += 1
            return False

    def __repr__(self):
        return "Optimal Cache, current size: {}".format(self.cache_size, len(self.seenset),
                                                        super().__repr__())


if __name__ == "__main__":
    from mimircache.cacheReader.plainReader import plainReader
    from mimircache.cacheReader.vscsiReader import vscsiReader
    import mimircache.c_cacheReader as c_cacheReader

    reader = plainReader('../data/test4.dat')
    # reader = vscsiCacheReader('../data/trace.vscsi')
    # for i in reader:
    #     print(i)

    # reader.reset()
    print(reader.get_num_of_total_requests())

    # for i in c_LRUProfiler.get_reuse_dist_seq(reader.cReader):
    #     print(i)
    #     pass

    num = 0
    c = Optimal(100, reader)
    hit = 0
    miss = 0

    for i in reader:
        if c.addElement(i):
            hit += 1
        else:
            miss += 1
        # c.printCacheLine()
        # print(c.real_ts)
        # print("")
        num += 1
    print("hit: %d, hit rate: %f" % (hit, hit / (hit + miss)))
