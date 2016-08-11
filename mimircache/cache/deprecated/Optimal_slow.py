import sys
from queue import PriorityQueue

from mimircache.cache.abstractCache import cache
import mimircache.c_LRUProfiler as c_LRUProfiler
import mimircache.c_heatmap as c_heatmap

import time


class Optimal(cache):
    def __init__(self, cache_size, reader):
        super().__init__(cache_size)
        reader.reset()
        self.reader = reader
        self.reader.lock.acquire()
        self.next_access = c_heatmap.get_next_access_dist(self.reader.cReader)
        self.reader.lock.release()
        # print(self.next_access)
        self.pq = PriorityQueue()
        self.seenset = set()

        # used to store the real timestamp because it is hard to del the element from priority queue
        # self.real_ts = {}
        self.ts = 0

    def get_reversed_reuse_dist(self):
        return c_LRUProfiler.get_reversed_reuse_dist(self.reader.cReader)

    def checkElement(self, element):
        """
        :param element:
        :return: whether the given element is in the cache
        """
        if element in self.seenset:
            return True
        else:
            return False

    def _updateElement(self, element):
        """ the given element is in the cache, now update it to new location
        :param element:
        :return: None
        """

        # TERRIBLE ALGORITHM HERE, NEEDS to BE CHANGED

        pq_new = PriorityQueue()
        while self.pq.qsize():
            element_t = self.pq.get()
            if element_t[1] == element:
                if self.next_access[self.ts] != -1:
                    pq_new.put((-(self.ts + self.next_access[self.ts]), element))
                else:
                    self.seenset.remove(element)
            else:
                pq_new.put(element_t)
        self.pq = pq_new



        # # update priorityQueue is hard and time-consuming (O(N)), so update the real value hash table
        # if self.next_access[element[0]] != -1:
        #     # pass
        #     self.real_ts[element[1]] = -(self.next_access[element[0]] + element[0])
        #     self.pq.put( (-(self.next_access[element[0]] + element[0]), element[1]) )
        # else:
        #     pass
        #     # self.real_ts[element[1]] = -sys.maxsize

    def _insertElement(self, element):
        """
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        """
        # self.seenset.add(element[1])
        # self.pq.put(-(self.reversed_rd[element[0]]+element[0], element[1]))

        if self.next_access[self.ts] == -1:
            # self.pq.put((-sys.maxsize, element))
            pass
        else:
            self.pq.put((-self.next_access[self.ts] - self.ts, element))
            # print("Add: {}:{}".format(-self.next_access[element[0]]-element[0], element[1]))
            self.seenset.add(element)

    def _printCacheLine(self):
        print("size %d" % len(self.seenset))
        for i in self.seenset:
            print(i, end='\t')
        print('')

    def _evictOneElement(self):
        """
        evict one element from the cache line
        :return: True on success, False on failure
        """
        # needs to change this for performance
        element = self.pq.get()
        self.seenset.remove(element[1])
        # print("evicting "+str(element))
        return element




        # element = self.pq.get()
        # print('evict: {}?: '.format(element[1]), end='\t')
        # self.seenset.remove(element[1])
        # if not element[1] in self.real_ts:
        #     print(element[1])
        # while element[1] in self.real_ts:
        #     if self.real_ts[element[1]] != element[0]:
        #         print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        #         element = self.pq.get()
        #     else:
        #         print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
        #         # del self.real_ts[element[1]]
        #         print(element[1])
        #         break

    def addElement(self, element):
        """
        :param element: the element in the reference, it can be in the cache, or not,
                        !!! Attention, for optimal, the element is a tuple of
                        (timestamp, real_request)
        :return: True if element in the cache
        """
        if self.pq.qsize() != len(self.seenset):
            print("ERROR: %d: %d" % (self.pq.qsize(), len(self.seenset)))
            print(self.seenset)
            sys.exit(1)
        if self.checkElement(element):
            self._updateElement(element)
            self.ts += 1
            return True
        else:
            self._insertElement(element)
            if self.pq.qsize() > self.cache_size:
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
