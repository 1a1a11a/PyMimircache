# coding=utf-8
import abc
from collections import deque
from queue import PriorityQueue
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req

from heapq import heapify, heappush, heappop


# modified from Matteo Dell'Amico https://gist.github.com/matteodellamico/4451520
class priority_dict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(priority_dict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(priority_dict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super(priority_dict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()


class LFU(Cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.cache_dict = priority_dict()  # key -> freq
        self.ts = 0

    def has(self, obj_id, **kwargs):
        """
        :param obj_id:
        :param kwargs:
        :return: whether the given element is in the cache
        """
        return obj_id in self.cache_dict

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, now update its frequency
        :param **kwargs:
        :param obj_id:
        :return: original rank
        """
        self.cache_dict[obj_id] += 1

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id:
        :return: True on success, False on failure
        """

        self.cache_dict[obj_id] = 1

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        return self.cache_dict.pop_smallest()

    def access(self, req_item, **kwargs):
        """
        :param obj_id: the element in the reference, it can be in the cache, or not
        :param kwargs:
        :return: -1 if not in cache, otherwise old rank
        """

        obj_id = req_item
        if isinstance(req_item, Req):
            obj_id = req_item.obj_id

        if self.has(obj_id, ):
            self._update(obj_id, )
            find_in_cache = True
        else:
            self._insert(obj_id, )
            if len(self.cache_dict) > self.cache_size:
                self.evict()
            find_in_cache = False

        self.ts += 1
        return find_in_cache

    def __repr__(self):
        return "LFU, given size: {}, current size: {}".format(self.cache_size, len(self.cache_dict))

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    import os
    from PyMimircache import CsvReader

    reader = CsvReader(os.path.expanduser("~/data/SmartCacheNew/largeData/akamai1TrafficID458"), init_params={"header": False, "delimiter": "\t", "label": 5, 'real_time': 1})

    cache = LFU2(cache_size=20000)
    i = 0
    req = reader.read_as_req_item()
    while req:
        hit = cache.access(req, )
        req = reader.read_as_req_item()
        if i % 2000 == 0:
            print(cache)
        i += 1
