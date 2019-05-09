# coding=utf-8
import abc
from collections import deque

from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class LFU(Cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.cache_dict = dict()  # key -> freq
        self.ts = 0
        self.least_freq = 1
        self.least_freq_list = deque()
        self.least_freq_set = set()

    def has(self, obj_id, **kwargs):
        """
        :param obj_id:
        :param kwargs:
        :return: whether the given element is in the cache
        """
        if obj_id in self.cache_dict:
            return True
        else:
            return False

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, now update its frequency
        :param **kwargs:
        :param obj_id:
        :return: original rank
        """
        self.cache_dict[obj_id] += 1
        if obj_id in self.least_freq_set:
            if len(self.least_freq_set) > 1:
                # more than one obj_id, so just remove this obj_id
                self.least_freq_set.remove(obj_id)
                # this is not O(1) operation
                self.least_freq_list.remove(obj_id)
            else:
                # this is the only obj_id, keep it in the set/list, add more with same freq
                self.least_freq = self.cache_dict[obj_id]
                for key, value in self.cache_dict.items():
                    if value == self.least_freq and key != obj_id:
                        self.least_freq_set.add(key)
                        self.least_freq_list.append(key)
                    elif value < self.least_freq:
                        raise RuntimeError()

        # self.cache_dict.move_to_end(obj_id)

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id:
        :return: True on success, False on failure
        """

        # if len(self.cache_dict) > self.cache_size:
        #     self.evict()

        self.cache_dict[obj_id] = 1
        if self.least_freq > 1:
            self.least_freq = 1
            self.least_freq_list.clear()
            self.least_freq_set.clear()

        self.least_freq_list.append(obj_id)
        self.least_freq_set.add(obj_id)
        print("after insert {}".format(len(self.least_freq_list)))

    def find_evict_key(self):
        evict_key = self.least_freq_list.popleft()
        return evict_key


    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        evict_key = self.find_evict_key()
        self.least_freq_set.remove(evict_key)
        del self.cache_dict[evict_key]

        while len(self.least_freq_list) == 0:
            if len(self.cache_dict) == 0:
                raise RuntimeError("size error")
            # after remove, there is no element in the least frequent set
            self.least_freq += 1
            for key, value in self.cache_dict.items():
                if value == self.least_freq:
                    self.least_freq_list.append(key)
                    self.least_freq_set.add(key)

    def access(self, req_item, **kwargs):
        """
        :param obj_id: the element in the reference, it can be in the cache, or not
        :param kwargs:
        :return: -1 if not in cache, otherwise old rank
        """

        if self.ts % 2000 == 0:
            print("LFU {}".format(self.ts))

        print("access {} {} {}".format(self.least_freq, len(self.least_freq_list), len(self.cache_dict)))

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
    from PyMimircache import CsvReader

    reader = CsvReader("/home/jason/data/SmartCacheNew/largeData/akamai1TrafficID458", init_params={"header": False, "delimiter": "\t", "label": 5, 'real_time': 1})

    cache = LFU(cache_size=20000)
    i = 0
    req = reader.read_as_req_item()
    while req:
        hit = cache.access(req, )
        req = reader.read_as_req_item()
        if i % 2000 == 0:
            print(cache)
        i += 1
