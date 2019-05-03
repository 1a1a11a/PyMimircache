# coding=utf-8
"""
this version is more efficient than Randomv0 which uses punching hole method, but _evictElement is not callable
any more
"""

import random
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class Random(Cache):
    def __init__(self, cache_size=1000, **kwargs):
        super().__init__(cache_size, **kwargs)
        # key -> linked list node (in reality, it should also contains value)
        self.cache_set = set()
        # to save all the keys, otherwise needs to populate from cache_set every time
        self.cache_line_list = []

    def has(self, obj_id, **kwargs):
        """
        :param **kwargs:
        :param req_id: the key of cache request
        :return: whether the given key is in the cache or not
        """
        return obj_id in self.cache_set

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, when it is requested again,
         usually we need to update it to new location, but in random, we don't need to do that
        :param **kwargs:
        :param obj_id: the key of cache request
        :return: None
        """

        pass

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id: the key of cache request
        :return: None
        """

        if len(self.cache_set) >= self.cache_size:
            rand_num = random.randrange(0, len(self.cache_line_list))
            evicted = self.cache_line_list[rand_num]
            self.cache_line_list[rand_num] = obj_id
            self.cache_set.remove(evicted)
        else:
            self.cache_line_list.append(obj_id)

        self.cache_set.add(obj_id)

    def _print_cache_line(self):
        for i in self.cache_set:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)

        print(' ')

    def evict(self, **kwargs):
        """
        not used
        :param **kwargs:
        :return:
        """
        pass

    def access(self, req_item, **kwargs):
        """
        :param **kwargs: 
        :param obj_id: the key of cache request, it can be in the cache, or not in the cache
        :return: True if element in the cache
        """

        obj_id = req_item
        if isinstance(req_item, Req):
            obj_id = req_item.obj_id

        if self.has(obj_id, ):
            self._update(obj_id, )
            return True
        else:
            self._insert(obj_id, )
            return False

    def __repr__(self):
        return "Random Replacement, given size: {}, current size: {}".\
            format(self.cache_size, len(self.cache_set), super().__repr__())
