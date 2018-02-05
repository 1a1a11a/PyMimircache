# coding=utf-8
"""
this version is more efficient than Randomv0 which uses punching hole method, but _evictElement is not callable
any more
"""

import random
from PyMimircache.cache.abstractCache import Cache


class Random(Cache):
    def __init__(self, cache_size=1000, **kwargs):
        super().__init__(cache_size, **kwargs)
        # key -> linked list node (in reality, it should also contains value)
        self.cache_set = set()
        # to save all the keys, otherwise needs to populate from cache_set every time
        self.cache_line_list = []

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id: the key of cache request
        :return: whether the given key is in the cache or not
        """
        if req_id in self.cache_set:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, when it is requested again,
         usually we need to update it to new location, but in random, we don't need to do that
        :param **kwargs:
        :param req_item: the key of cache request
        :return: None
        """

        pass

    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item: the key of cache request
        :return: None
        """

        if len(self.cache_set) >= self.cache_size:
            rand_num = random.randrange(0, len(self.cache_line_list))
            evicted = self.cache_line_list[rand_num]
            self.cache_line_list[rand_num] = req_item
            self.cache_set.remove(evicted)
        else:
            self.cache_line_list.append(req_item)

        self.cache_set.add(req_item)

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
        :param req_item: the key of cache request, it can be in the cache, or not in the cache
        :return: True if element in the cache
        """
        if self.has(req_item, ):
            self._update(req_item, )
            return True
        else:
            self._insert(req_item, )
            return False

    def __repr__(self):
        return "Random Replacement, given size: {}, current size: {}".\
            format(self.cache_size, len(self.cache_set), super().__repr__())
