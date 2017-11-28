# coding=utf-8
"""
this version is more efficient than Randomv0 which uses punching hole method, but _evictElement is not callable
any more
"""

import random
from mimircache.cache.abstractCache import cache


class Random(cache):
    def __init__(self, cache_size=1000, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.cache_set = set()  # key -> linked list node (in reality, it should also contains value)
        self.cache_line_list = [] # to save all the keys, otherwise needs to populate from cache_set every time

    def check_element(self, element):
        """
        :param element: the key of cache request
        :return: whether the given key is in the cache or not
        """
        if element in self.cache_set:
            return True
        else:
            return False

    def _update_element(self, element):
        """ the given element is in the cache, when it is requested again,
         usually we need to update it to new location, but in random, we don't need to do that
        :param element: the key of cache request
        :return: None
        """

        pass

    def _insert_element(self, element):
        """
        the given element is not in the cache, now insert it into cache
        :param element: the key of cache request
        :return: None
        """

        if len(self.cache_set) >= self.cache_size:
            rand_num = random.randrange(0, len(self.cache_line_list))
            evicted = self.cache_line_list[rand_num]
            self.cache_line_list[rand_num] = element
            self.cache_set.remove(evicted)
        else:
            self.cache_line_list.append(element)

        self.cache_set.add(element)

    def _printCacheLine(self):
        for i in self.cache_set:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)

        print(' ')

    def _evict_one_element(self):
        """
        not used
        :return:
        """
        pass

    def add_element(self, element):
        """
        :param element: the key of cache request, it can be in the cache, or not in the cache
        :return: True if element in the cache
        """
        if self.check_element(element):
            self._update_element(element)
            return True
        else:
            self._insert_element(element)
            return False

    def __repr__(self):
        return "Random Replacement, given size: {}, current size: {}".\
            format(self.cache_size, len(self.cache_set), super().__repr__())
