# coding=utf-8
from mimircache.cache.abstractCache import cache


class MRU(cache):
    def __init__(self, cache_size=1000, **kwargs):
        super(MRU, self).__init__(cache_size, **kwargs)
        self.cacheDict = dict()
        self.last_element = None

    def check_element(self, element):
        """
        :param element:
        :return: whether the given element is in the cache
        """
        if element in self.cacheDict:
            return True
        else:
            return False

    def _update_element(self, element):
        """ the given element is in the cache, now its frequency
        :param element:
        :return: original rank
        """
        self.last_element = element

    def _insert_element(self, element):
        """
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        """
        if len(self.cacheDict) == self.cache_size:
            self._evict_one_element()
        self.cacheDict[element] = element
        self.last_element = element

    def find_evict_key(self):
        return self.last_element

    def _printCacheLine(self):
        for key, value in self.cacheDict.items():
            print("{}: {}".format(key, value))

    def _evict_one_element(self):
        """
        evict one element from the cache line
        :return: True on success, False on failure
        """
        evict_key = self.find_evict_key()
        del self.cacheDict[evict_key]

    def add_element(self, element):
        """
        :param element: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank
        """
        # print(element, end=': \t')
        if self.check_element(element):
            self._update_element(element)
            return True
        else:
            self._insert_element(element)
            return False

    def __repr__(self):
        return "MRU, given size: {}, current size: {}".format(self.cache_size, len(self.cacheDict))

    def __str__(self):
        return self.__repr__()
