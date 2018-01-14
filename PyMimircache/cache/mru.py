# coding=utf-8
from PyMimircache.cache.abstractCache import Cache


class MRU(Cache):
    def __init__(self, cache_size=1000, **kwargs):
        super(MRU, self).__init__(cache_size, **kwargs)
        self.cacheDict = dict()
        self.last_element = None

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        if req_id in self.cacheDict:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now its frequency
        :param **kwargs:
        :param req_item:
        :return: original rank
        """
        self.last_element = req_item

    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        if len(self.cacheDict) == self.cache_size:
            self.evict()
        self.cacheDict[req_item] = req_item
        self.last_element = req_item

    def find_evict_key(self):
        return self.last_element

    def _printCacheLine(self):
        for key, value in self.cacheDict.items():
            print("{}: {}".format(key, value))

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        evict_key = self.find_evict_key()
        del self.cacheDict[evict_key]

    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank
        """
        # print(req_item, end=': \t')
        if self.has(req_item, ):
            self._update(req_item, )
            return True
        else:
            self._insert(req_item, )
            return False

    def __repr__(self):
        return "MRU, given size: {}, current size: {}".format(self.cache_size, len(self.cacheDict))

    def __str__(self):
        return self.__repr__()
