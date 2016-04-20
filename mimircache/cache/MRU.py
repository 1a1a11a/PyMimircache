import sys
import os

import abc
from collections import deque
from mimircache.cache.abstractCache import cache


class MRU(cache):
    def __init__(self, cache_size=1000):
        super(MRU, self).__init__(cache_size)
        self.cacheDict = dict()
        self.last_element = None

    def checkElement(self, element):
        '''
        :param content: the content for search
        :return: whether the given element is in the cache
        '''
        if element in self.cacheDict:
            return True
        else:
            return False

    def _updateElement(self, element):
        ''' the given element is in the cache, now its frequency
        :param element:
        :return: original rank
        '''
        self.last_element = element

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        if len(self.cacheDict) == self.cache_size:
            self._evictOneElement()
        self.cacheDict[element] = element
        self.last_element = element

    def find_evict_key(self):
        return self.last_element

    def printCacheLine(self):
        for key, value in self.cacheDict.items():
            print("{}: {}".format(key, value))

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        evict_key = self.find_evict_key()
        del self.cacheDict[evict_key]

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank
        '''
        # print(element, end=': \t')
        if self.checkElement(element):
            self._updateElement(element)
            return True
        else:
            self._insertElement(element)
            return False

    def __repr__(self):
        return "MRU, given size: {}, current size: {}".format(self.cache_size, len(self.cacheDict))

    def __str__(self):
        return self.__repr__()
