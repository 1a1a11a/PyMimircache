import sys
import os

import abc
from collections import deque
from mimircache.cache.abstractCache import cache


class abstractLFU(cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.cacheDict = dict()  # key -> freq
        self.largest_freq = -1
        self.largest_freq_element = None
        self.least_freq = 1
        self.least_freq_elements_list = deque()
        self.least_freq_elements_set = set()

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
        self.cacheDict[element] += 1
        # if self.cacheDict[element]>self.largest_freq:
        #     self.largest_freq = self.cacheDict[element]
        #     self.largest_freq_element = element
        if element in self.least_freq_elements_set:
            if len(self.least_freq_elements_set) > 1:
                # more than one element, so just remove this element
                self.least_freq_elements_set.remove(element)
                self.least_freq_elements_list.remove(element)
            else:
                # this is the only element, keep it in the set/list, add more with same freq
                for key, value in self.cacheDict.items():
                    if value == self.cacheDict[element] and key != element:
                        self.least_freq_elements_set.add(key)
                        self.least_freq_elements_list.append(key)

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        # print("***%s***"%element)
        if len(self.cacheDict) >= self.cache_size:
            self._evictOneElement()

        self.cacheDict[element] = 1
        # if self.cacheDict[element]>self.largest_freq:
        #     self.largest_freq = self.cacheDict[element]
        #     self.largest_freq_element = element
        if self.least_freq == 1:
            # this one needs to be added
            self.least_freq_elements_list.append(element)
            self.least_freq_elements_set.add(element)
        elif self.least_freq > 1:
            self.least_freq = 1
            self.least_freq_elements_list.clear()
            self.least_freq_elements_set.clear()

            self.least_freq_elements_list.append(element)
            self.least_freq_elements_set.add(element)

    @abc.abstractmethod
    def find_evict_key(self):
        pass

    def printCacheLine(self):
        for key, value in self.cacheDict.items():
            print("{}: {}".format(key, value))

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        evict_key = self.find_evict_key()
        self.least_freq_elements_set.remove(evict_key)
        del self.cacheDict[evict_key]
        while len(self.least_freq_elements_list) == 0:
            if len(self.cacheDict) == 0:
                break
            # after remove, there is no element in the least frequent set
            self.least_freq += 1
            for key, value in self.cacheDict.items():
                if value == self.least_freq:
                    self.least_freq_elements_list.append(key)
                    self.least_freq_elements_set.add(key)

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
        return "LFU, given size: {}, current size: {}".format(self.cache_size, len(self.cacheDict))

    def __str__(self):
        return self.__repr__()
