import sys
import os
import random

from mimircache.cache.abstractCache import cache


class optimal(cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.seenset = set()

    def checkElement(self, element):
        '''
        :param content: the content for search
        :return: whether the given element is in the cache
        '''
        if element in self.seenset:
            return True
        else:
            return False

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: None
        '''

        pass

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        self.seenset.add(element)

    def printCacheLine(self):
        for i in self.seenset:
            print(i, end='\t')

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        # element = random.choice(list(self.cacheDict.keys()))
        pass

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: True if element in the cache
        '''
        if self.checkElement(element):
            return True
        else:
            self._insertElement(element)
            return False

    def __repr__(self):
        return "Optimal Cache, current size: {}".format(self.cache_size, len(self.seenset),
                                                        super().__repr__())
