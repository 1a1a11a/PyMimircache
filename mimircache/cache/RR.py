import sys
import os
import random

from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList


class RR(cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.cacheDict = dict()  # key -> linked list node (in reality, it should also contains value)
        self.cache_line_list = []  # to save all the keys, otherwise needs to populate from cacheDict every time


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
        if len(self.cacheDict) >= self.cache_size:
            self._evictOneElement()
        self.cacheDict[element] = ""
        self.cache_line_list.append(element)

    def printCacheLine(self):
        for i in self.cacheSet:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)

        print(' ')

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        # element = random.choice(list(self.cacheDict.keys()))
        rand_num = random.randrange(0, len(self.cache_line_list))
        element = self.cache_line_list[rand_num]
        count = 0
        while not element:
            rand_num = random.randrange(0, len(self.cache_line_list))
            element = self.cache_line_list[rand_num]
            count += 1

        # mark this element as deleted, put a hole on it
        self.cache_line_list[rand_num] = None

        if (count > 10):
            # if there are too many holes, then we need to resize the list
            new_list = [e for e in self.cache_line_list if e]
            del self.cache_line_list
            self.cache_line_list = new_list

        del self.cacheDict[element]

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: None
        '''
        if self.checkElement(element):
            self._updateElement(element)
            return True
        else:
            self._insertElement(element)
            return False

    def __repr__(self):
        return "Random Replacement, given size: {}, current size: {}".format(self.cache_size,
                                                                             len(self.cacheDict),
                                                                             super().__repr__())
