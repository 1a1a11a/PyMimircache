import sys
import os

from mimircache.cache.abstractCache import cache
from mimircache.cache.LRU import LRU
from mimircache.utils.LinkedList import LinkedList


class S4LRU(cache):
    def __init__(self, cache_size=1000):
        '''
        add the fourth part first, then gradually goes up to third, second and first level,
        final eviction is from fourth part
        :param cache_size: size of cache
        :return:
        '''
        super(S4LRU, self).__init__(cache_size)

        # Maybe use four linkedlist and a dict will be more efficient?
        self.firstLRU = LRU(self.cache_size // 4)
        self.secondLRU = LRU(self.cache_size // 4)
        self.thirdLRU = LRU(self.cache_size // 4)
        self.fourthLRU = LRU(self.cache_size // 4)

    def checkElement(self, element):
        '''
        :param content: the content for search
        :return: whether the given element is in the cache
        '''
        if element in self.firstLRU or element in self.secondLRU or \
                        element in self.thirdLRU or element in self.fourthLRU:
            return True
        else:
            return False

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: None
        '''
        if element in self.firstLRU:
            self.firstLRU._updateElement(element)
        elif element in self.secondLRU:
            # element is in second, remove from second, insert to end of first,
            # evict from first to second if needed
            self._move_to_upper_level(element, self.secondLRU, self.firstLRU)
        elif element in self.thirdLRU:
            self._move_to_upper_level(element, self.thirdLRU, self.secondLRU)
        elif element in self.fourthLRU:
            self._move_to_upper_level(element, self.fourthLRU, self.thirdLRU)

    def _move_to_upper_level(self, element, lowerLRU, upperLRU):
        '''
        move the element from lowerLRU to upperLRU, remove the element from lowerLRU,
        insert into upperLRU, if upperLRU is full, evict one into lowerLRU
        :param element: move the element from lowerLRU to upperLRU
        :param lowerLRU: element in lowerLRU is easier to be evicted
        :param upperLRU: element in higherLRU is evicted into lowerLRU first
        :return:
        '''

        # get the node and remove from lowerLRU
        node = lowerLRU.cacheDict[element]
        lowerLRU.cacheLinkedList.removeNode(node)
        del lowerLRU.cacheDict[element]

        # insert into upperLRU
        evicted_key = upperLRU._insertElement(node.content)

        # if there are element evicted from upperLRU, add to lowerLRU
        if evicted_key:
            lowerLRU._insertElement(evicted_key)

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: evicted element
        '''
        return self.fourthLRU._insertElement(element)

    def printCacheLine(self):
        print("first: ")
        self.firstLRU.printCacheLine()
        print("second: ")
        self.secondLRU.printCacheLine()
        print("third: ")
        self.thirdLRU.printCacheLine()
        print("fourth: ")
        self.fourthLRU.printCacheLine()

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        pass

    def addElement(self, element):
        '''
        :param element: a cache request, it can be in the cache, or not
        :return: None
        '''
        if self.checkElement(element):
            self._updateElement(element)
            return True
        else:
            self._insertElement(element)
            return False

    def __repr__(self):
        return "S4LRU, given size: {}, current 1st part size: {}, current 2nd size: {}, \
            current 3rd part size: {}, current fourth part size: {}". \
            format(self.cache_size, self.firstLRU.cacheLinkedList.size, self.secondLRU.cacheLinkedList.size,
                   self.thirdLRU.cacheLinkedList.size, self.fourthLRU.cacheLinkedList.size)
