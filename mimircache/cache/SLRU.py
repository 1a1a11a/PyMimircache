import sys
import os

from mimircache.cache.abstractCache import cache
from mimircache.cache.LRU import LRU
from mimircache.utils.LinkedList import LinkedList


class SLRU(cache):
    def __init__(self, cache_size=1000, ratio=1, *args, **kargs):
        '''

        :param cache_size: size of cache
        :param args: raio: the ratio of protected/probationary
        :return:
        '''
        super().__init__(cache_size)
        self.ratio = ratio
        # Maybe use two linkedlist and a dict will be more efficient?
        self.protected = LRU(int(self.cache_size * self.ratio / (self.ratio + 1)))
        self.probationary = LRU(int(self.cache_size * 1 / (self.ratio + 1)))

    def checkElement(self, element):
        '''
        :param content: the content for search
        :return: whether the given element is in the cache
        '''
        if element in self.protected or element in self.probationary:
            return True
        else:
            return False

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: None
        '''
        if element in self.protected:
            self.protected._updateElement(element)
        else:
            # element is in probationary, remove from probationary, insert to end of protected,
            # evict from protected to probationary if needed

            # get the node and remove from probationary
            node = self.probationary.cacheDict[element]
            self.probationary.cacheLinkedList.removeNode(node)
            del self.probationary.cacheDict[element]

            # insert into protected
            evicted_key = self.protected._insertElement(node.content)

            # if there are element evicted from protected area, add to probationary area
            if evicted_key:
                self.probationary._insertElement(evicted_key)

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: evicted element
        '''
        return self.probationary._insertElement(element)

    def printCacheLine(self):
        print("protected: ")
        self.protected.printCacheLine()
        print("probationary: ")
        self.probationary.printCacheLine()

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
            # self.printCacheLine()
            return True
        else:
            self._insertElement(element)
            # self.printCacheLine()
            return False

    def __repr__(self):
        return "SLRU, given size: {}, given protected part size: {}, given probationary part size: {}, \
            current protected part size: {}, current probationary size: {}". \
            format(self.cache_size, self.protected.cache_size, self.probationary.cache_size,
                   self.protected.cacheLinkedList.size, self.probationary.cacheLinkedList.size)
