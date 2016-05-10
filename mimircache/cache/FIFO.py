import sys
import os

from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList
from mimircache.cache.LRU import LRU


class FIFO(LRU):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: None
        '''
        pass

    def __repr__(self):
        return "FIFO, given size: {}, current size: {}, {}".format( \
            self.cache_size, self.cacheLinkedList.size, super().__repr__())
