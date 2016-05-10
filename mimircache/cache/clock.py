import sys
import os

from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList
from mimircache.cache.LRU import LRU


class clock(LRU):
    '''
    second chance page replacement algorithm
    '''

    def __init__(self, cache_size=1000):
        # use node id to represent the reference bit
        super(clock, self).__init__(cache_size)
        self.hand = None  # points to the node for examination/eviction

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it
        :param element:
        :return: None
        '''
        node = self.cacheDict[element]
        node.id = 1

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        if self.cacheLinkedList.size >= self.cache_size:
            self._evictOneElement()

        node = self.cacheLinkedList.insertAtTail(element, id=1)
        self.cacheDict[element] = node
        if not self.hand:
            # this is the first element
            assert self.cacheLinkedList.size == 1, "insert element error"
            self.hand = node

    def printCacheLine(self):
        for i in self.cacheLinkedList:
            try:
                print("{}({})".format(i.content, i.id), end='\t')
            except:
                print("{}({})".format(i.content, i.id))

        print(' ')

    def _find_evict_node(self):
        node = self.hand
        if node.id == 0:
            self.hand = node.next
            if not self.hand:
                # tail
                self.hand = self.cacheLinkedList.next
            return node
        else:
            # set reference bit to 0
            while (node.id == 1):
                node.set_id(0)
                node = node.next
                if not node:
                    # tail
                    node = self.cacheLinkedList.head.next
            self.hand = node.next
            return node

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        node = self._find_evict_node()
        self.cacheLinkedList.removeNode(node)
        del self.cacheDict[node.content]

        return True

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: None
        '''
        if self.checkElement(element):
            self._updateElement(element)
            # self.printCacheLine()
            if len(self.cacheDict) != self.cacheLinkedList.size:
                print("1*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(self.cacheLinkedList.size, len(self.cacheDict)))
                import sys
                sys.exit(-1)
            return True
        else:
            self._insertElement(element)
            # self.printCacheLine()
            if len(self.cacheDict) != self.cacheLinkedList.size:
                print("2*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(self.cacheLinkedList.size, len(self.cacheDict)))
                import sys
                sys.exit(-1)
            return False

    def __repr__(self):
        return "second chance cache, given size: {}, current size: {}, {}".format( \
            self.cache_size, self.cacheLinkedList.size, super().__repr__())
