import sys
import os

from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList


class LRU(cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.cacheLinkedList = LinkedList()
        self.cacheDict = dict()  # key -> linked list node (in reality, it should also contains value)

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

        node = self.cacheDict[element]
        self.cacheLinkedList.moveNodeToTail(node)

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: evicted element or None
        '''
        return_content = None
        if self.cacheLinkedList.size >= self.cache_size:
            # print("{}: {}".format(self.cacheLinkedList.size, self.cache_size))
            # if self.cacheLinkedList.size == self.cache_size:
            #     self.printCacheLine()
            # print(self.cacheDict)
            return_content = self._evictOneElement()

        node = self.cacheLinkedList.insertAtTail(element)
        self.cacheDict[element] = node
        return return_content

    def printCacheLine(self):
        for i in self.cacheLinkedList:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)

        print(' ')

    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: content of evicted element
        '''
        content = self.cacheLinkedList.removeFromHead()
        # print(content)
        # self.printCacheLine()
        # print(self.cacheDict)
        del self.cacheDict[content]
        return content

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: None
        '''
        if self.checkElement(element):
            self._updateElement(element)
            if len(self.cacheDict) != self.cacheLinkedList.size:
                print("1*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(self.cacheLinkedList.size, len(self.cacheDict)))
                import sys
                sys.exit(-1)
            return True
        else:
            self._insertElement(element)
            if len(self.cacheDict) != self.cacheLinkedList.size:
                print("2*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(self.cacheLinkedList.size, len(self.cacheDict)))
                import sys
                sys.exit(-1)
            return False

    def __repr__(self):
        return "LRU, given size: {}, current size: {}, {}".format(self.cache_size, self.cacheLinkedList.size,
                                                                  super().__repr__())
