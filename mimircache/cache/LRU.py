import sys
import os

from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList


class LRU(cache):
    def __init__(self, cache_size=1000):
        super().__init__(cache_size)
        self.cacheLinkedList = LinkedList()
        self.cacheDict = dict()  # store the reuse dist, thus dict: content->reuse dist, rank begins from 1

    def checkElement(self, content):
        '''
        :param content: the content for search
        :return: whether the given element is in the cache
        '''
        if content in self.cacheDict:
            return True
        else:
            return False

    def getReuseDist(self, content):
        '''

        :param content: the content of element, which is also the key in cacheDict
        :return: rank if in the cache, otherwise -1
        '''
        return self.cacheDict.get(content, -1)

    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: original rank
        '''
        rank = self.getReuseDist(element)
        # if rank >= 10:
        #     print("WWWWWWWWHHHHHHHHAAAAAAATTTTTTT")
        #     print(len(self.cacheDict))
        #     print(element)
        #     print(self.cacheDict)
        #     self.printLinkedList()
        # even if reuse distance is 0, it still needs at least one cache line
        self.cacheDict[element] = 1
        node = None
        counter = 0
        for node in self.cacheLinkedList:
            # print(node.content, end='\t')
            if node == None:
                print("**************stop**************")
            if node.content == element:
                break
            self.cacheDict[node.content] += 1
            counter += 1
        if counter != rank - 1:
            print("error")
            print(str(rank) + ":\t" + str(counter))
        self.cacheLinkedList.moveNodeToHead(node)
        return rank

    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        self.cacheLinkedList.insertAtHead(element)
        self.cacheDict[element] = 0
        # above is set as 0 because it will increment by 1 in the following loop
        for i in self.cacheDict.keys():
            self.cacheDict[i] += 1
        if self.cacheLinkedList.size > self.cache_size:
            self._evictOneElement()

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
        :return: True on success, False on failure
        '''
        content = self.cacheLinkedList.removeFromTail()
        del self.cacheDict[content]

    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank
        '''
        # print(element, end=': \t')
        if self.checkElement(element):
            rank = self._updateElement(element)
            # print(self.cacheDict)
            # self.printLinkedList()
            return rank
        else:
            self._insertElement(element)
            # print(self.cacheDict)
            # self.printLinkedList()
            return -1

    def __repr__(self):
        return "LRU, size: {}, {}".format(self.cache_size, super().__repr__())
