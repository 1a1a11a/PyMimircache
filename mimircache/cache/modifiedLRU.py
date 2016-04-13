import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import math

from mimircache.utils.LinkedList import LinkedList


class modifiedLRU():
    def __init__(self, size=1000):
        self.max_size = size
        # this size includes holes, if want real element size use self.cacheLinkedList.size
        self.size = 0
        # self.cache   = OrderedDict()
        self.cacheLinkedList = LinkedList()
        self.cacheDict = dict()
        self.headPos = 1
        self.tailPos = 1
        self.numOfHoles = 0
        self.holePosSet = set()

    def checkElement(self, content):
        '''
        :param content: the content for search
        :return: whether the given element is in the cache
        '''
        if content in self.cacheDict:
            return True
        else:
            return False

    def getRank(self, content):
        '''

        :param content: the content of element, which is also the key in cacheDict
        :return: rank if in the cache, otherwise -1
        '''
        return self.cacheDict.get(content, (-1, -1))[0]

    def __updateElement__(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: original rank
        '''
        rank = self.cacheDict.get(element, (-1, -1))[0] - self.headPos  # this may have holes
        rank = rank - math.floor(rank / self.size * self.numOfHoles)  # adjust
        if (rank < 0):
            print("waht, rank<0???\t" + str(self.cacheDict.get(element, (-1, -1))[0]) + '\t' + str(self.headPos) + '\t')
            print(str(rank) + "\t" + str(self.cacheDict.get(element, (-1, -1))[0] - self.headPos) + '\t'
                  + str(self.size) + '\t' + str(self.numOfHoles))

        self.headPos -= 1
        self.cacheDict[element][0] = self.headPos
        self.numOfHoles += 1
        self.size += 1
        self.holePosSet.add(rank)

        self.cacheLinkedList.moveNodeToHead(self.cacheDict[element][1])
        return rank

    def __insertElement__(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        node = self.cacheLinkedList.insertAtHead(element)
        self.headPos -= 1
        self.cacheDict[element] = [self.headPos, node]
        self.size += 1

        if self.cacheLinkedList.size > self.max_size:
            self.__evictOneElement__()
            self.tailPos -= 1

    def printLinkedList(self):
        for i in self.cacheLinkedList:
            print(i.content)

    def __evictOneElement__(self):
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
        if self.checkElement(element):
            return self.__updateElement__(element)
        else:
            self.__insertElement__(element)
            return -1

    def reArrange(self):
        pass
