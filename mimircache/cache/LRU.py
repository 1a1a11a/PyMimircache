# coding=utf-8
from mimircache.cache.abstractCache import cache
from mimircache.utils.LinkedList import LinkedList


class LRU(cache):
    def __init__(self, cache_size=1000, **kwargs):

        super().__init__(cache_size, **kwargs)
        self.cacheLinkedList = LinkedList()
        self.cacheDict = dict()  # key -> linked list node (in reality, it should also contains value)

    def __len__(self):
        return len(self.cacheDict)

    def check_element(self, element):
        """
        :param element:
        :return: whether the given element is in the cache
        """
        if element in self.cacheDict:
            return True
        else:
            return False

    def _update_element(self, element):
        """ the given element is in the cache, now update it to new location
        :param element:
        :return: None
        """

        node = self.cacheDict[element]
        self.cacheLinkedList.moveNodeToTail(node)

    def _insert_element(self, element):
        """
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: evicted element or None
        """
        return_content = None
        node = self.cacheLinkedList.insertAtTail(element)
        self.cacheDict[element] = node
        return return_content

    def _printCacheLine(self):
        for i in self.cacheLinkedList:
            try:
                print(i.content, end='\t')
            except:
                print(i.content)

        print(' ')

    def _evict_one_element(self):
        """
        evict one element from the cache
        :return: content of evicted element
        """

        content = self.cacheLinkedList.removeFromHead()
        del self.cacheDict[content]
        return content

    def add_element(self, element):
        """
        :param element: the element in the trace, it can be in the cache, or not
        :return: None
        """
        if self.check_element(element):
            self._update_element(element)
            return True
        else:
            self._insert_element(element)
            if self.cacheLinkedList.size > self.cache_size:
                self._evict_one_element()
            return False

    def __repr__(self):
        return "LRU cache of size: {}, current size: {}, {}".\
            format(self.cache_size, self.cacheLinkedList.size, super().__repr__())
