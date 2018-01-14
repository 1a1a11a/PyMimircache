# coding=utf-8
from PyMimircache.cache.lru import LRU


class Clock(LRU):
    """
    second chance page replacement algorithm
    """

    def __init__(self, cache_size=1000, **kwargs):
        # use node id to represent the reference bit
        super(Clock, self).__init__(cache_size, **kwargs)
        self.hand = None  # points to the node for examination/eviction

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it
        :param **kwargs:
        :param req_item:
        :return: None
        """
        node = self.cacheDict[req_item]
        node.id = 1

    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        if self.cacheLinkedList.size >= self.cache_size:
            self.evict()

        node = self.cacheLinkedList.insert_at_tail(req_item, id=1)
        self.cacheDict[req_item] = node
        if not self.hand:
            # this is the first req_item
            assert self.cacheLinkedList.size == 1, "insert req_item error"
            self.hand = node

    def _printCacheLine(self):
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
            while node.id == 1:
                node.set_id(0)
                node = node.next
                if not node:
                    # tail
                    node = self.cacheLinkedList.head.next
            self.hand = node.next
            return node

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        node = self._find_evict_node()
        self.cacheLinkedList.remove_node(node)
        del self.cacheDict[node.content]

        return True

    def access(self, req_item, **kwargs):
        """
        :param **kwargs: 
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: None
        """
        if self.has(req_item, ):
            self._update(req_item, )
            # self.printCacheLine()
            if len(self.cacheDict) != self.cacheLinkedList.size:
                print("1*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(self.cacheLinkedList.size, len(self.cacheDict)))
                import sys
                sys.exit(-1)
            return True
        else:
            self._insert(req_item, )
            # self.printCacheLine()
            if len(self.cacheDict) != self.cacheLinkedList.size:
                print("2*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(self.cacheLinkedList.size, len(self.cacheDict)))
                import sys
                sys.exit(-1)
            return False

    def __repr__(self):
        return "second chance cache, given size: {}, current size: {}, {}".format(
            self.cache_size, self.cacheLinkedList.size, super().__repr__())
