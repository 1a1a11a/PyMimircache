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
        node = self.cache_dict[req_item]
        node.id = 1

    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        if self.cache_linked_list.size >= self.cache_size:
            self.evict()

        node = self.cache_linked_list.insert_at_tail(req_item, id=1)
        self.cache_dict[req_item] = node
        if not self.hand:
            # this is the first req_item
            assert self.cache_linked_list.size == 1, "insert req_item error"
            self.hand = node

    def _print_cache_line(self):
        for i in self.cache_linked_list:
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
                self.hand = self.cache_linked_list.next
            return node
        else:
            # set reference bit to 0
            while node.id == 1:
                node.set_id(0)
                node = node.next
                if not node:
                    # tail
                    node = self.cache_linked_list.head.next
            self.hand = node.next
            return node

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        node = self._find_evict_node()
        self.cache_linked_list.remove_node(node)
        del self.cache_dict[node.content]

        return True

    def access(self, req_item, **kwargs):
        """
        :param **kwargs: 
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: None
        """
        if self.has(req_item, ):
            self._update(req_item, )
            if len(self.cache_dict) != self.cache_linked_list.size:
                print(
                    "1*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(
                    self.cache_linked_list.size, len(self.cache_dict)))
                import sys
                sys.exit(-1)
            return True
        else:
            self._insert(req_item, )
            if len(self.cache_dict) != self.cache_linked_list.size:
                print(
                    "2*********########### ERROR detected in LRU size #############***********")
                print("{}: {}".format(
                    self.cache_linked_list.size, len(self.cache_dict)))
                import sys
                sys.exit(-1)
            return False

    def __repr__(self):
        return "second chance cache, given size: {}, current size: {}, {}".format(
            self.cache_size, self.cache_linked_list.size, super().__repr__())
