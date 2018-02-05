# coding=utf-8
from PyMimircache.cache.lru import LRU
from PyMimircache.cache.abstractCache import Cache


class S4LRU(Cache):
    def __init__(self, cache_size=1000, **kwargs):
        """
        add the fourth part first, then gradually goes up to third, second and first level,
        final eviction is from fourth part
        :param cache_size: size of cache
        :return:
        """
        super(S4LRU, self).__init__(cache_size, **kwargs)

        # Maybe use four linkedlist and a dict will be more efficient?
        self.first_lru = LRU(self.cache_size // 4)
        self.second_lru = LRU(self.cache_size // 4)
        self.third_lru = LRU(self.cache_size // 4)
        self.fourth_lru = LRU(self.cache_size // 4)

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        if req_id in self.first_lru or req_id in self.second_lru or \
                req_id in self.third_lru or req_id in self.fourth_lru:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        :return: None
        """
        if req_item in self.first_lru:
            self.first_lru._update(req_item, )
        elif req_item in self.second_lru:
            # req_item is in second, remove from second, insert to end of first,
            # evict from first to second if needed
            self._move_to_upper_level(
                req_item, self.second_lru, self.first_lru)
        elif req_item in self.third_lru:
            self._move_to_upper_level(
                req_item, self.third_lru, self.second_lru)
        elif req_item in self.fourth_lru:
            self._move_to_upper_level(
                req_item, self.fourth_lru, self.third_lru)

    def _move_to_upper_level(self, element, lowerLRU, upperLRU):
        """
        move the element from lowerLRU to upperLRU, remove the element from lowerLRU,
        insert into upperLRU, if upperLRU is full, evict one into lowerLRU
        :param element: move the element from lowerLRU to upperLRU
        :param lowerLRU: element in lowerLRU is easier to be evicted
        :param upperLRU: element in higherLRU is evicted into lowerLRU first
        :return:
        """

        # get the node and remove from lowerLRU
        node = lowerLRU.cache_dict[element]
        lowerLRU.cache_linked_list.remove_node(node)
        del lowerLRU.cache_dict[element]

        # insert into upperLRU
        evicted_key = upperLRU._insert(node.content, )

        # if there are element evicted from upperLRU, add to lowerLRU
        if evicted_key:
            lowerLRU._insert(evicted_key, )

    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: evicted element
        """
        return self.fourth_lru._insert(req_item, )

    def _print_cache_line(self):
        print("first: ")
        self.first_lru._print_cache_line()
        print("second: ")
        self.second_lru._print_cache_line()
        print("third: ")
        self.third_lru._print_cache_line()
        print("fourth: ")
        self.fourth_lru._print_cache_line()

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        pass

    def access(self, req_item, **kwargs):
        """
        :param **kwargs: 
        :param req_item: a cache request, it can be in the cache, or not
        :return: None
        """
        if self.has(req_item, ):
            self._update(req_item, )
            return True
        else:
            self._insert(req_item, )
            return False

    def __repr__(self):
        return "S4LRU, given size: {}, current 1st part size: {}, current 2nd size: {}, \
            current 3rd part size: {}, current fourth part size: {}". \
            format(self.cache_size, self.first_lru.cache_linked_list.size, self.second_lru.cache_linked_list.size,
                   self.third_lru.cache_linked_list.size, self.fourth_lru.cache_linked_list.size)
