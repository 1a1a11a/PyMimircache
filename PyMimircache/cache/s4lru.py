# coding=utf-8
from PyMimircache.cache.lru import LRU
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class S4LRU(Cache):
    def __init__(self, cache_size=1000, **kwargs):
        """
        add the fourth part first, then gradually goes up to third, second and first level,
        final eviction is from fourth part
        :param cache_size: size of cache
        :return:
        """
        super(S4LRU, self).__init__(cache_size, **kwargs)
        raise RuntimeError("S4LRU needs re-implementation")

        # Maybe use four linkedlist and a dict will be more efficient?
        self.first_lru = LRU(self.cache_size // 4)
        self.second_lru = LRU(self.cache_size // 4)
        self.third_lru = LRU(self.cache_size // 4)
        self.fourth_lru = LRU(self.cache_size // 4)

    def has(self, obj_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        return obj_id in self.first_lru or obj_id in self.second_lru or \
                obj_id in self.third_lru or obj_id in self.fourth_lru

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param obj_id:
        :return: None
        """
        if obj_id in self.first_lru:
            self.first_lru._update(obj_id, )
        elif obj_id in self.second_lru:
            # obj_id is in second, remove from second, insert to end of first,
            # evict from first to second if needed
            self._move_to_upper_level(
                obj_id, self.second_lru, self.first_lru)
        elif obj_id in self.third_lru:
            self._move_to_upper_level(
                obj_id, self.third_lru, self.second_lru)
        elif obj_id in self.fourth_lru:
            self._move_to_upper_level(
                obj_id, self.fourth_lru, self.third_lru)

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

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id:
        :return: evicted element
        """
        return self.fourth_lru._insert(obj_id, )

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
        :param obj_id: a cache request, it can be in the cache, or not
        :return: None
        """

        obj_id = req_item
        if isinstance(req_item, Req):
            obj_id = req_item.obj_id

        if self.has(obj_id, ):
            self._update(obj_id, )
            return True
        else:
            self._insert(obj_id, )
            return False

    def __repr__(self):
        return "S4LRU, given size: {}, current 1st part size: {}, current 2nd size: {}, \
            current 3rd part size: {}, current fourth part size: {}". \
            format(self.cache_size, self.first_lru.cache_linked_list.size, self.second_lru.cache_linked_list.size,
                   self.third_lru.cache_linked_list.size, self.fourth_lru.cache_linked_list.size)
