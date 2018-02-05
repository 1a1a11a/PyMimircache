# coding=utf-8
from PyMimircache.cache.lru import LRU
from PyMimircache.cache.abstractCache import Cache


class SLRU(Cache):
    def __init__(self, cache_size=1000, ratio=1, **kwargs):
        """

        :param cache_size: size of cache
        :param args: raio: the ratio of protected/probationary
        :return:
        """
        super().__init__(cache_size, **kwargs)
        self.ratio = ratio
        # Maybe use two linkedlist and a dict will be more efficient?
        self.protected = LRU(
            int(self.cache_size * self.ratio / (self.ratio + 1)))
        self.probationary = LRU(int(self.cache_size * 1 / (self.ratio + 1)))

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        if req_id in self.protected or req_id in self.probationary:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        :return: None
        """
        if req_item in self.protected:
            self.protected._update(req_item, )
        else:
            # req_item is in probationary, remove from probationary, insert to end of protected,
            # evict from protected to probationary if needed

            # get the node and remove from probationary
            node = self.probationary.cache_dict[req_item]
            self.probationary.cache_linked_list.remove_node(node)
            del self.probationary.cache_dict[req_item]

            # insert into protected
            evicted_key = self.protected._insert(node.content, )

            # if there are req_item evicted from protected area, add to probationary area
            if evicted_key:
                self.probationary._insert(evicted_key, )

    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: evicted element
        """
        return self.probationary._insert(req_item, )

    def _print_cache_line(self):
        print("protected: ")
        self.protected._print_cache_line()
        print("probationary: ")
        self.probationary._print_cache_line()

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
        return "SLRU, given size: {}, given protected part size: {}, given probationary part size: {}, \
            current protected part size: {}, current probationary size: {}". \
            format(self.cache_size, self.protected.cache_size, self.probationary.cache_size,
                   self.protected.cache_linked_list.size, self.probationary.cache_linked_list.size)
