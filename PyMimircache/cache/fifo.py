# coding=utf-8
from PyMimircache.cache.lru import LRU


class FIFO(LRU):
    def __init__(self, cache_size=1000, **kwargs):
        super().__init__(cache_size, **kwargs)

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param req_item:
        :return: None
        """
        pass

    def __repr__(self):
        return "FIFO cache of size {}, current size: {}, {}".format(
            self.cache_size, self.cache_linked_list.size, super().__repr__())
