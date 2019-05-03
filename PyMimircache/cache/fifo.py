# coding=utf-8
from PyMimircache.cache.lru import LRU


class FIFO(LRU):
    def __init__(self, cache_size=1000, **kwargs):
        super().__init__(cache_size, **kwargs)

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, now update it to new location
        :param **kwargs:
        :param obj_id:
        :return: None
        """
        pass

    def __repr__(self):
        return "FIFO cache of size {}, current size: {}".format(
            self.cache_size, self.get_size())
