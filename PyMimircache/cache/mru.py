# coding=utf-8
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class MRU(Cache):
    def __init__(self, cache_size=1000, **kwargs):
        super(MRU, self).__init__(cache_size, **kwargs)
        self.cache_dict = dict()
        self.last_element = None

    def has(self, req_id, **kwargs):
        """
        :param **kwargs:
        :param req_id:
        :return: whether the given element is in the cache
        """
        if req_id in self.cache_dict:
            return True
        else:
            return False

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache, now its frequency
        :param **kwargs:
        :param obj_id:
        :return: original rank
        """
        self.last_element = obj_id

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id:
        :return: True on success, False on failure
        """
        if len(self.cache_dict) == self.cache_size:
            self.evict()
        self.cache_dict[obj_id] = obj_id
        self.last_element = obj_id

    def find_evict_key(self):
        return self.last_element

    def _print_cache_line(self):
        for key, value in self.cache_dict.items():
            print("{}: {}".format(key, value))

    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        evict_key = self.find_evict_key()
        del self.cache_dict[evict_key]

    def access(self, req_item, **kwargs):
        """
        :param **kwargs:
        :param obj_id: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank
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
        return "MRU, given size: {}, current size: {}".format(self.cache_size, len(self.cache_dict))

    def __str__(self):
        return self.__repr__()
