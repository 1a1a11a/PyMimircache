# coding=utf-8

"""
    This is the new implementation of LRU using OrderedDict, and it is
    slightly faster than original implementation using home-made LinkedList


"""

from collections import OrderedDict
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class LRU(Cache):
    """
    LRU class for simulating a LRU cache

    """

    def __init__(self, cache_size, **kwargs):
        super().__init__(cache_size, **kwargs)
        self.cacheline_dict = OrderedDict()

    def has(self, obj_id, **kwargs):
        """
        check whether the given id in the cache or not

        :return: whether the given element is in the cache
        """
        return obj_id in self.cacheline_dict

    def _update(self, obj_id, **kwargs):
        """ the given element is in the cache,
        now update cache metadata and its content

        :param **kwargs:
        :param obj_id:
        :return: None
        """

        self.cacheline_dict.move_to_end(obj_id)

    def _insert(self, obj_id, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param obj_id:
        :return: evicted element or None
        """

        self.cacheline_dict[obj_id] = True

    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: id of evicted cacheline
        """

        evict_obj_id = self.cacheline_dict.popitem(last=False)[0]
        return evict_obj_id

    def access(self, req_item, **kwargs):
        """
        request access cache, it updates cache metadata,
        it is the underlying method for both get and put

        :param **kwargs:
        :param req_item: the request from the trace, it can be in the cache, or not
        :return: None
        """

        obj_id = req_item
        if isinstance(req_item, Req):
            obj_id = req_item.obj_id

        if self.has(obj_id):
            self._update(obj_id)
            return True
        else:
            self._insert(obj_id)
            if len(self.cacheline_dict) > self.cache_size:
                evict_item = self.evict()
                if "evict_item_list" in kwargs:
                    kwargs["evict_item_list"].append(evict_item)
            return False

    def __contains__(self, obj_id):
        return obj_id in self.cacheline_dict

    def __len__(self):
        return len(self.cacheline_dict)

    def __repr__(self):
        return "LRU cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_dict))
