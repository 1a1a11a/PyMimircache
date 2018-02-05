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

    def has(self, request_id, **kwargs):
        """
        check whether the given id in the cache or not

        :return: whether the given element is in the cache
        """
        if request_id in self.cacheline_dict:
            return True
        else:
            return False

    def _update(self, request_item, **kwargs):
        """ the given element is in the cache,
        now update cache metadata and its content

        :param **kwargs:
        :param request_item:
        :return: None
        """

        request_id = request_item
        if isinstance(request_item, Req):
            request_id = request_item.item_id

        self.cacheline_dict.move_to_end(request_id)

    def _insert(self, request_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param request_item:
        :return: evicted element or None
        """

        request_id = request_item
        if isinstance(request_item, Req):
            request_id = request_item.item_id

        self.cacheline_dict[request_id] = True

    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: id of evicted cacheline
        """

        request_id = self.cacheline_dict.popitem(last=False)
        return request_id

    def access(self, request_item, **kwargs):
        """
        request access cache, it updates cache metadata,
        it is the underlying method for both get and put

        :param **kwargs:
        :param request_item: the request from the trace, it can be in the cache, or not
        :return: None
        """

        request_id = request_item
        if isinstance(request_item, Req):
            request_id = request_item.item_id

        if self.has(request_id):
            self._update(request_item)
            return True
        else:
            self._insert(request_item)
            if len(self.cacheline_dict) > self.cache_size:
                self.evict()
            return False

    def __len__(self):
        return len(self.cacheline_dict)

    def __repr__(self):
        return "LRU cache of size: {}, current size: {}, {}".\
            format(self.cache_size, self.cacheLinkedList.size, super().__repr__())
