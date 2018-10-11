# coding=utf-8

"""
    This is the implementation of second change cache replacement algorithm


"""

from collections import OrderedDict, deque
from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class SecondChance(Cache):
    """
    LRU class for simulating a LRU cache

    """

    def __init__(self, cache_size, **kwargs):
        super(SecondChance, self).__init__(cache_size, **kwargs)
        self.cacheline_list = deque()
        self.cacheline_dict = {}

    def has(self, req_id, **kwargs):
        """
        check whether the given id in the cache or not

        :return: whether the given element is in the cache
        """
        if req_id in self.cacheline_dict:
            return True
        else:
            return False

    def _update(self, req_item, **kwargs):
        """ the given element is in the cache,
        now update cache metadata and its content

        :param **kwargs:
        :param req_item:
        :return: None
        """

        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id
        self.cacheline_dict[req_id] = 1


    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: evicted element or None
        """

        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id
        self.cacheline_list.append(req_id)
        self.cacheline_dict[req_id] = 1


    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: id of evicted cacheline
        """

        req_id = self.cacheline_list.popleft()
        while self.cacheline_dict[req_id] == 1:
            self.cacheline_dict[req_id] = 0
            self.cacheline_list.append(req_id)
            req_id = self.cacheline_list.popleft()

        del self.cacheline_dict[req_id]
        return req_id

    def access(self, req_item, **kwargs):
        """
        request access cache, it updates cache metadata,
        it is the underlying method for both get and put

        :param **kwargs:
        :param req_item: the request from the trace, it can be in the cache, or not
        :return: None
        """

        assert len(self.cacheline_list) == len(self.cacheline_dict)
        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id

        if self.has(req_id):
            return True
        else:
            self._insert(req_item)
            if len(self.cacheline_list) > self.cache_size:
                evict_item = self.evict()
            return False

    def __contains__(self, req_item):
        return req_item in self.cacheline_dict

    def __len__(self):
        return len(self.cacheline_dict)

    def __repr__(self):
        return "SecondChance cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_dict))
