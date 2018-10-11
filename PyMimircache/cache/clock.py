# coding=utf-8

"""
    This is the implementation of clock replacement algorithm


    author: Jason <peter.waynechina@gmail.com>
    2018/10/10

"""

from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


class Clock(Cache):
    """
    Clock class for simulating a Clock cache

    """

    def __init__(self, cache_size, **kwargs):
        super(Clock, self).__init__(cache_size, **kwargs)
        self.cacheline_list = [(None, 0)] * self.cache_size
        self.cacheline_dict = {}
        self.hand = 0       # always keep the hand at the pos where new request can be inserted

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
        pass


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
        assert self.cacheline_list[self.hand][0] is None or self.cacheline_list[self.hand][1] == -1, \
            "current pos {}, hand {}".format(self.cacheline_list[self.hand], self.hand)
        self.cacheline_list[self.hand] = (req_id, 1)
        self.cacheline_dict[req_id] = self.hand
        self.hand = (self.hand + 1) % self.cache_size


    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: id of evicted cacheline
        """

        while self.cacheline_list[self.hand][1] == 1:
            self.cacheline_list[self.hand] = (self.cacheline_list[self.hand][0], 0)
            self.hand = (self.hand + 1) % self.cache_size

        req_id = self.cacheline_list[self.hand][0]
        del self.cacheline_dict[req_id]
        self.cacheline_list[self.hand] = (self.cacheline_list[self.hand][0], -1)
        # self.hand = (self.hand + 1) % self.cache_size
        return req_id


    def access(self, req_item, **kwargs):
        """
        request access cache, it updates cache metadata,
        it is the underlying method for both get and put

        :param **kwargs:
        :param req_item: the request from the trace, it can be in the cache, or not
        :return: None
        """

        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id

        if self.has(req_id):
            return True
        else:
            if len(self.cacheline_dict) >= self.cache_size:
                evict_item = self.evict()
            self._insert(req_item)
            return False

    def __contains__(self, req_item):
        return req_item in self.cacheline_dict

    def __len__(self):
        return len(self.cacheline_dict)

    def __repr__(self):
        return "Clock cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_dict))
