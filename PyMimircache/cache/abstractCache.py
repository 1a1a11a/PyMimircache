# coding=utf-8
"""
    An abstract cache for cache replacement algorithms
    Author: Jason Yang <peter.waynechina@gmail.com> 2016/05
"""

import abc

class Cache:
    __metaclass__ = abc.ABCMeta
    all = ["access",
           "get",
           "access",
           "evict",
           "_update",
           "_insert"]

    def __init__(self, cache_size, **kwargs):
        self.cache_size = cache_size
        if self.cache_size <= 0:
            raise RuntimeError("cache size cannot be smaller than or equal 0")

    @abc.abstractmethod
    def evict(self, **kwargs):
        """
        evict one element from the cache line
        :param **kwargs:
        :return: True on success, False on failure
        """
        raise NotImplementedError("evict is not implemented")

    @abc.abstractmethod
    def access(self, req_item, **kwargs):
        """
        a general method shared by get and access, if an algorithm does not distinguish between
        get and access, then you can just use this
        :param **kwargs:
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank (if there is) or 1
        """
        raise NotImplementedError("access is not implemented")

    @abc.abstractmethod
    def has(self, req_item, **kwargs):
        """
        whether the cache has the request item or not
        :param **kwargs:
        :param req_item: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank (if there is) or 1
        """
        raise NotImplementedError("access is not implemented")

    @abc.abstractmethod
    def _update(self, req_item, **kwargs):
        """ the given element is in the cache, now update it, the following information will be updated
        cache replacement algorithm related metadata
        real request data (not used in current version)

        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        raise NotImplementedError("_update is not implemented")

    @abc.abstractmethod
    def _insert(self, req_item, **kwargs):
        """
        the given element is not in the cache, now insert it into cache
        :param **kwargs:
        :param req_item:
        :return: True on success, False on failure
        """
        raise NotImplementedError("_insert is not implemented")

    @abc.abstractmethod
    def get_size(self, **kwargs):
        """
        get current used size of cache
        :param **kwargs:
        :return: int
        """
        raise NotImplementedError("get_size is not implemented")

    def __len__(self):
        raise NotImplementedError("__len__ not implemented")

    def __contains__(self, req_id):
        return bool(self.has(req_id))

    def __repr__(self):
        return "abstract cache class, {}".format(super().__repr__())

    def __str__(self):
        return "abstract cache class, {}".format(super().__str__())
