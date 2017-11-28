# coding=utf-8
"""
    An abstract cache for cache replacement algorithms


    Author: Jason Yang <peter.waynechina@gmail.com> 2016/05

"""

import abc


class cache:
    __metaclass__ = abc.ABCMeta
    all = ["check_element",
           "_update_element",
           "_insert_element",
           "evict_one_element",
           "add_element"]

    @abc.abstractclassmethod
    def __init__(self, cache_size, **kwargs):
        self.cache_size = cache_size
        if self.cache_size <= 0:
            raise RuntimeError("cache size cannot be smaller than or equal 0")

    @abc.abstractclassmethod
    def check_element(self, element):
        """
        :param element: the element for search
        :return: whether the given element is in the cache
        """
        raise NotImplementedError("check_element is not implemented")

    @abc.abstractclassmethod
    def _update_element(self, element):
        """ the given element is in the cache, now update it to new location
        :param element:
        :return: True on success, False on failure
        """
        raise NotImplementedError("_update_element is not implemented")

    @abc.abstractclassmethod
    def _insert_element(self, element):
        """
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        """
        raise NotImplementedError("_insert_element is not implemented")

    @abc.abstractclassmethod
    def _evict_one_element(self):
        """
        evict one element from the cache line
        :return: True on success, False on failure
        """
        raise NotImplementedError("_evict_one_element is not implemented")

    @abc.abstractclassmethod
    def add_element(self, element):
        """
        :param element: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank (if there is) or 1
        """
        raise NotImplementedError("add_element is not implemented")

    def __contains__(self, item):
        return self.check_element(item)

    def __repr__(self):
        return "abstract cache class, {}".format(super().__repr__())

    def __str__(self):
        return "abstract cache class, {}".format(super().__str__())
