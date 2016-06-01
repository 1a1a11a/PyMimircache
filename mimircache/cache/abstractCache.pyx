import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import abc


class cache:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, cache_size):
        self.cache_size = cache_size
        if self.cache_size <= 0:
            raise RuntimeError("cache size cannot be smaller than or equal 0")

    @abc.abstractmethod
    def checkElement(self, element):
        '''
        :param element: the element for search
        :return: whether the given element is in the cache
        '''
        raise NotImplementedError("check_element class is not implemented")

    @abc.abstractmethod
    def _updateElement(self, element):
        ''' the given element is in the cache, now update it to new location
        :param element:
        :return: True on success, False on failure
        '''
        raise NotImplementedError("__update_element__ class is not implemented")

    @abc.abstractmethod
    def _insertElement(self, element):
        '''
        the given element is not in the cache, now insert it into cache
        :param element:
        :return: True on success, False on failure
        '''
        raise NotImplementedError("__insert_element__ class is not implemented")

    @abc.abstractmethod
    def _evictOneElement(self):
        '''
        evict one element from the cache line
        :return: True on success, False on failure
        '''
        raise NotImplementedError("__evict_one_element__ class is not implemented")

    @abc.abstractmethod
    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the cache, or not
        :return: -1 if not in cache, otherwise old rank (if there is) or 1
        '''
        raise NotImplementedError("add_element class is not implemented")

    @abc.abstractmethod
    def printCacheLine(self):
        return

    def __contains__(self, item):
        return self.checkElement(item)

    def __repr__(self):
        return "abstract cache class, {}".format(super().__repr__())

    def __repr__(self):
        return "abstract cache class, {}".format(super().__str__())
