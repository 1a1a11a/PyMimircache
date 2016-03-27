import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import abc


class cache:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, cache_size):
        self.cache_size = cache_size


    @abc.abstractmethod
    def checkElement(self, element):
        '''
        :param element: the element for search
        :return: whether the given element is in the Cache
        '''
        raise NotImplementedError("check_element class is not implemented")

    @abc.abstractmethod
    def _updateElement(self, element):
        ''' the given element is in the Cache, now update it to new location
        :param element:
        :return: True on success, False on failure
        '''
        raise NotImplementedError("__update_element__ class is not implemented")


    @abc.abstractmethod
    def _insertElement(self, element):
        '''
        the given element is not in the Cache, now insert it into Cache
        :param element:
        :return: True on success, False on failure
        '''
        raise NotImplementedError("__insert_element__ class is not implemented")


    @abc.abstractmethod
    def _evictOneElement(self):
        '''
        evict one element from the Cache line
        :return: True on success, False on failure
        '''
        raise NotImplementedError("__evict_one_element__ class is not implemented")


    @abc.abstractmethod
    def addElement(self, element):
        '''
        :param element: the element in the reference, it can be in the Cache, or not
        :return: -1 if not in Cache, otherwise old rank (if there is) or 1
        '''
        raise NotImplementedError("add_element class is not implemented")

    @abc.abstractmethod
    def printCacheLine(self):
        return
