"""
.. module:: mimircache.cache
    :platform: Linux, OSX
    :synopsis: The module for all available cache replacement algorithms, currently we have ARC, clock, FIFO,
                LFU_LRU, LFU_MRU, LFU_RR, LRU, MRU, optimal, Random, SLRU, S4LRU,
                To write your own cache replacement algorithm, just use abstractCache.py as base class and override
                all the methods in abstractCache.py. An example is given below:

>>> from mimircache.cache.abstractCache import cache
>>> class Random(cache):
>>>     def __init__(self, cache_size):
>>>         super().__init__(cache_size)
>>>         self.cache_dict = dict()
>>>         self.cache_line_list = []  # to save all the keys, otherwise needs to populate from cache_dict every time
>>>
>>>     def checkElement(self, element):
            '''
            :param element: the key of cache request
            :return: whether the given key is in the cache or not
            '''
>>>         if element in self.cache_dict:
>>>             return True
>>>         else:
>>>             return False
>>>
>>>     def _updateElement(self, element):
            ''' the given element is in the cache, when it is requested again,
             usually we need to update it to new location, but in random, we don't need to do that
            :param element: the key of cache request
            :return: None
            '''
>>>         pass
>>>
>>>     def _insertElement(self, element):
            '''
            the given element is not in the cache, now insert it into cache
            :param element: the key of cache request
            :return: None
            '''
>>>         if len(self.cache_dict) >= self.cache_size:
>>>             self._evictOneElement()
>>>         self.cache_dict[element] = ""
>>>         self.cache_line_list.append(element)
>>>
>>>     def _evictOneElement(self):
            '''
            evict one element from the cache line
            if we delete one element from list every time, it would be O(N) on
            every request, which is too expensive, so we choose to put a hole
            on the list every time we delete it, when there are too many holes
            we re-generate the cache line list
            :return: None
            '''
>>>         rand_num = random.randrange(0, len(self.cache_line_list))
>>>         element = self.cache_line_list[rand_num]
>>>         count = 0
>>>         while not element:
>>>         rand_num = random.randrange(0, len(self.cache_line_list))
>>>         element = self.cache_line_list[rand_num]
>>>         count += 1
            # mark this element as deleted, put a hole on it
>>>         self.cache_line_list[rand_num] = None
>>>         if count > 10:
                # if there are too many holes, then we need to resize the list
>>>             new_list = [e for e in self.cache_line_list if e]
>>>             del self.cache_line_list
>>>             self.cache_line_list = new_list
>>>         del self.cache_dict[element]
>>>
>>>     def addElement(self, element):
            '''
            :param element: the key of cache request, it can be in the cache, or not in the cache
            :return: True if element in the cache
            '''
>>>         if self.checkElement(element):
>>>             self._updateElement(element)
>>>             return True
>>>         else:
>>>             self._insertElement(element)
>>>             return False
>>>
>>>     def __repr__(self):
>>>         return "Random Replacement, given size: {}, current size: {}".format(self.cache_size, len(self.cache_dict), super().__repr__())


.. autoclass:: abstractCache
    :members:

.. autoclass:: LRU

# ARC, clock, FIFO, LFU_LRU, LFU_MRU, LFU_RR, MRU, optimal, RR, S4LRU, SLRU


.. moduleauthor:: Juncheng Yang <peter.waynechina@gmail.com>, Ymir Vigfusson

"""


# __all__ = ['LRU, ARC, clock, FIFO, LFU_LRU, LFU_MRU, LFU_RR, MRU, optimal, RR, S4LRU, SLRU']
