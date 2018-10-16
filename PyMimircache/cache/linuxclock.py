# coding=utf-8

"""
    This is the implementation of (immitation of) linux clock replacement algorithm


    author: Reza <r68karimi+github@gmail.com>
    10/11/2018

"""

from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req


HI_WATERMARK = 4.0
LOW_WATERMARK = 2.0
BALANCE_TRIG = 0.9

class LinuxClock(Cache):
    """
    LinuxClock class to approximate linux page cache eviction algorithm

    """

    def __init__(self, cache_size, **kwargs):
        super(LinuxClock, self).__init__(cache_size, **kwargs)
        
        self.cacheline_active_dict = {}
        self.cacheline_inactive_dict = {}
        
        self.cacheline_active_list = []
        self.cacheline_inactive_list = []


    def _balance_lists(self, **kwargs):

        if len(self.cacheline_inactive_dict) != 0 and ( (len(self.cacheline_active_dict)  / len(self.cacheline_inactive_dict)) > BALANCE_TRIG) and (len(self.cacheline_active_dict)  / len(self.cacheline_inactive_dict)) > HI_WATERMARK : 
            while (len(self.cacheline_active_dict)  / len(self.cacheline_inactive_dict) > LOW_WATERMARK) :
                tmp = self.cacheline_active_list.pop() # take one item off active's tail
                self.cacheline_active_dict.pop(tmp)
                self.cacheline_inactive_list.insert(0, tmp) # insert it at the head of inactive list
                self.cacheline_inactive_dict[tmp] = True 
        return True
        

    def has(self, req_id, **kwargs):
        """
        check whether the given id in the cache or not
        :return: whether the given element is in the cache
        """
        if (req_id in self.cacheline_active_dict) or (req_id in self.cacheline_inactive_dict):
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
        

        if req_id in self.cacheline_active_dict : # page is already active
            self.cacheline_active_list.remove(req_id)
            self.cacheline_active_list.insert(0, req_id)

            #pass # do nothing

        else: # page is inactive, move it to the active list 
            self.cacheline_inactive_dict.pop(req_id) # remove from inactive
            self.cacheline_inactive_list.remove(req_id)

            self.cacheline_active_dict[req_id] = True # insert to active
            self.cacheline_active_list.insert(0, req_id)

        return True


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

        self.cacheline_inactive_dict[req_id] = True # insert to inactive
        self.cacheline_inactive_list.insert(0, req_id)

    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: id of evicted cacheline
        """

        tmp = -1

        if (len(self.cacheline_inactive_list) == 0) : # if inactive list is empty, take from active's tail
            tmp = self.cacheline_active_list.pop()
            self.cacheline_active_dict.pop(tmp)

        else: 
            tmp = self.cacheline_inactive_list.pop() # take one item off inactive's tail
            self.cacheline_inactive_dict.pop(tmp)
            
        return tmp


    def access(self, req_item, **kwargs):
        """
        request access cache, it updates cache metadata,
        it is the underlying method for both get and put

        :param **kwargs:
        :param req_item: the request from the trace, it can be in the cache, or not
        :return: None
        """

        self._balance_lists() # balance the lists at each access

        req_id = req_item
        if isinstance(req_item, Req):
            req_id = req_item.item_id

        if self.has(req_id):
            self._update(req_id)
            return True
        else:
            if (len(self.cacheline_active_dict) + len(self.cacheline_inactive_dict)) >= self.cache_size:
                evict_item = self.evict()
            self._insert(req_item)
            return False

    def __contains__(self, req_item):
        return (req_item in self.cacheline_active_dict) or (req_item in self.cacheline_inactive_dict)

    def __len__(self):
        return len(self.cacheline_active_dict) + len(self.cacheline_inactive_dict)

    def __repr__(self):
        return "LinuxClock cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_active_dict) + len(self.cacheline_inactive_dict))
