# coding=utf-8

"""
    This is the implementation TEAR cache replacement algorithm

    author: Reza <r68karimi+github@gmail.com>
    10/18/2018

"""

from PyMimircache.cache.abstractCache import Cache
from PyMimircache.cacheReader.requestItem import Req



COLD = 0
COLD_TEST = 1
COOL_TEST = 2
WARM = 3
WARM_TEST = 4
HOT_TEST = 5
HOT = 6

SAMPLE_RATE = 100000


class Tear(Cache):
    """
    Tear class to implement TEAR (TEmperature-Aware Replacement) cache eviction algorithm

    """

    def __init__(self, cache_size, **kwargs):
        super(Tear, self).__init__(cache_size, **kwargs)
        
        self.cacheline_dict = {}
    
        self.sampled_accesses = 0
        

    def has(self, req_id, **kwargs):
        """
        check whether the given id in the cache or not
        :return: whether the given element is in the cache
        """
        if (req_id in self.cacheline_dict): 
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

        self.cacheline_dict[req_id] = [COLD, False]
        

    def evict(self, **kwargs):
        """
        evict one cacheline from the cache

        :param **kwargs:
        :return: id of evicted cacheline
        """

        tmp_cold = None
        tmp_warm = None
        tmp_hot = None
        
        for key in self.cacheline_dict.keys():
            value = self.cacheline_dict[key][0]
            if (value == COLD or value == COLD_TEST): 
                tmp_cold = key
                self.cacheline_dict.pop(tmp_cold)
                return tmp_cold

        for key in self.cacheline_dict.keys():
            value = self.cacheline_dict[key][0]
            if (value == WARM or value == COOL_TEST or WARM_TEST):
                tmp_warm = key
                self.cacheline_dict.pop(tmp_warm)
                return tmp_cold

        for key in self.cacheline_dict.keys():
            value = self.cacheline_dict[key][0]
            if (value == HOT or value == HOT_TEST):
                tmp_hot = key
                self.cacheline_dict.pop(tmp_hot)
                return tmp_cold
    

    def update_temperatures(self, **kwargs):
        for key in self.cacheline_dict.keys():
            value = self.cacheline_dict[key][0]
            accessed = self.cacheline_dict[key][1]
            if (accessed):
                if (value == COLD):
                    self.cacheline_dict[key] = COLD_TEST
                elif (value == COLD_TEST):
                    self.cacheline_dict[key] = WARM
                elif (value == COOL_TEST):
                    self.cacheline_dict[key] = WARM
                elif (value == WARM):
                    self.cacheline_dict[key] = HOT_TEST
                elif (value == WARM_TEST):
                    self.cacheline_dict[key] = HOT
                elif (value == HOT_TEST):
                    self.cacheline_dict[key] = HOT
                
                
            else: 
                if (value == HOT):
                    self.cacheline_dict[key] = WARM_TEST
                elif (value == WARM_TEST):
                    self.cacheline_dict[key] = WARM
                elif (value == HOT_TEST):
                    self.cacheline_dict[key] = WARM
                elif (value == COLD_TEST):
                    self.cacheline_dict[key] = COLD
                elif (value == COOL_TEST):
                    self.cacheline_dict[key] = COLD
                elif (value == WARM):
                    self.cacheline_dict[key] = COOL_TEST
                
            self.cacheline_dict[key][1] = False # unset access bit

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

        self.sampled_accesses += 1

        if (self.sampled_accesses >= SAMPLE_RATE): 
            self.update_temperatures()

        if self.has(req_id):
            self._update(req_id)
            return True
        else:
            if len(self.cacheline_dict) >= self.cache_size:
                evict_item = self.evict()
            self._insert(req_item)
            return False

    def __contains__(self, req_item):
        return (req_item in self.cacheline_dict)

    def __len__(self):
        return len(self.cacheline_dict)

    def __repr__(self):
        return "Tear cache of size: {}, current size: {}".\
            format(self.cache_size, len(self.cacheline_dict))
