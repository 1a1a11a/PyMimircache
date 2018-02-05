# coding=utf-8

"""
this module contains the CacheLine class, which describes a cacheLine

"""


class CacheLine:
    def __init__(self, item_id, size=1, op=None, cost=-1, **kwargs):
        self._item_id = item_id
        self._size = size
        self._op = op
        self._cost = cost

    @property
    def item_id(self):
        return self._item_id

    @property
    def size(self):
        return self._size

    @property
    def op(self):
        return self._op

    @property
    def cost(self):
        return self._cost
