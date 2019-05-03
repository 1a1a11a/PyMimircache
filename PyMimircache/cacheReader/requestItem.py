# coding=utf-8

"""
this module contains the Req class, which describes a request

"""



class Req:
    def __init__(self, obj_id, ts=-1, size=1, op=None, cost=-1, **kwargs):
        self._obj_id = obj_id
        self._ts = ts
        self._size = size
        self._op = op
        self._cost = cost

    @property
    def obj_id(self):
        return self._obj_id

    @property
    def ts(self):
        return self._ts

    @property
    def size(self):
        return self._size

    @property
    def op(self):
        return self._op

    @property
    def cost(self):
        return self._cost

