# coding=utf-8

"""
    this module contains the Request class, which describes a request

"""

import collections

try:
    from dataclasses import dataclass
    import inspect


    @dataclass
    class Request:
        logical_time: int
        obj_id: int
        real_time: int = -1
        obj_size: int = -1
        cnt: int = 1
        op: str = ""
        ttl: int = -1

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            s = f"Request(logical_time={self.logical_time}, real_time={self.real_time}, obj={self.obj_id}, size={self.obj_size}"
            s += f"cnt={self.cnt}" if self.cnt != 1 else ""
            s += f"op={self.op}" if self.op else ""
            s += f"ttl={self.ttl}" if self.ttl != -1 else ""
            return s

    @dataclass
    class FullRequest:
        logical_time: int
        obj_id: int
        real_time: int = -1
        obj_size: int = -1
        req_size: int = -1
        req_range_start: int = -1
        req_range_end: int = -1
        cnt: int = 1
        op: str = ""
        ttl: int = -1

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            members = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
            members = [a for a in members if not (
                    a[0].startswith('__') and a[0].endswith('__'))]

            s = f"Request("
            for member in members:
                if member[1] != -1 and member[1] != "":
                    s += "{}={},\t".format(*member)
            s = s[:-1] + ")"
            return s

except:
    Request = collections.namedtuple('Request', 'logical_time obj_id real_time obj_size cnt op ttl')
    Request.__new__.__defaults__ = (None,) * len(Request._fields)

    FullRequest = collections.namedtuple('FullRequest', 'logical_time obj_id real_time obj_size req_size '
                                                        'req_range_start req_range_end cnt op ttl')
    FullRequest.__new__.__defaults__ = (None,) * len(FullRequest._fields)
