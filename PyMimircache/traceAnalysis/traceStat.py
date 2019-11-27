# coding=utf-8


"""
    this module provides the stat of the trace

    Jason Yang <peter.waynechina@gmail.com>

"""

from PyMimircache.utils.printing import *
from collections import defaultdict, namedtuple, Counter
from pprint import pformat


class TraceStat:
    """
    this class provides stat calculation for a given trace
    """

    def __init__(self, reader, num_popular_obj=8):
        self.reader = reader
        self.num_popular_obj = num_popular_obj
        self.access_freq_list = None

        self.num_of_req = 0
        self.num_of_obj = 0
        self.cold_miss_ratio = 0

        self.popular_obj = []
        self.num_one_hit_wonders = 0
        self.freq_mean = 0
        self.time_span = 0

        self._calculate()

    def _calculate(self):
        """
        calculate all the stat using the reader
        :return:
        """

        self.reader.reset()
        self.num_of_req = 0
        counter = Counter()

        req = self.reader.read_one_req()
        start_ts = req.real_time
        last_ts = req.real_time
        for req in self.reader:
            counter[req.obj_id] += 1
            self.num_of_req += 1
            last_ts = req.real_time

        self.time_span = last_ts - start_ts
        self.num_of_obj = len(counter)

        self.cold_miss_ratio = self.num_of_obj / self.num_of_req
        self.freq_mean = self.num_of_req / self.num_of_obj
        self.num_one_hit_wonders = sum([1 for v in counter.values() if v == 1])
        self.popular_obj = counter.most_common(self.num_popular_obj)

        # l is a list of (obj, freq) in descending order
        # high_freq_objs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # self.popular_obj = high_freq_objs[:self.num_popular_obj]

        self.reader.reset()

    def get_stat(self, format="str"):
        """
        return stat in the format of string or tuple
        :param format:
        :return:
        """

        s = "dat: {}\nnum of requests: {}\nnum of objects/blocks: {}\n" \
            "num of one-hit-wonders: {}\n" \
            "cold miss ratio: {:.4f}\n" \
            "frequency mean: {:.2f}\ntime span: {}\n" \
            "top N popular (obj, num of requests): \n{}\n".format(self.reader.trace_path,
                                                                  self.num_of_req,
                                                                  self.num_of_obj,
                                                                  self.num_one_hit_wonders,
                                                                  self.cold_miss_ratio,
                                                                  self.freq_mean, self.time_span,
                                                                  pformat(self.popular_obj),
                                                                  )

        if format == "str":
            return s

        elif format == "tuple":
            Stat = namedtuple('Stat', 'num_of_req num_of_obj num_one_hit_wonders cold_miss_ratio '
                                      'freq_mean time_span popular_obj')
            stat_tuple = Stat(
                num_of_req=self.num_of_req,
                num_of_obj=self.num_of_obj,
                num_one_hit_wonders=self.num_one_hit_wonders,
                cold_miss_ratio=self.cold_miss_ratio,
                freq_mean=self.freq_mean, time_span=self.time_span,
                popular_obj=pformat(self.popular_obj)
            )
            return stat_tuple

        elif format == "dict":
            d = self.__dict__.copy()
            del d["num_popular_obj"]
            return d

        else:
            WARNING("unknown format %s, return string instead".format(format))
        return s

    def __repr__(self):
        return self.get_stat()

    def __str__(self):
        return self.get_stat()
