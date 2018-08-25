# coding=utf-8


"""
this module provides the stat of the trace
"""

from pprint import pformat
from collections import defaultdict
from PyMimircache.utils.printing import *


class TraceStat:
    """
    this class provides stat calculation for a given trace
    """
    def __init__(self, reader, top_N_popular=8, keep_access_freq_list=False, time_period=[-1, 0]):
        self.reader = reader
        self.top_N_popular = top_N_popular
        self.keep_access_freq_list = keep_access_freq_list
        self.access_freq_list = None
        # stat data representation:
        #       0:  not initialized,
        #       -1: error while obtaining data

        self.num_of_requests = 0
        self.num_of_uniq_obj = 0
        self.cold_miss_ratio = 0

        self.top_N_popular_obj = []
        self.num_of_obj_with_freq_1 = 0
        self.freq_mean = 0
        self.time_span = 0
        self.time_period = time_period

        # self.freq_median = 0
        # self.freq_mode = 0

        self._calculate()


    def _calculate(self, time_period=[-1, 0]):
        """
        calculate all the stat using the reader
        :return:
        """

        d = defaultdict(int)

        # request count counts the number of requests read which is needed to track when the request 
        # to be counted. 
        # sucessful_count counts the number of request counted towards the statistics

        request_count = 0
        sucessful_count = 0 

        if self.reader.support_real_time:
            r = self.reader.read_time_req()
            assert r is not None, "failed to read time and request from reader"

            # if time period ends is zero that means there is no time period
            # else it means we don't know when to get the first time stamp so set to zero
            if not (time_period[1]):
                first_time_stamp = r[0]
            else:
                first_time_stamp = 0

            current_time_stamp = -1
            time_period = self.time_period
            assert time_period[1] >= time_period[0], "end time cannot be smaller or equal to start time"

            while r:
                if time_period[1]:
                    if (request_count >= time_period[0] and request_count <= time_period[1]):
                        d[r[1]] += 1
                        sucessful_count += 1
                    elif (request_count >= time_period[1]):
                        current_time_stamp = r[0]
                        break
                else:
                    d[r[1]] += 1

                request_count += 1

                if not (time_period[1]): 
                    current_time_stamp = r[0]

                r = self.reader.read_time_req()

            last_time_stamp = current_time_stamp

            self.time_span = last_time_stamp - first_time_stamp

        else:
            for i in self.reader:
                d[i] += 1

        self.reader.reset()

        self.num_of_uniq_obj = len(d)
        for _, v in d.items():
            self.num_of_requests += v

        self.cold_miss_ratio = self.num_of_uniq_obj/self.num_of_requests

        # l is a list of (obj, freq) in descending order
        l = sorted(d.items(), key=lambda x: x[1], reverse=True)
        if self.keep_access_freq_list:
            self.access_freq_list = l

        self.top_N_popular_obj = l[:self.top_N_popular]
        for i in range(len(l)-1, -1, -1):
            if l[i][1] == 1:
                self.num_of_obj_with_freq_1 += 1
            else:
                break

        self.freq_mean = self.num_of_requests / (float) (self.num_of_uniq_obj)


    def get_access_freq_list(self):
        return self.access_freq_list

    def get_stat(self, return_format="str"):
        """
        return stat in the format of string or tuple
        :param return_format:
        :return:
        """

        s = "dat: {}\nnumber of requests: {}\nnumber of uniq obj/blocks: {}\n" \
            "cold miss ratio: {:.4f}\ntop N popular (obj, num of requests): \n{}\n" \
            "number of obj/block accessed only once: {}\n" \
            "frequency mean: {:.2f}\n{}".format(self.reader.file_loc,
                                                self.num_of_requests,
                                                self.num_of_uniq_obj,
                                                self.cold_miss_ratio,
                                                pformat(self.top_N_popular_obj),
                                                self.num_of_obj_with_freq_1,
                                                self.freq_mean,
                                                "time span: {}".format(self.time_span) if self.time_span else "")

        if return_format == "str":
            return s

        elif return_format == "tuple":
            return (self.num_of_requests, self.num_of_uniq_obj, self.cold_miss_ratio, self.top_N_popular_obj,
                    self.num_of_obj_with_freq_1, self.freq_mean, self.time_span)

        elif return_format == "dict":
            d = self.__dict__.copy()
            del d["top_N_popular"]
            return d

        else:
            WARNING("unknown return format, return string instead")
            return s


    def get_top_N(self):
        return self.top_N_popular_obj


    def __repr__(self):
        return self.get_stat()


    def __str__(self):
        return self.get_stat()