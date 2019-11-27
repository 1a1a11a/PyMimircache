# coding=utf-8
"""
    this module provides function for reading from multiple traces,
    it supports either reading in the ordering of real_time or logical_time (round robin)

    this module has not been tested, use with caution

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""

import heapq


class MultiReader:
    """
    MultiReader class for round robin read from multiple readers

    """
    def __init__(self, readers, ordering="real_time"):
        """
        initialize a MultiReader instance

        :param readers: a list of readers
        :param ordering: reading ordering, either real_time or logical_time
        """

        self.readers = readers
        self.ordering = ordering
        assert self.ordering in ["real_time", "logical_time"], \
            "allowed ordering: real_time, logical_time"
        assert len(self.readers) != 0, "reader list is empty"

        self.reader_buffer = []

        for reader_idx in range(len(self.readers)):
            req = self.readers[reader_idx].read_one_req()
            self.reader_buffer.append((getattr(req, self.ordering), reader_idx, req))

        heapq.heapify(self.reader_buffer)

    def read_one_req(self):
        """
        read one request from MultiReader
        :return: one request
        """

        element = self.read_with_reader_idx()
        if element is not None:
            return element[1]
        else:
            return None

    # noinspection PyPep8Naming
    def read_with_reader_idx(self):
        """
        read one request and also return its reader_idx
        :return: (reader_idx, request)
        """

        if len(self.reader_buffer) == 0:
            return None

        item = heapq.heappop(self.reader_buffer)
        _, reader_idx, ret_req = item

        req = self.readers[reader_idx].read_one_req()
        if req:
            ts = getattr(req, self.ordering)
            heapq.heappush(self.reader_buffer, (ts, reader_idx, req))

        return ret_req

    def reset(self):
        """
        reset all readers and MultiReader

        """
        for reader in self.readers:
            reader.reset()

    def close(self):
        """
        close all readers

        """
        for reader in self.readers:
            reader.close()

    def __iter__(self):
        return self

    def __next__(self):
        v = self.read_one_req()
        if v is not None:
            return v
        else:
            raise StopIteration

    def next(self):
        """ part of the iterator implementation """
        return self.__next__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return "a MultiReader using {} for reading ordering".format(self.ordering)

