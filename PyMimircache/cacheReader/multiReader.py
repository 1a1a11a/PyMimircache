# coding=utf-8
"""
    this module provides function for reading from multiple traces,
    it supports either reading in the order of real_time or virtual_time (round robin)

    this module has not been tested, use with caution

    Author: Jason Yang <peter.waynechina@gmail.com> 2016/08

"""

import heapq


class MultiReader:
    """
    MultiReader class

    """
    def __init__(self, readers, reading_type="real_time", read_info="timeReq"):
        """
        initialize a MultiReader instance

        :param readers: a list of readers
        :param reading_type: reading order, either real_time or virtual_time
        """

        self.readers = readers
        self.reading_type = reading_type
        self.read_info = read_info
        assert self.reading_type in ["real_time", "virtual_time"], \
            "allowed reading_type: real_time, virtual_time"
        assert len(self.readers) != 0, "reader list is empty"

        self.num_of_read = 0
        # self.reader_buffer = heapdict()
        self.reader_buffer = []

        if self.reading_type == "real_time":
            for j in range(len(self.readers)):
                t, req = self.readers[j].read_time_req()
                self.reader_buffer.append((t, j, req))
                self.num_of_read += 1

        elif self.reading_type == "virtual_time":
            for j in range(len(self.readers)):
                req = self.readers[j].read_one_req()
                self.reader_buffer.append((self.num_of_read, j, req))
                self.num_of_read += 1

        heapq.heapify(self.reader_buffer)


    def read_one_req(self):
        """
        read one request from MultiReader
        :return: one request
        """

        element = self.read_with_readerID()
        if element is not None:
            return element[1]
        else:
            return None

    # noinspection PyPep8Naming
    def read_with_readerID(self):
        """
        read one request and also return its readerID
        :return: (readerID, request)
        """

        # item, pri = self.reader_buffer.popitem()
        # pos, req = item

        item = heapq.heappop(self.reader_buffer)
        t_old, readerID, req_old = item

        reader = self.readers[readerID]

        if self.reading_type == "real_time":
            item = reader.read_time_req()
            if item is not None:
                t, req = item
                heapq.heappush(self.reader_buffer, (t, readerID, req))
            else:
                return None

        elif self.reading_type == "virtual_time":
            req = reader.read_one_req()
            if req is not None:
                # self.reader_buffer[(pos, req)] = self.num_of_read
                heapq.heappush(self.reader_buffer, (self.num_of_read, readerID, req))
            else:
                return None

        self.num_of_read += 1
        return readerID, req_old



    def reset(self):
        """
        reset all readers and MultiReader

        """
        for reader in self.readers:
            reader.reset()
        self.num_of_read = 0

    def close_all_readers(self):
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


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return "a MultiReader using {} for reading order".format(self.reading_type)



if __name__ == "__main__":
    from PyMimircache import *
    from PyMimircache.bin.conf import *

    reader_list = []
    for i in range(1, 10):
        reader_list.append(CsvReader("/home/jason/ALL_DATA/Akamai/day/2016100{}.sort".format(i),
                                     init_params=AKAMAI_CSV))
    mReader = MultiReader(reader_list) #, reading_type="virtual_time")

    for n, i in enumerate(mReader):
        print(i)
        if n>10000:
            break