import os
from ctypes import *
import logging

from mimircache.cacheReader.abstractReader import cacheReaderAbstract
import mimircache.c_cacheReader as c_cacheReader


class vscsiReader(cacheReaderAbstract):
    def __init__(self, file_loc, data_type='l', open_c_reader=True):
        super().__init__(file_loc, data_type='l')
        if open_c_reader:
            self.cReader = c_cacheReader.setup_reader(file_loc, 'v', data_type=data_type)


        self.get_num_of_total_requests()


    # def get_first_line(self):
    #     self.reset()
    #     return next(self.lines())



    def reset(self):
        if self.cReader:
            c_cacheReader.reset_reader(self.cReader)

    def read_one_element(self):
        return c_cacheReader.read_one_element(self.cReader)


    def read_time_request(self):
        """
        return real_time information for the request in the form of (time, request)
        :return:
        """
        return c_cacheReader.read_time_request(self.cReader)

    def read_one_request_full_info(self):
        """
        obtain more info for the request in the form of (time, request, size)
        :return:
        """
        return c_cacheReader.read_one_request_full_info(self.cReader)

    def get_average_size(self):
        """
        sum sizes for all the requests, then divided by number of requests
        :return:
        """
        sizes = 0
        counter = 0

        t = self.read_one_request_full_info()
        while t:
            sizes += t[2]
            counter += 1
            t = self.read_one_request_full_info()
        self.reset()
        return sizes/counter



    def __next__(self):  # Python 3
        super().__next__()
        element = c_cacheReader.read_one_element(self.cReader)
        if element!=None:
            return element
        else:
            raise StopIteration


    def __repr__(self):
        return "vscsi cache reader, %s" % super().__repr__()


if __name__ == "__main__":
    reader = vscsiReader('../data/trace.vscsi')
    print(reader.get_average_size())
