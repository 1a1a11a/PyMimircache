# coding=utf-8
import os
from ctypes import *
import logging

from mimircache.cacheReader.abstractReader import cacheReaderAbstract
import mimircache.c_cacheReader as c_cacheReader


class vscsiReader(cacheReaderAbstract):
    def __init__(self, file_loc, data_type='l', block_unit_size=0, disk_sector_size=512, open_c_reader=True):
        super().__init__(file_loc, data_type='l', block_unit_size=block_unit_size,
                         disk_sector_size=disk_sector_size)
        if open_c_reader:
            self.cReader = c_cacheReader.setup_reader(file_loc, 'v', data_type=data_type,
                                                      block_unit_size=block_unit_size,
                                                      disk_sector_size=disk_sector_size)

        self.get_num_total_req()


    def reset(self):
        if self.cReader:
            c_cacheReader.reset_reader(self.cReader)

    def read_one_element(self):
        r = c_cacheReader.read_one_element(self.cReader)
        if r and self.block_unit_size != 0 and self.disk_sector_size != 0:
            r = r * self.disk_sector_size // self.block_unit_size
        return r

    def read_time_request(self):
        """
        return real_time information for the request in the form of (time, request)
        :return:
        """
        r = c_cacheReader.read_time_request(self.cReader)
        if r and self.block_unit_size != 0 and self.disk_sector_size != 0:
            r[1] = r[1] * self.disk_sector_size // self.block_unit_size
        return r

    def read_one_request_full_info(self):
        """
        obtain more info for the request in the form of (time, request, size)
        :return:
        """
        r = c_cacheReader.read_one_request_full_info(self.cReader)
        if r and self.block_unit_size != 0 and self.disk_sector_size != 0:
            r = list(r)
            r[1] = r[1] * self.disk_sector_size // self.block_unit_size
        return r

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
        element = self.read_one_element()
        if element is not None:
            return element
        else:
            raise StopIteration


    def __repr__(self):
        return "vscsiReader of {}".format(self.file_loc)


if __name__ == "__main__":
    reader = vscsiReader('../data/trace.vscsi')
    print(reader.get_average_size())
