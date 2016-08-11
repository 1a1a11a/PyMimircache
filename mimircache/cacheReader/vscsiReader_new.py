import os
from ctypes import *
import logging

from mimircache.cacheReader.abstractReader import cacheReaderAbstract
import mimircache.c_cacheReader as c_cacheReader


class vscsiReader(cacheReaderAbstract):
    def __init__(self, file_loc, open_c_reader=True):
        super().__init__(file_loc)
        if open_c_reader:
            self.cReader = c_cacheReader.setup_reader(file_loc, 'v')


        self.get_num_of_total_requests()


    # def get_first_line(self):
    #     self.reset()
    #     return next(self.lines())



    def reset(self):
        if self.cReader:
            c_cacheReader.reset_reader(self.cReader)

    def read_one_element(self):
        return c_cacheReader.read_one_element(self.cReader)


    # def __next__(self):  # Python 3
    #     super().__next__()
    #     if self.buffer_pointer < self.c_read_size.value:
    #         self.buffer_pointer += 1
    #         return self.c_long_lbn[self.buffer_pointer - 1]
    #
    #     elif self.read_in_num < self.c_num_of_rec.value:
    #         self._read()
    #         self.buffer_pointer = 1
    #         return self.c_long_lbn[self.buffer_pointer - 1]
    #     else:
    #         raise StopIteration

    def read_time_request(self):
        return c_cacheReader.read_time_request(self.cReader)



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

    # # usage one: for reading all elements
    num = 0
    ofile = open('testoutput', 'w')
    t = reader.read_time_request()
    while t != None:
        # print(t)
        ofile.write('{}\n'.format(t))
        t= reader.read_time_request()

        num += 1


        # print("{}: {}".format(num, i))
        # print(num)
        # print(reader)
        # print(reader.get_first_line())
        # print(reader.get_last_line())

    #
    # # usage two: best for reading one element each time
    # s = reader.read_one_element()
    # # s2 = next(reader)
    # while (s):
    #     print(s)
    #     s = reader.read_one_element()
    #     # s2 = next(reader)

    # test3: read first 10 elements twice
    # for i in range(10):
    #     print(reader.read_one_element())
    # print("after reset")
    # reader.reset()
    # for i in range(10):
    #     print(reader.read_one_element())
