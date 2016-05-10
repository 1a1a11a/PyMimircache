import os
from ctypes import *
import logging

from mimircache.cacheReader.abstractReader import cacheReaderAbstract


class vscsiCacheReader(cacheReaderAbstract):
    def __init__(self, file_loc):
        super().__init__(file_loc)
        self.buffer_size = 0  # number of trace lines in the buffer
        self.buffer_pointer = 0  # point to the element in the buffer
        self.read_in_num = 0

        self.vscsiC = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.get_lib_name()))
        self.vscsiC.argTypes = [c_char_p, c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        self.c_mem = c_void_p()
        self.c_ver = c_int(-1)
        self.c_delta = c_int(-1)
        self.c_num_of_rec = c_int(-1)

        self.c_read_size = c_int()
        self.c_long_ts = (c_long * 10000)()
        self.c_int_len = (c_int * 10000)()
        self.c_long_lbn = (c_long * 10000)()
        self.c_int_cmd = (c_int * 10000)()

        try:
            # python 2
            filename = c_char_p(self.file_loc)
        except:
            # python 3
            filename = c_char_p(self.file_loc.encode())

        self.vscsiC.setup(filename, byref(self.c_mem), byref(self.c_ver), byref(self.c_delta), byref(self.c_num_of_rec))
        # self.vscsiC.read_trace.resType = c_long
        self.vscsiC.read_trace2.resType = c_int
        self.mem_original = c_void_p(self.c_mem.value)
        self._read()


    def get_lib_name(self):
        for name in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if 'libvscsi' in name and '.py' not in name:
                return name


    def get_first_line(self):
        self.reset()
        return next(self.lines())

    def get_last_line(self):
        self.c_mem = c_void_p(self.mem_original.value + self.c_delta.value * (self.c_num_of_rec.value - 1))
        self.vscsiC.read_trace2(byref(self.c_mem), byref(self.c_ver), c_int(1), self.c_long_ts, self.c_int_len,
                                self.c_long_lbn, self.c_int_cmd)
        return (self.c_long_ts[0], self.c_int_cmd[0], self.c_int_len[0], self.c_long_lbn[0])


    def reset(self):
        logging.debug("reset")
        self.buffer_pointer = 0
        self.read_in_num = 0
        self.counter = 0
        self.c_read_size.value = 0
        self.c_mem = c_void_p(self.mem_original.value)

    def read_one_element(self):
        if self.buffer_pointer < self.c_read_size.value:
            self.buffer_pointer += 1
            return (self.c_long_lbn[self.buffer_pointer - 1])

        elif self.read_in_num < self.c_num_of_rec.value:
            self._read()
            self.buffer_pointer = 1
            return (self.c_long_lbn[self.buffer_pointer - 1])
        else:
            return None


    def get_num_total_lines(self):
        return self.c_num_of_rec.value


    def lines(self):
        # ts, cmd, size, lbn
        '''
        return detailed information in vscsi reader
        :return:
        '''
        self.reset()
        while (self.read_in_num < self.c_num_of_rec.value or self.buffer_pointer < self.c_read_size.value):
            if self.buffer_pointer < self.c_read_size.value:
                self.buffer_pointer += 1
                yield (self.c_long_ts[self.buffer_pointer - 1], self.c_int_cmd[self.buffer_pointer - 1], \
                       self.c_int_len[self.buffer_pointer - 1], self.c_long_lbn[self.buffer_pointer - 1])

            elif self.read_in_num < self.c_num_of_rec.value:
                self._read()
                self.buffer_pointer = 1
                yield (self.c_long_ts[self.buffer_pointer - 1], self.c_int_cmd[self.buffer_pointer - 1], \
                       self.c_int_len[self.buffer_pointer - 1], self.c_long_lbn[self.buffer_pointer - 1])

    def __next__(self):  # Python 3
        super().__next__()
        if self.buffer_pointer < self.c_read_size.value:
            self.buffer_pointer += 1
            return (self.c_long_lbn[self.buffer_pointer - 1])

        elif self.read_in_num < self.c_num_of_rec.value:
            self._read()
            self.buffer_pointer = 1
            return (self.c_long_lbn[self.buffer_pointer - 1])
        else:
            raise StopIteration

    def _read(self):
        if self.c_num_of_rec.value - self.read_in_num >= 10000:
            self.c_read_size = c_int(10000)
        else:
            self.c_read_size = c_int(self.c_num_of_rec.value - self.read_in_num)
        self.read_in_num += self.c_read_size.value
        self.vscsiC.read_trace2(byref(self.c_mem), byref(self.c_ver), self.c_read_size, self.c_long_ts, self.c_int_len,
                                self.c_long_lbn, self.c_int_cmd)

    def __repr__(self):
        return "vscsi cache reader, %s" % super().__repr__()


if __name__ == "__main__":
    reader = vscsiCacheReader('../data/trace_CloudPhysics_bin')

    # # usage one: for reading all elements
    num = 0
    print(reader.c_num_of_rec)
    for line in reader.lines():
        if num == 0:
            prev = line[0]
            num += 1
            continue
        print(line[0] - prev)
        prev = line[0]

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
