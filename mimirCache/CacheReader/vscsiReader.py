import os
from collections import deque
from ctypes import *

from mimirCache.CacheReader.abstractReader import cacheReaderAbstract


class vscsiCacheReader(cacheReaderAbstract):
    def __init__(self, file_loc):
        super().__init__(file_loc)
        # self.buffer = deque()
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
        self.read()


    def get_lib_name(self):
        for name in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if 'libvscsi' in name and '.py' not in name:
                return name


    def reset(self):
        self.counter = 0
        self.c_mem = c_void_p(self.mem_original.value)

    def read_one_element(self):
        super().read_one_element()
        return self.vscsiC.read_trace(byref(self.c_mem), byref(self.c_ver), byref(self.c_delta))

    def lines(self):
        # ts, cmd, size, lbn
        while (self.read_in_num < self.c_num_of_rec.value or self.buffer_pointer < self.c_read_size.value):
            if self.buffer_pointer < self.c_read_size.value:
                self.buffer_pointer += 1
                yield (self.c_long_ts[self.buffer_pointer - 1], self.c_int_cmd[self.buffer_pointer - 1], \
                       self.c_int_len[self.buffer_pointer - 1], self.c_long_lbn[self.buffer_pointer - 1])

            elif self.read_in_num < self.c_num_of_rec.value:
                self.read()
                self.buffer_pointer = 1
                yield (self.c_long_ts[self.buffer_pointer - 1], self.c_int_cmd[self.buffer_pointer - 1], \
                       self.c_int_len[self.buffer_pointer - 1], self.c_long_lbn[self.buffer_pointer - 1])





    def __next__(self):  # Python 3
        super().__next__()

        if self.buffer_pointer < self.c_read_size.value:
            self.buffer_pointer += 1
            return (self.c_long_lbn[self.buffer_pointer - 1])

        elif self.read_in_num < self.c_num_of_rec.value:
            self.read()
            self.buffer_pointer = 1
            return (self.c_long_lbn[self.buffer_pointer - 1])
        else:
            raise StopIteration

    def read(self):
        if self.c_num_of_rec.value - self.read_in_num >= 10000:
            self.c_read_size = c_int(10000)
        else:
            self.c_read_size = c_int(self.c_num_of_rec.value - self.read_in_num)
        self.read_in_num += self.c_read_size.value
        print(self.read_in_num)

        self.vscsiC.read_trace2(byref(self.c_mem), byref(self.c_ver), self.c_read_size, self.c_long_ts, self.c_int_len,
                                self.c_long_lbn, self.c_int_cmd)

    def __repr__(self):
        return "vscsi cache reader, %s" % super.__repr__()





if __name__ == "__main__":
    reader = vscsiCacheReader('../Data/trace_CloudPhysics_bin')

    # # usage one: for reading all elements
    num = 0
    print(reader.c_num_of_rec)
    for i in reader.lines():
        num += 1
        # print("{}: {}".format(num, i))
    print(num)

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
