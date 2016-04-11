import os
from ctypes import *

from mimirCache.CacheReader.readerAbstract import cacheReaderAbstract


class vscsiReader(cacheReaderAbstract):
    def __init__(self, file_loc):
        super().__init__(file_loc)
        self.vscsiC = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.get_lib_name()))
        self.vscsiC.argTypes = [c_char_p, c_void_p, POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        self.mem = c_void_p()
        self.ver = c_int(-1)
        self.delta = c_int(-1)
        self.num_of_rec = c_int(-1)

        try:
            # python 2
            filename = c_char_p(self.file_loc)
        except:
            # python 3
            filename = c_char_p(self.file_loc.encode())

        self.vscsiC.setup(filename, byref(self.mem), byref(self.ver), byref(self.delta), byref(self.num_of_rec))
        self.vscsiC.read_trace.resType = c_long
        self.mem_original = c_void_p(self.mem.value)

    def get_lib_name(self):
        for name in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if 'libvscsi' in name and '.py' not in name:
                return name




    def reset(self):
        self.counter = 0
        self.mem = c_void_p(self.mem_original.value)

    def read_one_element(self):
        super().read_one_element()
        return self.vscsiC.read_trace(byref(self.mem), byref(self.ver), byref(self.delta))

    def __next__(self):  # Python 3
        super().__next__()
        element = self.vscsiC.read_trace(byref(self.mem), byref(self.ver), byref(self.delta))
        if element:
            return element
        else:
            raise StopIteration


if __name__ == "__main__":
    reader = vscsiReader('vscsi/trace')

    # # usage one: for reading all elements
    for i in reader:
        print(i)

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
