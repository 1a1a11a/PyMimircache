import sys
import os

import logging

from ctypes import *
from enum import Enum

from mimircache.cache.LRU import LRU
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.profiler.abstract.abstractLRUProfiler import abstractLRUProfiler


# TODO: to optimize use numpy for ctype return value
# from numpy.ctypeslib import ndpointer





class parda_mode(Enum):
    seq = 1
    openmp = 2
    mpi = 3
    hybrid = 4


class pardaProfiler(abstractLRUProfiler):
    def __init__(self, cache_size, reader, bin_size=1):
        super().__init__(LRU, cache_size, bin_size, reader)

        # load the shared object file
        self.parda_seq = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.get_lib_name()))

        self.c_cache_size = c_long(self.cache_size)

        # if the given file is not basic reader, needs conversion
        if not isinstance(reader, plainCacheReader):
            self.prepare_file()
        else:
            self.num_of_trace_elements = self.reader.get_num_total_lines()

        self.calculated = False

    def get_lib_name(self):
        for name in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if 'libparda' in name and '.py' not in name:
                return name

    def prepare_file_remove_one(self):
        """
        this function will prepare the file, meanwhile remove the request that appear only once
        :return:
        """
        self.num_of_trace_elements = 0
        logging.debug("changing file format")
        seen_dict = {}
        for e in self.reader:
            seen_dict[e] = seen_dict.get(e, 0) + 1
        self.reader.reset()
        print(len(seen_dict))
        with open('temp.dat', 'w') as ofile:
            i = self.reader.read_one_element()
            while i != None:
                self.num_of_trace_elements += 1
                if seen_dict[i] > 1:
                    ofile.write(str(i) + '\n')
                i = self.reader.read_one_element()
        self.reader = plainCacheReader('temp.dat')
        print(self.num_of_trace_elements)



    def prepare_file(self):
        self.num_of_trace_elements = 0
        logging.debug("changing file format")
        with open('temp.dat', 'w') as ofile:
            i = self.reader.read_one_element()
            while i != None:
                self.num_of_trace_elements += 1
                ofile.write(str(i) + '\n')
                i = self.reader.read_one_element()
        self.reader = plainCacheReader('temp.dat')

    def addOneTraceElement(self, element):
        # do not need this function in this profiler
        pass

    def run(self, mode=parda_mode.seq, *args, **kargs):
        self.c_float_array = (c_float * (self.cache_size + 1))()

        c_line_num = c_long(self.num_of_trace_elements)
        c_file_name = c_char_p(self.reader.file_loc.encode())

        if mode == parda_mode.seq:
            self.mode = mode
            self.parda_seq.classical_tree_based_stackdist(c_file_name, c_line_num, self.c_cache_size,
                                                          self.c_float_array)

        elif mode == parda_mode.openmp:
            self.num_of_threads = kargs["threads"]
            self.mode = mode
            c_threads = c_int(self.num_of_threads)
            self.parda_seq.parda_seperate_file(c_file_name, c_threads, c_line_num)
            print("all files have been separated for parallel")
            # self.parda_seq.parda_omp_stackdist.restype = POINTER(c_double)
            self.parda_seq.parda_omp_stackdist(c_file_name, c_line_num, c_threads, self.c_cache_size,
                                               self.c_float_array)

        elif mode == parda_mode.mpi:
            print('sorry, currently, mpi mode is not supported')
        elif mode == parda_mode.hybrid:
            print('sorry, currently, openmp/mpi hybrid mode is not supported')
        else:
            print("does not support given run mode")

        self.run_aux()

    def run_with_specified_lines(self, begin_line, end_line):
        c_line_num = c_long(self.num_of_trace_elements)
        c_file_name = c_char_p(self.reader.file_loc.encode())
        self.c_float_array = (c_float * (self.cache_size + 1))()

        c_begin = c_long(begin_line)
        c_end = c_long(end_line)
        self.parda_seq.classical_with_line_specification(c_file_name, c_line_num, self.c_cache_size,
                                                         c_begin, c_end, self.c_float_array)
        self.run_aux()

    def run_aux(self):
        self.calculated = True

        for i in range(self.cache_size):
            self.HRC_bin[i] = self.c_float_array[i + 1] * 100

        self.HRC[0] = self.HRC_bin[0]
        for i in range(1, self.cache_size):
            self.HRC[i] = self.HRC_bin[i] + self.HRC[i - 1]

        # the size of list is one larger than cache_size, if bin_size is 1
        self.HRC[-1] = self.HRC[-2]

        for i in range(self.num_of_blocks):
            self.MRC[i] = 100 - self.HRC[i]

        for i in range(len(self.MRC)):
            pass

    def get_reuse_distance(self):
        c_line_num = c_long(self.num_of_trace_elements)
        c_file_name = c_char_p(self.reader.file_loc.encode())
        self.c_long_array = (c_long * (self.num_of_trace_elements))()

        self.parda_seq.get_reuse_distance(c_file_name, c_line_num, self.c_cache_size, self.c_long_array)
        # for i in self.c_long_array:
        #     print(i+1)

        return self.c_long_array

    def _test(self, cache_size=2000):
        c_line_num = c_long(self.num_of_trace_elements)
        c_file_name = c_char_p(self.reader.file_loc.encode())
        self.c_long_array = (c_long * (self.num_of_trace_elements + 1))()

        # self.parda_seq.main(c_line_num, c_file_name)
        self.parda_seq.get_reuse_distance_smart(c_file_name, c_line_num, self.c_cache_size, self.c_long_array)
        count = 0
        print(len(self.c_long_array))
        for i in self.c_long_array:
            # print(i)
            if i > -1 and i < cache_size:
                count += 1
        print(count)
        print(len(self.c_long_array))
        self.parda_seq.get_reuse_distance(c_file_name, c_line_num, self.c_cache_size, self.c_long_array)
        count = 0
        for i in self.c_long_array:
            if i > -1 and i < cache_size:
                count += 1
        print(count)

        def __del__(self):
            if os.path.exists('temp.dat'):
                os.remove('temp.dat')


if __name__ == "__main__":
    # p = pardaProfiler(30000, plainCacheReader("../data/parda.trace"))
    # p = pardaProfiler(30000, csvCacheReader("../data/trace_CloudPhysics_txt", 4))
    # p = parda(LRU, 3000000, basicCacheReader("temp.dat"))
    # p.run(parda_mode.seq, threads=4)
    # p.get_reuse_distance()
    import os
    import time
    import shutil

    # from pympler.tracker import SummaryTracker

    # tracker = SummaryTracker()

    t1 = time.time()
    for i in range(6):
        p = pardaProfiler(30000, vscsiCacheReader("../data//trace_CloudPhysics_bin"))
        # rd_a = p.get_reuse_distance()
    t2 = time.time()

    print(t2 - t1)
    p = pardaProfiler(30000, vscsiCacheReader("../data//trace_CloudPhysics_bin"))
    p.run()
    p.plotHRC()
    # print(len())
    # count = 0
    # for i in rd_a:
    #     if i==0:
    #         count+=1
    # print(count)

    # tracker.print_diff()


    # p = pardaProfiler(2000, csvCacheReader('../data/mining/mining.dat.original', 1))
    # p = pardaProfiler(2000, plainCacheReader('../data/mining/mining.dat.original'))
    # p._test()


    # for f in os.listdir('../data/mining/'):
    #     shutil.copy('../data/mining/' + f, '../data/mining/mining.dat')
    #     print(f)
    #     p._test()
    # p.run_with_specified_lines(10000, 20000)
        # p.plotHRC(autosize=True, autosize_threshhold=0.00001)
