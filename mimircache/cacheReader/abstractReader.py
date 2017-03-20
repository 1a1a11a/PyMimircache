# coding=utf-8
import abc
import os
from multiprocessing import Lock
from collections import defaultdict
import mimircache.c_cacheReader as c_cacheReader


class cacheReaderAbstract(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __init__(self, file_loc, data_type='c'):
        self.file_loc = file_loc
        self.trace_file = None
        self.cReader = None
        assert (os.path.exists(file_loc)), "data file({}) does not exist".format(file_loc)

        self.counter = 0
        self.num_of_line = -1
        self.num_of_uniq_req = -1
        self.lock = Lock()

    def reset(self):
        """
        reset the read location back to beginning
        :return:
        """
        self.counter = 0
        self.trace_file.seek(0, 0)
        if self.cReader:
            c_cacheReader.reset_reader(self.cReader)

    def get_num_of_total_requests(self):
        if self.num_of_line != -1:
            return self.num_of_line

        # clear before counting
        self.num_of_line = 0
        if self.cReader:
            self.num_of_line = c_cacheReader.get_num_of_lines(self.cReader)
        else:
            while self.read_one_element() is not None:
                self.num_of_line += 1
        self.reset()

        return self.num_of_line

    def get_request_num_distribution(self):
        d = defaultdict(int)
        for i in self:
            d[i] += 1
        self.reset()
        return d

    def get_num_of_unique_requests(self):
        if self.num_of_uniq_req == -1:
            self.num_of_uniq_req = len(self.get_request_num_distribution())
        return self.num_of_uniq_req

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __len__(self):
        return self.get_num_of_total_requests()

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



    @abc.abstractclassmethod
    def read_one_element(self):
        pass

    def close(self):
        try:
            if self.trace_file:
                self.trace_file.close()
                self.trace_file = None
            if self.cReader and c_cacheReader is not None:
                c_cacheReader.close_reader(self.cReader)
                self.cReader = None
        except Exception as e:
            print("Exception during close reader: {}, ccacheReader={}".format(e, c_cacheReader))

    @abc.abstractclassmethod
    def __next__(self):  # Python 3
        self.counter += 1

    # @atexit.register
    def __del__(self):
        self.close()
