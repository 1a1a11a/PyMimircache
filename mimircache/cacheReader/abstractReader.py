# coding=utf-8
import abc
import os
from multiprocessing import Lock
from collections import defaultdict
import mimircache.c_cacheReader as c_cacheReader


class cacheReaderAbstract(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta

    def __init__(self, file_loc, data_type='c', block_unit_size=0, disk_sector_size=0):
        self.file_loc = file_loc
        self.trace_file = None
        self.cReader = None
        self.data_type = data_type
        self.block_unit_size = block_unit_size
        self.disk_sector_size = disk_sector_size
        if self.disk_sector_size!=0:
            assert data_type == 'l', "block size option only support on block request(data type l)"
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

    def get_num_total_req(self):
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

    def get_req_freq_distribution(self):
        d = defaultdict(int)
        for i in self:
            d[i] += 1
        self.reset()
        return d

    def get_num_unique_req(self):
        if self.num_of_uniq_req == -1:
            self.num_of_uniq_req = len(self.get_req_freq_distribution())
        return self.num_of_uniq_req

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __len__(self):
        return self.get_num_total_req()

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



    @abc.abstractmethod
    def read_one_element(self):
        pass

    def close(self):
        try:
            if self is not None:
                if self.trace_file:
                    self.trace_file.close()
                    self.trace_file = None
                if self.cReader and c_cacheReader is not None:
                    c_cacheReader.close_reader(self.cReader)
                    self.cReader = None
        except Exception as e:
            # return
            print("Exception during close reader: {}, ccacheReader={}".format(e, c_cacheReader))

    @abc.abstractmethod
    def __next__(self):  # Python 3
        self.counter += 1

    # @atexit.register
    def __del__(self):
        self.close()
