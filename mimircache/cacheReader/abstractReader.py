import abc
import os
from multiprocessing import Lock
import mimircache.c_cacheReader as c_cacheReader


class cacheReaderAbstract(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __init__(self, file_loc):
        self.file_loc = file_loc
        assert (os.path.exists(file_loc)), "data file does not exist"
        # self.trace_file = open(file_loc, 'r')
        self.counter = 0
        self.trace_file = None
        self.cReader = None
        self.num_of_line = -1
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

        if self.cReader:
            self.num_of_line = c_cacheReader.get_num_of_lines(self.cReader)
        else:
            while self.read_one_element():
                # use the one above, not output progress
                self.num_of_line += 1
            self.reset()

        return self.num_of_line

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __len__(self):
        if self.num_of_line == -1:
            print(self.get_num_of_total_requests())

        return self.num_of_line


    @abc.abstractclassmethod
    def read_one_element(self):
        pass
        # self.counter += 1
        # if (self.counter % 1000 == 0):
        #     print('read in ' + str(self.counter) + ' records')
        # raise NotImplementedError

    def close(self):
        try:
            # pass
            self.trace_file.close()
            if self.cReader:
                # pass
                c_cacheReader.close_reader(self.cReader)
        except:
            pass

    @abc.abstractclassmethod
    def __next__(self):  # Python 3
        self.counter += 1
        # if (self.counter % 100000 == 0):
        #     print('read in ' + str(self.counter) + ' records')
        # raise NotImplementedError

    # @atexit.register
    def __del__(self):
        self.close()
