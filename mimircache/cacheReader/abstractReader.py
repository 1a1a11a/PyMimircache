import abc
import os


class cacheReaderAbstract(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __init__(self, file_loc):
        self.file_loc = file_loc
        assert (os.path.exists(file_loc)), "data file does not exist"
        self.trace_file = open(file_loc, 'r')
        self.counter = 0

    def reset(self):
        '''
        reset the read location back to beginning
        :return:
        '''
        self.counter = 0
        self.trace_file.seek(0, 0)

    def get_num_total_lines(self):
        self.num_of_line = 0
        while self.read_one_element():
            # use the one above, not output progress
            # for i in self:
            self.num_of_line += 1
        self.reset()
        return self.num_of_line

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    @abc.abstractclassmethod
    def read_one_element(self):
        pass
        # self.counter += 1
        # if (self.counter % 1000 == 0):
        #     print('read in ' + str(self.counter) + ' records')
        # raise NotImplementedError

    @abc.abstractclassmethod
    def __next__(self):  # Python 3
        self.counter += 1
        if (self.counter % 100000 == 0):
            print('read in ' + str(self.counter) + ' records')
            # raise NotImplementedError

    # @atexit.register
    def __del__(self):
        try:
            self.trace_file.close()
        except:
            pass
