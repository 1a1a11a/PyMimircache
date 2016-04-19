import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import abc


class profilerAbstract(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __init__(self, cache_class, cache_size, reader):
        self.cache_class = cache_class
        self.cache_size = cache_size
        self.reader = reader
        self.num_of_trace_elements = 0
        # self.trace_file = open(self.trace_loc, 'r')

    @abc.abstractclassmethod
    def run(self):
        pass
        # for record in self.reader():
        #     self.addOneTraceElement(record)
        # return

    # def readData(self):
    #     # this method can be overriden in the subclass
    #     line_num = 0
    #     for line in self.trace_file:
    #         yield line.strip('\n').split()[4]
    #         line_num += 1
    #         if line_num%1000 == 0:
    #             print(line_num)

    @abc.abstractclassmethod
    def addOneTraceElement(self, element):
        self.num_of_trace_elements += 1
        return

        # @abc.abstractclassmethod
        # def printResult(self):
        #     return
        #
        # @abc.abstractclassmethod
        # def returnResult(self):
        #     return
        #
        # # @abc.abstractclassmethod
        # # def showGraph(self):
        # #     raise NotImplementedError("showGraph is not implemented")
        #
        # @abc.abstractclassmethod
        # def outputResult(self, file_loc=""):
        #     return


if __name__ == "__main__":
    pA = profilerAbstract("aaa", 10, "../data/LRU.MRC")
    for record in pA.readData():
        print(record)
