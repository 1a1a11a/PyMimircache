# coding=utf-8
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


    @abc.abstractclassmethod
    def addOneTraceElement(self, element):
        self.num_of_trace_elements += 1
        return

