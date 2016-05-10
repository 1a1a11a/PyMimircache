import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import math

from mimircache.profiler.abstract.abstractLRUProfiler import abstractLRUProfiler


class basicLRUProfiler(abstractLRUProfiler):
    def __init__(self, cache_class, cache_size, bin_size, reader):
        super().__init__(cache_class, cache_size, bin_size, reader)

    def run(self):
        for record in self.reader:
            self.addOneTraceElement(record)
        self.calculate()

    def addOneTraceElement(self, element):
        super().addOneTraceElement(element)
        rank = self.cache.addElement(element)
        if rank != -1:
            self.HRC_bin[math.ceil(rank / self.bin_size)] += 1


if __name__ == "__main__":
    pass
