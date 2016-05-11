import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import traceback
import abc
import math
from os import path

# deal with headless situation
# if Constants.headless:
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from mimircache.cache.LRU import LRU
from mimircache.profiler.abstract.abstractProfiler import profilerAbstract

DEBUG = False


class abstractLRUProfiler(profilerAbstract):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __init__(self, cache_class, cache_size, bin_size, reader):
        super().__init__(cache_class, cache_size, reader)
        self.bin_size = bin_size
        if (cache_size != -1):
            self.num_of_cache = self.num_of_blocks = math.ceil(cache_size / bin_size) + 1
            self.HRC = [0] * self.num_of_blocks
            self.MRC = [0] * self.num_of_blocks
            self.HRC_bin = [0] * self.num_of_blocks
        assert self.cache_class == LRU
        self.cache = self.cache_class(self.cache_size)
        self.calculated = False

    @abc.abstractclassmethod
    def addOneTraceElement(self, element):
        super().addOneTraceElement(element)
        self.calculated = False
        return

    def calculate(self):
        if DEBUG:
            print(self.HRC_bin)
        self.HRC[0] = self.HRC_bin[0] / self.num_of_trace_elements * 100
        for i in range(1, len(self.HRC)):
            self.HRC[i] = self.HRC[i - 1] + self.HRC_bin[i] / self.num_of_trace_elements * 100
        if DEBUG:
            print(self.HRC)
        self.calculated = True
        for i in range(len(self.HRC)):
            self.MRC[i] = 100 - self.HRC[i]

    def printMRC(self):
        if not self.calculated:
            self.calculate()
        for x in self.MRC:
            print(x)

    def printHRC(self):
        if not self.calculated:
            self.calculate()
        for x in self.HRC:
            print(x)

    def returnMRC(self):
        if not self.calculated:
            self.calculate()
        return self.MRC

    def returnHRC(self):
        if not self.calculated:
            self.calculate()
        return self.HRC

    def plotMRC(self, autosize=False, autosize_threshold=0.01):
        if not self.calculated:
            self.calculate()
        try:
            figure_MRC = plt.figure(1)
            # p_MRC = figure_MRC.add_subplot(1,1,1)

            # change the x-axis range according to threshhold
            if autosize:
                for i in range(len(self.MRC) - 1, 2, -1):
                    if (self.MRC[i - 1] - self.MRC[i]) / self.MRC[i] > autosize_threshold:
                        break
                num_of_blocks = i

            else:
                num_of_blocks = self.num_of_blocks
            if DEBUG:
                print("number of blocks: {}".format(num_of_blocks))
                print("total size: {}".format(self.bin_size * num_of_blocks))

            plt.plot(range(0, self.bin_size * num_of_blocks, self.bin_size), self.MRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate/%")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig("figure_temp_MRC.pdf")
        except Exception as e:
            plt.savefig("figure_temp_MRC.pdf")
            print("the plotting function is not wrong, is this a headless server?")
            print(e)
            traceback.print_exc()

    def plotHRC(self, autosize=False, autosize_threshold=0.001):
        if not self.calculated:
            self.calculate()
        try:
            figure_HRC = plt.figure(2)

            # change the x-axis range according to threshhold
            if autosize:
                for i in range(len(self.HRC) - 1, 2, -1):
                    if (self.HRC[i] - self.HRC[i - 1]) / self.HRC[i] > autosize_threshold:
                        break
                num_of_blocks = i

            else:
                num_of_blocks = self.num_of_blocks
            if DEBUG:
                print("number of blocks: {}".format(num_of_blocks))
                print("total size: {}".format(self.bin_size * num_of_blocks))

            line = plt.plot(range(0, self.bin_size * num_of_blocks, self.bin_size), self.HRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate/%")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig("figure_temp_HRC.pdf")
        except Exception as e:
            plt.savefig("figure_temp_HRC.pdf")
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def outputMRC(self, file_loc=""):
        ''' save the MRC list to the given location
        :param file_loc: can be a fileName or foldername
        :return:
        '''
        if not self.calculated:
            self.calculate()

        if path.isfile(file_loc):
            with open(file_loc, 'w') as file:
                for x in self.MRC:
                    file.write(str(x) + '\n')
        elif path.isdir(file_loc):
            with open(file_loc + '/' + str(self.cache_class.__name__) + '.MRC', 'w') as file:
                for x in self.MRC:
                    file.write(str(x) + '\n')
        return True

    def outputHRC(self, file_loc=""):
        ''' save the MRC list to the given location
        :param file_loc: can be a fileName or foldername
        :return:
        '''
        if not self.calculated:
            self.calculate()

        if path.isfile(file_loc):
            with open(file_loc, 'w') as file:
                for x in self.HRC:
                    file.write(str(x) + '\n')
        elif path.isdir(file_loc):
            with open(file_loc + '/' + str(self.cache_class.__name__) + '.HRC', 'w') as file:
                for x in self.HRC:
                    file.write(str(x) + '\n')
        return True
