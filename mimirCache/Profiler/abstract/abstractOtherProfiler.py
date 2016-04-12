''' this module is used for all other cache replacement algorithms excluding LRU,
    for LRU, see getMRCAbstractLRU
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import abc
import math
from multiprocessing import Process, Pipe, Array
from os import path

# deal with headless situation
# if Constants.headless:
#     import matplotlib
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mimirCache.Cache.LRU import LRU
from mimirCache.Profiler.abstract.abstractProfiler import profilerAbstract

debug = False


class abstractOtherProfiler(profilerAbstract):

    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __init__(self, cache_class, cache_size, bin_size, reader):
        super().__init__(cache_class, cache_size, reader)

        self.bin_size = bin_size

        if (cache_size != -1):
            self.num_of_cache = self.num_of_blocks = math.ceil(cache_size / bin_size) + 1
            self.HRC = [0] * self.num_of_blocks
            self.MRC = [0] * self.num_of_blocks

        self.cache = None
        self.cache_list = None
        if debug:
            print(type(self.cache_class))
            print(type(LRU))
            print(self.cache_class == LRU)
        self.process_list = []
        self.cache_distribution = [[0]] * self.num_of_process

        # shared memory for storing MRC count
        self.MRC_array = Array('i', range(self.num_of_blocks))

        # dispatch different cache size to different processes
        for i in range(self.num_of_blocks):
            if i < self.num_of_process:
                self.cache_distribution[i % self.num_of_process][0] = i
            else:
                self.cache_distribution[i % self.num_of_process].append(i)

            # build pipes for communication between main process and children process
            # the pipe mainly sends element from main process to children

        self.pipe_list = []

        for i in range(self.num_of_process):
            self.pipe_list.append(Pipe())
            p = Process(target=self._addOneTraceElementSingleProcess,
                        args=(self.num_of_process, i, self.cache_class, self.cache_distribution[i],
                              self.pipe_list[i], self.MRC_array))
            self.process_list.append(p)
            p.start()

        self.calculated = False



    @abc.abstractclassmethod
    def addOneTraceElement(self, element):
        super().addOneTraceElement(element)
        if self.cache_class != LRU:
            for i in range(len(self.pipe_list)):
                self.pipe_list[i].send(element)
        return

    def _addOneTraceElementSingleProcess(self, num_of_process, process_num, cache_class, cache_size_list, pipe,
                                         MRC_array):
        cache_list = []
        for i in range(len(cache_size_list)):
            cache_list.append(cache_class(cache_size_list[i]))
        element = pipe.recv()

        # TODO this part should be changed
        while element != 'END_1a1a11a_ENDMARKER':
            for i in len(cache_list):
                if cache_list[i].addElement(element) == -1:
                    MRC_array[(i - 1) * num_of_process + process_num] += 1
            element = pipe.recv()

    def printMRC(self):
        super().printResult()
        if not self.calculated:
            self.calculate()
        for x in self.MRC:
            print(x)

    def printHRC(self):
        super().printResult()
        if not self.calculated:
            self.calculate()
        for x in self.HRC:
            print(x)

    def returnMRC(self):
        super().returnResult()
        if not self.calculated:
            self.calculate()
        return self.MRC

    def returnHRC(self):
        super().returnResult()
        if not self.calculated:
            self.calculate()
        return self.HRC

    def plotMRC(self):
        if not self.calculated:
            self.calculate()
        try: 
            figure_MRC = plt.figure(1)
            line = plt.plot(range(0, self.bin_size*self.num_of_blocks, self.bin_size), self.MRC)
            plt.xlabel("Cache Size")
            plt.ylabel("Miss Rate/%")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig('temp_MRC')
        except Exception as e:
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def plotHRC(self):
        if not self.calculated:
            self.calculate()
        try:
            figure_HRC = plt.figure(2)
            line = plt.plot(range(0, self.bin_size * self.num_of_blocks, self.bin_size), self.HRC)
            plt.xlabel("Cache Size")
            plt.ylabel("Hit Rate/%")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig('temp_HRC')
        except Exception as e:
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
                    file.write(str(x)+'\n')
        elif path.isdir(file_loc):
            with open(file_loc + '/' + str(self.cache_class.__name__) + '.MRC', 'w') as file:
                for x in self.MRC:
                    file.write(str(x)+'\n')
        return True

    def outputHRC(self, file_loc=""):
        ''' save the HRC list to the given location
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



    @abc.abstractclassmethod
    def calculate(self):
        self.calculated = True
        # if self.cache_class == LRU:
        #     for i in range(len(self.HRC)):
        #         self.MRC[i] = 100 - self.HRC[i] * 100 / self.num_of_trace_elements
        # else:
        for i in range(len(self.MRC)):
            self.MRC[i] = self.MRC_array[i] * 100 / self.num_of_trace_elements
        for i in range(len(self.HRC)):
            self.HRC[i] = 100 - self.MRC[i]
