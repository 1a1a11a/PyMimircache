""" this module is used for all other cache replacement algorithms excluding LRU(LRU also works, but slow compared to
    using pardaProfiler),
"""
# -*- coding: utf-8 -*-


import math
import os

# deal with headless situation
import traceback

import matplotlib
# matplotlib.use('Agg')

from multiprocessing import Process, Pipe, Array
from os import path

from mimircache.cacheReader.vscsiReader import vscsiCacheReader

from mimircache.cache.ARC import ARC
from mimircache.cache.clock import clock
from mimircache.cache.FIFO import FIFO
from mimircache.cache.LFU_LRU__NEED_OPTIMIZATION import LFU_LRU
from mimircache.cache.LFU_MRU import LFU_MRU
from mimircache.cache.LFU_RR import LFU_RR
from mimircache.cache.LRU import LRU
from mimircache.cache.MRU import MRU
from mimircache.cache.RR import RR
from mimircache.cache.SLRU import SLRU
from mimircache.cache.S4LRU import S4LRU
from mimircache.cache.Optimal import optimal

from mimircache.cacheReader.plainReader import plainCacheReader

import mimircache.c_generalProfiler as c_generalProfiler

import matplotlib.pyplot as plt

from mimircache.profiler.abstract.abstractProfiler import profilerAbstract

debug = True


class generalProfiler(profilerAbstract):
    def __init__(self, cache_class, cache_params, bin_size, reader, num_of_process):
        if isinstance(cache_params, tuple):
            super(generalProfiler, self).__init__(cache_class, cache_params[0], reader)
            self.cache_size = cache_params[0]
            self.cache_params = cache_params[1:]

        elif isinstance(cache_params, int):
            super(generalProfiler, self).__init__(cache_class, cache_params, reader)
            self.cache_size = cache_params
            self.cache_params = ()

        else:
            raise RuntimeError("cache parameter not corrected, given: {}, expect either int as cache size or \
            tuple for full argument".format(cache_params))

        self.bin_size = bin_size
        self.num_of_process = num_of_process

        if self.cache_size != -1:
            if int(self.cache_size / bin_size) * bin_size == self.cache_size:
                self.num_of_cache = self.num_of_blocks = int(self.cache_size / bin_size)
            else:
                self.num_of_cache = self.num_of_blocks = math.ceil(self.cache_size / bin_size) + 1
            self.HRC = [0] * self.num_of_blocks
            self.MRC = [0] * self.num_of_blocks

        self.cache = None
        self.cache_list = None

        self.process_list = []
        self.cache_distribution = [[] for _ in range(self.num_of_process)]

        # shared memory for storing MRC count
        self.MRC_array = Array('i', range(self.num_of_blocks))
        for i in range(len(self.MRC_array)):
            self.MRC_array[i] = 0

        # dispatch different cache size to different processes
        for i in range(self.num_of_blocks):
            self.cache_distribution[i % self.num_of_process].append((i + 1) * bin_size)
            # print(self.cache_distribution)

            # build pipes for communication between main process and children process
            # the pipe mainly sends element from main process to children

        self.pipe_list = []

        for i in range(self.num_of_process):
            self.pipe_list.append(Pipe())
            p = Process(target=self._addOneTraceElementSingleProcess,
                        args=(self.num_of_process, i, self.cache_class, self.cache_distribution[i],
                              self.cache_params, self.pipe_list[i][1], self.MRC_array))
            self.process_list.append(p)
            p.start()

        self.calculated = False
        self._auto_output_filename = str(os.path.basename(self.reader.file_loc).split('.')[
                                             0]) + '_' + self.cache_class.__name__ + '_' + str(self.cache_size)

    def addOneTraceElement(self, element):
        super().addOneTraceElement(element)

        for i in range(len(self.pipe_list)):
            # print("send out: " + element)
            self.pipe_list[i][0].send(element)

        return

    # noinspection PyMethodMayBeStatic
    def _addOneTraceElementSingleProcess(self, num_of_process, process_num, cache_class, cache_size_list,
                                         cache_args, pipe, MRC_array):
        """

        :param num_of_process:
        :param process_num:
        :param cache_class:
        :param cache_size_list: a list of different cache size dispached to this process
        :param cache_args: the extra argument (besides size) for instantiate a cache class
        :param pipe:            for receiving cache record from main process
        :param MRC_array:       storing MRC count for all cache sizes
        :return:
        """
        cache_list = []
        for i in range(len(cache_size_list)):
            if len(cache_args) > 0:
                cache_list.append(cache_class(cache_size_list[i], *cache_args))
            else:
                cache_list.append(cache_class(cache_size_list[i]))

        elements = pipe.recv()

        # TODO this part should be changed
        while elements[-1] != 'END_1a1a11a_ENDMARKER':
            for i in range(len(cache_list)):
                # print("i = %d"%i)
                # cache_list[i].printCacheLine()
                # print('')
                for element in elements:
                    if not cache_list[i].addElement(element):
                        MRC_array[i * num_of_process + process_num] += 1
            elements = pipe.recv()
            # print(element)
            # print(cache_list)

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
            # figure_MRC = plt.figure(1)
            plt.clf()
            # p_MRC = figure_MRC.add_subplot(1,1,1)

            # change the x-axis range according to threshhold
            if autosize:
                for i in range(len(self.MRC) - 1, 2, -1):
                    if (self.MRC[i - 1] - self.MRC[i]) / self.MRC[i] > autosize_threshold:
                        break
                num_of_blocks = i

            else:
                num_of_blocks = self.num_of_blocks

            plt.plot(range(0, self.bin_size * num_of_blocks, self.bin_size), self.MRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate/%")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig("figure_temp_MRC.png")
        except Exception as e:
            plt.savefig("figure_temp_MRC.png")
            print("the plotting function is not wrong, is this a headless server?")
            print(e)
            traceback.print_exc()

    def plotHRC(self, autosize=False, autosize_threshold=0.001):
        if not self.calculated:
            self.calculate()
        try:
            plt.clf()
            # figure_HRC = plt.figure(2)

            # change the x-axis range according to threshhold
            if autosize:
                for i in range(len(self.HRC) - 1, 2, -1):
                    if (self.HRC[i] - self.HRC[i - 1]) / self.HRC[i] > autosize_threshold:
                        break
                num_of_blocks = i

            else:
                num_of_blocks = self.num_of_blocks

            plt.plot(range(0, self.bin_size * num_of_blocks, self.bin_size), self.HRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate/%")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig("figure_temp_HRC.png")
        except Exception as e:
            plt.savefig("figure_temp_HRC.png")
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def outputMRC(self, file_loc="./"):
        """ save the MRC list to the given location
        :param file_loc: can be a fileName or foldername
        :return:
        """
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

    def outputHRC(self, file_loc="./"):
        """ save the HRC list to the given location
        :param file_loc: can be a fileName or foldername
        :return:
        """
        if not self.calculated:
            self.calculate()

        if path.isfile(file_loc):
            with open(file_loc, 'w') as file:
                for x in self.HRC:
                    file.write(str(x) + '\n')
        elif path.isdir(file_loc):
            with open(self._auto_output_filename + '.HRC', 'w') as file:
                for x in self.HRC:
                    file.write(str(x) + '\n')
        return True

    def calculate(self):
        self.calculated = True
        for i in range(len(self.pipe_list)):
            self.pipe_list[i][0].send(["END_1a1a11a_ENDMARKER"])
            self.pipe_list[i][0].close()
        for i in range(len(self.process_list)):
            self.process_list[i].join()

        for i in range(len(self.MRC)):
            # print(self.MRC_array[i])
            self.MRC[i] = self.MRC_array[i] * 100 / self.num_of_trace_elements
        for i in range(len(self.HRC)):
            self.HRC[i] = 100 - self.MRC[i]

    def run(self, buffer_size=10000):
        super().run()
        self.reader.reset()
        l = []
        for i in self.reader:
            l.append(i)
            if len(l) == buffer_size:
                self.add_elements(l)
                l.clear()
                # self.addOneTraceElement(i)
        # p.printMRC()
        if len(l) > 0:
            self.add_elements(l)
        self.calculate()
        # self.outputHRC()
        # self.plotHRC()

    def add_elements(self, elements):
        for element in elements:
            super().addOneTraceElement(element)

        for i in range(len(self.pipe_list)):
            # print("send out: " + element)
            self.pipe_list[i][0].send(elements)

        return


if __name__ == "__main__":
    import time

    t1 = time.time()
    # r = plainCacheReader('../../data/test')
    # r = plainCacheReader('../data/parda.trace')
    r = vscsiCacheReader('../data/trace.vscsi')

    # p = generalProfiler(LRU, 6000, 20, r, 48)
    # p = generalProfiler(ARC, (10, 0.5), 10, r, 1)
    # p = generalProfiler(LRU, 800, 5, r, 4)
    # p = generalProfiler(ARC, 5, 5, r, 1)
    # p = generalProfiler(optimal, (2000, r), 500, r, 8)
    # p.run()
    # print(p.HRC)
    t2 = time.time()
    print("TIME: %f" % (t2 - t1))
    t1 = time.time()

    hr = c_generalProfiler.get_hit_rate(r.cReader, 2000, "Optimal", bin_size=500, num_of_threads=8)
    print(hr)

    t2 = time.time()
    print("TIME: %f" % (t2 - t1))

    # for i in r:
    #     # print(i)
    #     p.addOneTraceElement(i)
    # # p.printMRC()
    # p.plotHRC()
