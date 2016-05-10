''' this module is used for all other cache replacement algorithms including LRU,
    but for LRU, we can use reuse distance for more efficient profiling, see basicLRUProfiler

'''
import math
import os

# deal with headless situation
# import matplotlib
# matplotlib.use('Agg')

from multiprocessing import Process, Pipe, Array
from os import path
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

from mimircache.cacheReader.plainReader import plainCacheReader

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

        if (self.cache_size != -1):
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
        self._auto_output_filename = os.path.basename(self.reader.file_loc).split('.')[
                                         0] + '_' + self.cache_class.__name__ + '_' + str(self.cache_size)

    def addOneTraceElement(self, element):
        super().addOneTraceElement(element)

        for i in range(len(self.pipe_list)):
            # print("send out: " + element)
            self.pipe_list[i][0].send(element)

        return

    def _addOneTraceElementSingleProcess(self, num_of_process, process_num, cache_class, cache_size_list, \
                                         cache_args, pipe, MRC_array):
        '''

        :param num_of_process:
        :param process_num:
        :param cache_class:
        :param cache_size_list: a list of different cache size dispached to this process
        :param cache_args: the extra argument (besides size) for instantiate a cache class
        :param pipe:            for receiving cache record from main process
        :param MRC_array:       storing MRC count for all cache sizes
        :return:
        '''
        cache_list = []
        for i in range(len(cache_size_list)):
            if len(cache_args) > 0:
                cache_list.append(cache_class(cache_size_list[i], *cache_args))
            else:
                cache_list.append(cache_class(cache_size_list[i]))

        element = pipe.recv()

        # TODO this part should be changed
        while element != 'END_1a1a11a_ENDMARKER':
            for i in range(len(cache_list)):
                # print("i = %d"%i)
                # cache_list[i].printCacheLine()
                # print('')
                if cache_list[i].addElement(element) == False:
                    MRC_array[i * num_of_process + process_num] += 1
            element = pipe.recv()
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

    def plotMRC(self):
        if not self.calculated:
            self.calculate()
        try:
            figure_MRC = plt.figure(1)
            line = plt.plot(range(self.bin_size, self.bin_size * (self.num_of_blocks + 1), self.bin_size), self.MRC)
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate/%")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig(
                os.path.basename(self.reader.file_loc).split('.')[0] + '_' + self.cache_class.__name__ + '_' + str(
                    self.cache_size) + '_MRC')
        except Exception as e:
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def plotHRC(self):
        if not self.calculated:
            self.calculate()
        try:
            figure_HRC = plt.figure(2)
            line = plt.plot(range(self.bin_size, self.bin_size * (self.num_of_blocks + 1), self.bin_size), self.HRC)
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate/%")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig(
                os.path.basename(self.reader.file_loc).split('.')[0] + '_' + self.cache_class.__name__ + '_' + str(
                    self.cache_size) + '_HRC')
            print("plot saved")
        except Exception as e:
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def outputMRC(self, file_loc="./"):
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

    def outputHRC(self, file_loc="./"):
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
            with open(self._auto_output_filename + '.HRC', 'w') as file:
                for x in self.HRC:
                    file.write(str(x) + '\n')
        return True

    def calculate(self):
        self.calculated = True
        for i in range(len(self.pipe_list)):
            self.pipe_list[i][0].send("END_1a1a11a_ENDMARKER")
            self.pipe_list[i][0].close()
        for i in range(len(self.process_list)):
            self.process_list[i].join()

        for i in range(len(self.MRC)):
            # print(self.MRC_array[i])
            self.MRC[i] = self.MRC_array[i] * 100 / self.num_of_trace_elements
        for i in range(len(self.HRC)):
            self.HRC[i] = 100 - self.MRC[i]

    def run(self):
        super().run()
        self.reader.reset()
        for i in self.reader:
            self.addOneTraceElement(i)
        # p.printMRC()
        self.outputHRC()
        self.plotHRC()


if __name__ == "__main__":
    import time
    import cProfile

    t1 = time.time()
    # r = plainCacheReader('../../data/test')
    r = plainCacheReader('../../data/parda.trace')

    # p = generalProfiler(LRU, 6000, 20, r, 48)
    # p = generalProfiler(ARC, (10, 0.5), 10, r, 1)
    p = generalProfiler(LFU_LRU, 800, 80, r, 4)
    # p = generalProfiler(ARC, 5, 5, r, 1)


    import pstats, io

    pr = cProfile.Profile()
    pr.enable()

    p.run()

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    t2 = time.time()
    print("TIME: %f" % (t2 - t1))

    # for i in r:
    #     # print(i)
    #     p.addOneTraceElement(i)
    # # p.printMRC()
    # p.plotHRC()
