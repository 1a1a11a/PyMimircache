# coding=utf-8
"""
this is the multiTraceProfiler, 
user pass in a list of readers, the profiler obtain information on all readers
it also allows doing arithmetic calculation on the dataset, by that, it means, 
it can merge the reader traces into one, and also provide info on the merged dataset, 
this operation is called add/addition, 
other operations currently are not supported 
"""


from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.cacheReader.traceMixer import traceMixer
from mimircache.cacheReader.binaryReader import binaryReader
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


class multiTraceProfiler:
    """
    This is a profiler for profiling multiple traces
    """

    def __init__(self, readers, cache_name, cache_size=-1,
                 bin_size=-1, cache_params=None, num_of_threads=4):
        self.readers = readers
        self.cache_name = cache_name
        self.cache_size = cache_size
        self.bin_size = bin_size
        self.cache_params = cache_params
        self.num_of_threads = num_of_threads
        self.hrs = []

        if cache_name.lower() == "lru":
            self.bin_size = 1
            self.profilers = [LRUProfiler(reader, cache_size=cache_size) for reader in self.readers]
        else:
            assert cache_size != -1, "please provide cache size for non-LRU profiler"
            if bin_size == -1:
                self.bin_size = cache_size//100
            self.profilers = [cGeneralProfiler(reader, self.cache_name, self.cache_size, self.bin_size,
                                               cache_params=self.cache_params,
                                               num_of_threads=self.num_of_threads)
                              for reader in self.readers]


    def get_hit_rate(self):
        """
        get hit rate of all traces
        """
        if len(self.hrs):
            self.hrs.clear()
        # self.hrs = [0] * len(self.readers)
        # with ProcessPoolExecutor(max_workers=self.num_of_threads) as pool:
        #     results = {pool.submit(p.get_hit_rate):i for i, p in enumerate(self.profilers)}
        #     for future in as_completed(results):
        #         self.hrs[results[future]] = future.result()

        for p in self.profilers:
            self.hrs.append(p.get_hit_rate())
        return self.hrs


    def plotHRCs(self, op=None, clf=True, save=True, figname="HRC_multi_trace.pdf", **kwargs):
        if op is None:
            if len(self.hrs) == 0:
                self.get_hit_rate()
            for i in range(len(self.hrs)):
                label = self.readers[i].file_loc[self.readers[i].file_loc.rfind('/')+1:]
                plt.plot(np.arange(0, self.bin_size*(len(self.hrs[i])-2), self.bin_size),
                         self.hrs[i][:-2], label=label)

        elif op == "+" or op == "add" or op == "addition":
            assert 'mix_mode' in kwargs, "please provide mix_mode when doing trace mixing/addition"
            # the following only works on Linux
            mixed_trace = tempfile.NamedTemporaryFile()
            traceMixer.mix_asBinary(readers=self.readers, mix_mode="real_time", output=mixed_trace.name)
            mixed_reader = binaryReader(mixed_trace.name, init_params={"label": 1, "fmt": "<L"})
            if self.cache_name == "LRU":
                profiler = LRUProfiler(mixed_reader, cache_size=self.cache_size)
            else:
                profiler = cGeneralProfiler(mixed_reader, cache_name=self.cache_name,
                                            cache_size=self.cache_size, bin_size=self.bin_size,
                                            cache_params=self.cache_params, num_of_threads=self.num_of_threads)
            hr = profiler.get_hit_rate()
            plt.plot(np.arange(0, self.bin_size * (len(hr) - 2), self.bin_size),
                     hr[:-2], label="mixed")

        if save:
            plt.legend(loc="best")
            plt.xlabel("Cache size/items")
            plt.ylabel("Hit Ratio")
            plt.savefig(figname)
            if clf:
                plt.clf()



if __name__ == "__main__":
    import time
    from mimircache import *
    t1 = time.time()
    mtp = multiTraceProfiler([vscsiReader("/root/testDat/"+f) for f in os.listdir("/root/testDat/")],
                             "LRU", cache_size=200000)
    mtp.plotHRCs(clf=False)
    # mtp.plotHRCs(op="+", mix_mode="real_time")
    print(time.time() - t1)