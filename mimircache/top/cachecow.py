import os

from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader

from mimircache.cache.LRU import LRU
from mimircache.cache.ARC import ARC
from mimircache.cache.clock import clock
from mimircache.cache.FIFO import FIFO
from mimircache.cache.LFU_MRU import LFU_MRU
from mimircache.cache.LFU_RR import LFU_RR
from mimircache.cache.MRU import MRU
from mimircache.cache.RR import RR
from mimircache.cache.SLRU import SLRU
from mimircache.cache.S4LRU import S4LRU

from mimircache.profiler.generalProfiler import generalProfiler
from mimircache.profiler.pardaProfiler import pardaProfiler
from mimircache.profiler.pardaProfiler import parda_mode
from mimircache.profiler.heatmap import heatmap

class cachecow:
    def __init__(self, **kargs):
        self.cache_size = -1
        if 'size' in kargs:
            assert isinstance(kargs['size'], int), "size can only be an integer"
            self.cache_size = kargs['size']
        self.reader = None

        self.cacheclass_mapping = {}
        self.prepare_cacheclass_mapping()

    def prepare_cacheclass_mapping(self):
        self.cacheclass_mapping['lru'] = LRU
        self.cacheclass_mapping['arc'] = ARC
        self.cacheclass_mapping['clock'] = clock
        self.cacheclass_mapping['fifo'] = FIFO
        self.cacheclass_mapping['lfu_mru'] = LFU_MRU
        self.cacheclass_mapping['lfu_rr'] = LFU_RR
        self.cacheclass_mapping['mru'] = MRU
        self.cacheclass_mapping['rr'] = RR
        self.cacheclass_mapping['slru'] = SLRU
        self.cacheclass_mapping['s4lru'] = S4LRU

    def open(self, file_path):
        # assert os.path.exists(file_path), "data file does not exist"
        self.reader = plainCacheReader(file_path)

    def csv(self, file_path, column):
        self.reader = csvCacheReader(file_path, column=column)

    def vscsi(self, file_path):
        self.reader = vscsiCacheReader(file_path)

    def set_size(self, size):
        assert isinstance(size, int), "size can only be an integer"
        self.cache_size = size

    def reset(self):
        self.reader.reset()

    def _profiler_pre_check(self, **kargs):
        """
        check whether user has provided new cache size and data information
        :param kargs:
        :return:
        """
        if 'size' in kargs:
            size = kargs['size']
        else:
            size = self.cache_size

        if 'data' in kargs and 'dataType' in kargs:
            if kargs['dataType'] == 'plain':
                reader = plainCacheReader(kargs['data'])
            if kargs['dataType'] == 'csv':
                reader = csvCacheReader(kargs['data'])
            if kargs['dataType'] == 'vscsi':
                reader = vscsiCacheReader(kargs['data'])
        elif 'reader' in kargs:
            reader = kargs['reader']
        else:
            reader = self.reader

        assert size != -1, "you didn't provide size for cache"
        assert reader, "you didn't provide a reader nor data (data file and data type)"

        return size, reader

    def heatmap(self, mode, interval, **kargs):
        """

        :param mode:
        :param interval:
        :param kargs:
        :return:
        """

        size, reader = self._profiler_pre_check(kargs=kargs)

        hm = heatmap()
        hm.run(mode, interval, size, reader, kargs=kargs)


    def profiler(self, cache_class, **kargs):
        """
        profiler
        :param cache_class:
        :param kargs:
        :return:
        """
        size, reader = self._profiler_pre_check(kargs=kargs)

        profiler = None

        # print(cache_class)

        if cache_class.lower() == "lru":
            profiler = pardaProfiler(size, reader)
            # print(profiler)
        else:
            if 'bin_size' in kargs:
                bin_size = kargs['bin_size']
            else:
                raise RuntimeError("please give bin_size parameter if you want to profile on non-LRU cache "
                                   "replacement algorithm")
            if 'num_of_process' in kargs:
                num_of_process = kargs['num_of_process']
            else:
                num_of_process = 4

            profiler = generalProfiler(self.cacheclass_mapping[cache_class.lower()], size, bin_size,
                                       reader, num_of_process)


        return profiler

    def __iter__(self):
        assert self.reader, "you haven't provided a data file"
        return self.reader

    def next(self):
        return self.__next__()

    def __next__(self):  # Python 3
        return self.reader.next()

    def test(self):
        print(self.cache_size)


if __name__ == "__main__":
    m = cachecow(size=2000)
    m.vscsi('../data/trace_CloudPhysics_bin')
    # m.heatmap('r', 100000000)

    # m.test()
    m.open('../data/parda.trace')
    # p = m.profiler("LRU")

    p = m.profiler('mru', bin_size=10, data='../data/parda.trace', dataType='plain')
    p.run()
    # print(len(p.HRC))
    # print(p.HRC)
    # print(p.MRC[-1])
    rdist = p.get_reuse_distance()
    # for i in rdist:
    #     print(i)
    # else:
    #     print('no element')
    print(rdist)
    print(rdist[-1])
    # p.plotHRC()
    # p.plotMRC()

    # line_num = 0
    # for i in m:
    #     print(i)
    #     line_num += 1
    #     if line_num > 10:
    #         break
