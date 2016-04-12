import os

from mimirCache.CacheReader.csvReader import csvCacheReader
from mimirCache.CacheReader.plainReader import plainCacheReader
from mimirCache.CacheReader.vscsiReader import vscsiCacheReader

from mimirCache.Cache.LRU import LRU

from mimirCache.Profiler.pardaProfiler import pardaProfiler
from mimirCache.Profiler.pardaProfiler import parda_mode


class cacheCow:
    def __init__(self, **kargs):
        self.cache_size = -1
        if 'size' in kargs:
            assert isinstance(kargs['size'], int), "size can only be an integer"
            self.cache_size = kargs['size']
        self.reader = None

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

    def profiler(self, cache_class, **kargs):

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
        else:
            reader = self.reader

        assert size != -1, "you didn't provide size for cache"
        assert reader, "you didn't provide data file or data type"

        profiler = None

        # print(cache_class)
        # print(LRU)
        if cache_class == LRU:
            profiler = pardaProfiler(LRU, size, reader)
            # print(profiler)

        else:
            # TODO
            # parameters about cache class can be passed in kargs
            pass

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
    m = cacheCow(size=10000)
    # m.test()
    m.open('../Data/parda.trace')
    p = m.profiler(LRU)
    # p = m.profiler(LRU, data='../Data/parda.trace', dataType='plain')
    # p.run()
    # print(len(p.HRC))
    # print(p.MRC[-1])
    rdist = p.get_reuse_distance()
    print(rdist[-2])
    # p.plotHRC()
    # p.plotMRC()

    # line_num = 0
    # for i in m:
    #     print(i)
    #     line_num += 1
    #     if line_num > 10:
    #         break
