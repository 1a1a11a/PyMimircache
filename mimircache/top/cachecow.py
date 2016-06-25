from mimircache.cache.ARC import ARC
from mimircache.cache.FIFO import FIFO
from mimircache.cache.LFU_MRU import LFU_MRU
from mimircache.cache.LFU_RR import LFU_RR
from mimircache.cache.LRU import LRU
from mimircache.cache.MRU import MRU
from mimircache.cache.Random import Random
from mimircache.cache.S4LRU import S4LRU
from mimircache.cache.SLRU import SLRU
from mimircache.cache.clock import clock
from mimircache.cacheReader.csvReader import csvCacheReader
from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
from mimircache.profiler.LRUProfiler import LRUProfiler
from mimircache.profiler.generalProfiler import generalProfiler
from mimircache.profiler.heatmap import heatmap
import mimircache.const as const
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.cHeatmap import cHeatmap
from mimircache.const import *


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
        self.cacheclass_mapping['random'] = Random
        self.cacheclass_mapping['slru'] = SLRU
        self.cacheclass_mapping['s4lru'] = S4LRU

    def open(self, file_path):
        # assert os.path.exists(file_path), "data file does not exist"
        self.reader = plainCacheReader(file_path)
        return self.reader

    def csv(self, file_path, column):
        self.reader = csvCacheReader(file_path, column=column)
        return self.reader

    def vscsi(self, file_path):
        self.reader = vscsiCacheReader(file_path)
        return self.reader

    def set_size(self, size):
        assert isinstance(size, int), "size can only be an integer"
        self.cache_size = size

    def reset(self):
        self.reader.reset()

    def _profiler_pre_check(self, cache_class, **kargs):
        """
        check whether user has provided new cache size and data information
        :param kargs:
        :return:
        """
        reader = None
        if 'size' in kargs:
            size = kargs['size']
        else:
            size = self.cache_size

        if 'data' in kargs and 'dataType' in kargs:
            if kargs['dataType'] == 'plain':
                reader = plainCacheReader(kargs['data'])
            if kargs['dataType'] == 'csv':
                assert 'column' in kargs, "you didn't provide column number for csv reader"
                reader = csvCacheReader(kargs['data'], kargs['column'])
            if kargs['dataType'] == 'vscsi':
                reader = vscsiCacheReader(kargs['data'])
        elif 'reader' in kargs:
            reader = kargs['reader']
        else:
            reader = self.reader

        if cache_class.lower() != 'lru':
            assert size != -1, "you didn't provide size for cache"
        assert reader, "you didn't provide a reader nor data (data file and data type)"
        self.size = size
        self.reader = reader

        return size, reader

    def heatmap(self, cache_class, mode, interval, plot_type, **kwargs):
        """

        :param mode:
        :param interval:
        :param kargs:
        :return:
        """

        size, reader = self._profiler_pre_check(cache_class, **kwargs)
        if cache_class.lower() in const.c_available_cache:
            hm = cHeatmap()
            hm.heatmap(reader, mode, interval, plot_type, cache_size=size, **kwargs)
        else:
            hm = heatmap()
            hm.run(mode, interval, plot_type, reader, **kwargs)

    def differential_heatmap(self):
        pass

    def twoDplot(self):
        pass

    def profiler(self, cache_name, **kargs):
        """
        profiler
        :param cache_class:
        :param kargs:
        :return:
        """
        size, reader = self._profiler_pre_check(cache_name, **kargs)

        profiler = None
        cache_params = None
        bin_size = -1
        num_of_process = DEFAULT_NUM_OF_PROCESS

        # print(cache_class)

        if cache_name.lower() == "lru":
            profiler = LRUProfiler(reader, size)
        else:
            if 'bin_size' in kargs:
                bin_size = kargs['bin_size']

            if 'num_of_process' in kargs:
                num_of_process = kargs['num_of_process']

            if 'cache_params' in kargs:
                cache_params = kargs['cache_params']

            if isinstance(cache_name, str):
                if cache_name.lower() in c_available_cache:
                    profiler = cGeneralProfiler(reader, cache_name, size, bin_size, cache_params, num_of_process)
                else:
                    profiler = generalProfiler(reader, self.cacheclass_mapping[cache_name.lower()], size, bin_size,
                                               cache_params, num_of_process)
            else:
                profiler = generalProfiler(reader, cache_name, size, bin_size, cache_params, num_of_process)

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
    c = cachecow(size=48000)
    c.vscsi('../data/trace.vscsi')
    # m.heatmap('r', 100000000)

    # m.test()
    # m.open('../data/parda.trace')
    p = c.profiler("LRU")
    print(p.get_reuse_distance())
    p.plotHRC()

    p = c.profiler('optimal')
    print(p.get_hit_count())

    # c.heatmap('LRU', 'r', 100000000, "hit_rate_start_time_end_time")

    # m.heatmap('lru', 'r', 10000000, "hit_rate_start_time_end_time", data='../data/trace.vscsi', dataType='vscsi',
    #           num_of_process=8,  # LRU=False, cache='optimal',
    #           cache_size=2000, save=False, fixed_range=True, change_label=True,
    #           figname="r.png")

    # p = m.profiler('mru', bin_size=200, data='../data/parda.trace', dataType='plain', num_of_process=4)
    # p.run()
    # print(len(p.HRC))
    # print(p.HRC)
    # print(p.MRC[-1])
    # rdist = p.get_reuse_distance()
    # for i in rdist:
    #     print(i)
    # else:
    #     print('no element')
    # print(rdist)
    # print(rdist[-1])
    # p.plotHRC()
    # p.plotMRC()

    # line_num = 0
    # for i in m:
    #     print(i)
    #     line_num += 1
    #     if line_num > 10:
    #         break
