from mimircache.cache.adaptiveSLRU import AdaptiveSLRU
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
from mimircache.profiler.cGeneralProfiler import cGeneralProfiler
from mimircache.profiler.cHeatmap import cHeatmap
from mimircache.const import *
from mimircache.profiler.twoDPlots import *


class cachecow:
    def __init__(self, **kwargs):
        self.reader = None
        self.cacheclass_mapping = {}
        self.prepare_cacheclass_mapping()

    def prepare_cacheclass_mapping(self):
        self.cacheclass_mapping['lru'] = LRU
        self.cacheclass_mapping['arc'] = AdaptiveSLRU
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

    def _profiler_pre_check(self, **kwargs):
        """
        check whether user has provided new cache size and data information
        :param kwargs:
        :return:
        """
        reader = None

        if 'num_of_process' in kwargs:
            num_of_process = kwargs['num_of_process']
        elif 'num_of_threads' in kwargs:
            num_of_process = kwargs['num_of_threads']
        else:
            num_of_process = DEFAULT_NUM_OF_PROCESS

        if 'data' in kwargs and 'dataType' in kwargs:
            if kwargs['dataType'] == 'plain':
                reader = plainCacheReader(kwargs['data'])
            if kwargs['dataType'] == 'csv':
                assert 'column' in kwargs, "you didn't provide column number for csv reader"
                reader = csvCacheReader(kwargs['data'], kwargs['column'])
            if kwargs['dataType'] == 'vscsi':
                reader = vscsiCacheReader(kwargs['data'])
        elif 'reader' in kwargs:
            reader = kwargs['reader']
        else:
            reader = self.reader

        assert reader, "you didn't provide a reader nor data (data file and data type)"
        self.reader = reader

        return reader, num_of_process

    def heatmap(self, mode, interval, plot_type, algorithm="LRU", cache_params=None, cache_size=-1, **kwargs):
        """

        :param mode:
        :param interval:
        :param kwargs: algorithm:
        :return:
        """

        reader, num_of_process = self._profiler_pre_check(**kwargs)

        l = ["avg_rd_start_time_end_time", "hit_rate_start_time_cache_size"]

        if plot_type in l:
            hm = heatmap()
        else:
            if algorithm.lower() in c_available_cache:
                hm = cHeatmap()

            else:
                hm = heatmap()

        hm.heatmap(reader, mode, interval, plot_type,
                   cache_size=cache_size,
                   algorithm=cache_alg_mapping[algorithm.lower()],
                   cache_params=cache_params,
                   **kwargs)

    def differential_heatmap(self, mode, interval, plot_type, algorithm1,
                             algorithm2="Optimal", cache_params1=None, cache_params2=None, cache_size=-1, **kwargs):
        """
        alg2 - alg1
        :param alg1:
        :param alg2:
        :param mode:
        :param interval:
        :param plot_type:
        :param kwargs:
        :return:
        """
        figname = 'differential_heatmap.png'
        if 'figname' in kwargs:
            figname = kwargs['figname']

        assert cache_size != -1, "you didn't provide size for cache"

        reader, num_of_process = self._profiler_pre_check(**kwargs)

        if algorithm1.lower() in c_available_cache and algorithm2.lower() in c_available_cache:
            hm = cHeatmap()
            hm.differential_heatmap(reader, mode, interval, plot_type,
                                    cache_size=cache_size,
                                    algorithm1=cache_alg_mapping[algorithm1.lower()],
                                    algorithm2=cache_alg_mapping[algorithm2.lower()],
                                    cache_params1=cache_params1,
                                    cache_params2=cache_params2,
                                    **kwargs)

        else:
            hm = heatmap()
            if algorithm1.lower() not in c_available_cache:
                xydict1 = hm.calculate_heatmap_dat(reader, mode, interval, plot_type,
                                                   cache_size=cache_size, algorithm=algorithm1,
                                                   cache_params=cache_params1,
                                                   **kwargs)[0]
            else:
                xydict1 = c_heatmap.heatmap(reader.cReader, mode, interval, plot_type,
                                            cache_size=cache_size, algorithm=algorithm1,
                                            cache_params=cache_params1,
                                            num_of_threads=num_of_process)

            if algorithm2.lower() not in c_available_cache:
                xydict2 = hm.calculate_heatmap_dat(reader, mode, interval, plot_type,
                                                   cache_size=cache_size, algorithm=algorithm2,
                                                   cache_params=cache_params2,
                                                   **kwargs)[0]
            else:
                xydict2 = c_heatmap.heatmap(reader.cReader, mode, interval, plot_type,
                                            cache_size=cache_size, algorithm=algorithm2,
                                            cache_params=cache_params2,
                                            num_of_threads=num_of_process)

            cHm = cHeatmap()
            text = "      differential heatmap\n      cache size: {},\n      cache type: {}-{},\n" \
                   "      time type: {},\n      time interval: {},\n      plot type: \n{}".format(
                cache_size, algorithm2, algorithm1, mode, interval, plot_type)

            x1, y1 = xydict1.shape
            x1 = int(x1 / 2.8)
            y1 /= 8
            if mode == 'r':
                cHm.set_plot_params('x', 'real_time', xydict=xydict1, label='start time (real)',
                                    text=(x1, y1, text))
                cHm.set_plot_params('y', 'real_time', xydict=xydict1, label='end time (real)')
            else:
                cHm.set_plot_params('x', 'virtual_time', xydict=xydict1, label='start time (virtual)',
                                    text=(x1, y1, text))
                cHm.set_plot_params('y', 'virtual_time', xydict=xydict1, label='end time (virtual)')

            cHm.draw_heatmap((xydict2 - xydict1) / xydict1, figname=figname)

    def profiler(self, algorithm, cache_params=None, cache_size=-1, **kwargs):
        """
        profiler
        :param cache_class:
        :param kwargs:
        :return:
        """
        reader, num_of_process = self._profiler_pre_check(**kwargs)

        profiler = None
        cache_params = None
        bin_size = -1

        if algorithm.lower() == "lru":
            profiler = LRUProfiler(reader, cache_size)
        else:
            assert cache_size != -1, "you didn't provide size for cache"
            if 'bin_size' in kwargs:
                bin_size = kwargs['bin_size']

            if isinstance(algorithm, str):
                if algorithm.lower() in c_available_cache:
                    profiler = cGeneralProfiler(reader, cache_alg_mapping[algorithm.lower()], cache_size, bin_size,
                                                cache_params, num_of_process)
                else:
                    profiler = generalProfiler(reader, self.cacheclass_mapping[algorithm.lower()], cache_size, bin_size,
                                               cache_params, num_of_process)
            else:
                profiler = generalProfiler(reader, algorithm, cache_size, bin_size, cache_params, num_of_process)

        return profiler

    def __iter__(self):
        assert self.reader, "you haven't provided a data file"
        return self.reader

    def next(self):
        return self.__next__()

    def __next__(self):  # Python 3
        return self.reader.next()

    def __del__(self):
        if self.reader:
            self.reader.close()

    def twoDPlot(self, mode, time_interval, plot_type, **kwargs):
        if plot_type == 'cold_miss':
            cold_miss_2d(self.reader, mode, time_interval)
        elif plot_type == 'request_num':
            request_num_2d(self.reader, mode, time_interval)
        else:
            print("currently don't support your specified plot_type: " + str(plot_type))



if __name__ == "__main__":
    CACHE_SIZE = 2000
    c = cachecow()
    c.vscsi('../data/trace.vscsi')
    # m.heatmap('r', 100000000)

    # m.test()
    # m.open('../data/parda.trace')
    p = c.profiler("LRU")
    print(p.get_reuse_distance())
    p.plotHRC()

    p = c.profiler('optimal', cache_size=CACHE_SIZE)
    print(p.get_hit_count())

    c.twoDPlot('v', 1000, "cold_miss")

    c.heatmap('r', 1000000, "hit_rate_start_time_end_time", "LRU", cache_size=CACHE_SIZE, num_of_process=8,
              figname="abc.png")

    c.differential_heatmap('r', 1000000, "hit_rate_start_time_end_time", "LRU", cache_size=CACHE_SIZE, num_of_process=8,
                           figname='2.png')

    c.differential_heatmap('r', 10000000, "hit_rate_start_time_end_time", "LRU", "ARC", cache_size=CACHE_SIZE,
                           num_of_process=8, figname='3.png')

    # c.differential_heatmap('r', 1000000, "hit_rate_start_time_end_time", "LRU", num_of_process=8, figname='2.png')

    TIME_INTERVAL = 10000000000
    from mimircache.cacheReader.vscsiReader import vscsiCacheReader

    PATH = '/run/shm/traces/'
    c = cachecow()

    for f in os.listdir(PATH):
        if f.endswith('vscsitrace'):
            # reader = vscsiCacheReader(PATH + f)
            c.vscsi(PATH + f)
            c.heatmap('r', TIME_INTERVAL, "rd_distribution",
                      figname='0627_rd_distribution/' + f + '_rd_distribution_' + str(TIME_INTERVAL) + '.png')
            # request_num_2d(reader, 'r', TIME_INTERVAL,
            #                figname='0627_rd_distribution/' + f + '_rd_distribution_' + str(TIME_INTERVAL) + '.png')
