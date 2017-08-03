# coding=utf-8
import os
import mimircache.c_LRUProfiler as c_LRUProfiler
from mimircache.cacheReader.abstractReader import cacheReaderAbstract
import matplotlib.pyplot as plt
from mimircache.utils.printing import *
from matplotlib.ticker import FuncFormatter

class LRUProfiler:
    all = ["get_hit_count", "get_hit_rate", "get_miss_rate", "get_reuse_distance",
           "plotMRC", "plotHRC", "get_best_cache_sizes"]

    def __init__(self, reader, cache_size=-1, cache_params=None):
        self.cache_size = cache_size
        self.reader = reader
        if cache_params is not None and 'block_unit_size' in cache_params:
            self.with_size = True
            self.block_unit_size = cache_params["block_unit_size"]
        else:
            self.with_size = False

        assert isinstance(reader, cacheReaderAbstract), \
            "you provided an invalid cacheReader: {}".format(reader)


    def addOneTraceElement(self, element):
        # do not need this function in this profiler
        pass

    def save_reuse_dist(self, file_loc, rd_type):
        assert rd_type == 'rd' or rd_type == 'frd', "please provide a valid reuse distance type, currently support rd and frd"
        c_LRUProfiler.save_reuse_dist(self.reader.cReader, file_loc, rd_type)

    def load_reuse_dist(self, file_loc, rd_type):
        assert rd_type == 'rd' or rd_type == 'frd', "please provide a valid reuse distance type, currently support rd and frd"
        c_LRUProfiler.load_reuse_dist(self.reader.cReader, file_loc, rd_type)

    def get_hit_count(self, **kargs):
        """
        0~size(included) are for counting rd=0~size-1, size+1 is
        out of range, size+2 is cold miss, so total is size+3 buckets
        :param kargs:
        :return:
        """
        if 'cache_size' not in kargs:
            kargs['cache_size'] = self.cache_size
        if self.with_size:
            print("not supported yet")
            return None
        else:
            hit_count = c_LRUProfiler.get_hit_count_seq(self.reader.cReader, **kargs)
        return hit_count

    def get_hit_rate(self, **kwargs):
        """

        :param kwargs:
        :return: a numpy array of CACHE_SIZE+3, 0~CACHE_SIZE corresponds to hit rate of size 0~CACHE_SIZE,
         size 0 should always be 0, CACHE_SIZE+1 is out of range, CACHE_SIZE+2 is cold miss,
         so total is CACHE_SIZE+3 buckets
        """
        kargs = {}
        if 'cache_size' not in kwargs:
            kargs['cache_size'] = self.cache_size
        else:
            kargs['cache_size'] = kwargs['cache_size']
        if 'begin' in kwargs:
            kargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            kargs['end'] = kwargs['end']

        if self.with_size:
            hit_rate = c_LRUProfiler.get_hit_rate_with_size(self.reader.cReader,
                                                            block_unit_size=self.block_unit_size, **kargs)
        else:
            hit_rate = c_LRUProfiler.get_hit_rate_seq(self.reader.cReader, **kargs)
        return hit_rate


    def get_hit_rate_shards(self, sample_ratio=0.01, **kwargs):
        from mimircache.cacheReader.tracePreprocesser import tracePreprocessor
        kargs = {}
        if 'cache_size' not in kwargs:
            kargs['cache_size'] = self.cache_size
        else:
            kargs['cache_size'] = kwargs['cache_size']

        pp = tracePreprocessor(self.reader)
        N1, N2, traceName, fmt = pp.prepare_for_shards(sample_ratio=sample_ratio, has_time=False)
        correction = N2 - N1
        print("correction: {}".format(correction))
        # correction = 0
        tempReader = binaryReader(traceName, init_params={"label":1, "fmt": fmt})

        if self.with_size:
            print("not supported yet")
            return None
        else:
            hit_rate = c_LRUProfiler.get_hit_rate_seq_shards(tempReader.cReader, sample_ratio=sample_ratio,
                                                       correction=correction, **kargs)
        return hit_rate

    def get_miss_rate(self, **kargs):
        if 'cache_size' not in kargs:
            kargs['cache_size'] = self.cache_size
        if self.with_size:
            print("not supported yet")
            return None
        else:
            miss_rate = c_LRUProfiler.get_miss_rate_seq(self.reader.cReader, **kargs)
        return miss_rate

    def get_reuse_distance(self, **kargs):
        if self.with_size:
            print("not supported yet")
            return None
        else:
            rd = c_LRUProfiler.get_reuse_dist_seq(self.reader.cReader, **kargs)
        return rd

    def get_future_reuse_distance(self, **kargs):
        if self.with_size:
            print("not supported yet")
            return None
        else:
            frd = c_LRUProfiler.get_future_reuse_dist(self.reader.cReader, **kargs)
        return frd

    def plotMRC(self, figname="MRC.png", auto_resize=False, threshold=0.98, **kwargs):
        print("not updated")
        EXTENTION_LENGTH = 1024
        MRC = self.get_miss_rate(**kwargs)
        try:
            stop_point = len(MRC) - 3
            if self.cache_size == -1 and 'cache_size' not in kwargs and auto_resize:
                for i in range(len(MRC) - 3, 0, -1):
                    if MRC[i] >= MRC[-3] / threshold:
                        stop_point = i
                        break
                if stop_point + EXTENTION_LENGTH < len(MRC) - 3:
                    stop_point += EXTENTION_LENGTH
                else:
                    stop_point = len(MRC) - 3

            plt.plot(MRC[:stop_point])
            plt.xlabel("cache Size")
            plt.ylabel("Miss Ratio")
            plt.title('Miss Ratio Curve', fontsize=18, color='black')
            if not 'no_save' in kwargs or not kwargs['no_save']:
                plt.savefig(figname, dpi=600)
                INFO("plot is saved at the same directory")
            try:
                plt.show()
            except:
                pass
            plt.clf()
            return stop_point
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function is not wrong, is this a headless server? {}".format(e))

    def plotHRC(self, figname="HRC.png", auto_resize=False, threshold=0.98, **kwargs):
        EXTENTION_LENGTH = 1024
        HRC = self.get_hit_rate(**kwargs)
        try:
            stop_point = len(HRC) - 3
            if self.cache_size == -1 and 'cache_size' not in kwargs and auto_resize:
                for i in range(len(HRC) - 3, 0, -1):
                    if HRC[i] <= HRC[-3] * threshold:
                        stop_point = i
                        break
                if stop_point + EXTENTION_LENGTH < len(HRC) - 3:
                    stop_point += EXTENTION_LENGTH
                else:
                    stop_point = len(HRC) - 3

            plt.xlim(0, stop_point)
            plt.plot(HRC[:stop_point])
            if self.with_size:
                plt.gca().xaxis.set_major_formatter(FuncFormatter(
                        lambda x, p: int(x * self.block_unit_size//1024//1024)))
                plt.xlabel("Cache Size (MB)")
            else:
                plt.xlabel("Cache Size")

            plt.ylabel("Hit Ratio")
            plt.title('Hit Ratio Curve', fontsize=18, color='black')
            if not 'no_save' in kwargs or not kwargs['no_save']:
                plt.savefig(figname, dpi=600)
                INFO("plot is saved at the same directory")
            try:
                plt.show()
            except:
                pass
            plt.clf()
            return HRC[:stop_point]
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function is not wrong, is this a headless server? {}".format(e))

    def plotHRC_withShards(self, figname="HRC.png", auto_resize=False, threshold=0.98, **kwargs):
        print("not updated yet")
        EXTENTION_LENGTH = 1024
        HRC = self.get_hit_rate(**kwargs)
        HRC_shards = self.get_hit_rate_shards(**kwargs)
        try:
            stop_point = len(HRC) - 3
            if self.cache_size == -1 and 'cache_size' not in kwargs and auto_resize:
                for i in range(len(HRC) - 3, 0, -1):
                    if HRC[i] <= HRC[-3] * threshold:
                        stop_point = i
                        break
                if stop_point + EXTENTION_LENGTH < len(HRC) - 3:
                    stop_point += EXTENTION_LENGTH
                else:
                    stop_point = len(HRC) - 3

            if len(HRC_shards)-3 < stop_point:
                stop_point = len(HRC_shards) - 3
            print("{} len {}:{}".format(self.reader.file_loc, len(HRC), len(HRC_shards)))
            plt.xlim(0, stop_point)
            plt.plot(HRC[:stop_point], label="LRU")
            plt.plot(HRC_shards[:stop_point], label="LRU_shards")
            plt.xlabel("cache Size")
            plt.ylabel("Hit Ratio")
            plt.legend(loc="best")
            plt.title('Hit Ratio Curve', fontsize=18, color='black')
            if not 'no_save' in kwargs or not kwargs['no_save']:
                plt.savefig(figname, dpi=600)
                INFO("plot is saved at the same directory")
            try:
                plt.show()
            except:
                pass
            plt.clf()
            return stop_point
        except Exception as e:
            plt.savefig(figname)
            WARNING("the plotting function is not wrong, is this a headless server? {}".format(e))

    def __del__(self):
        if os.path.exists('temp.dat'):
            os.remove('temp.dat')


