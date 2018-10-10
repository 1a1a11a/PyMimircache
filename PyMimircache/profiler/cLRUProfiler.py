# coding=utf-8

"""
this module deals with LRU related profiling,
it uses O(nlogn) algorithm to profile without sampling
Current implementation is using single thread, TODO: implement parallel parda

Author: Jason Yang <peter.waynechina@gmail.com> 2016/07

"""
import os
import socket
from PyMimircache.const import INTERNAL_USE
from PyMimircache.const import ALLOW_C_MIMIRCACHE
from PyMimircache.const import INSTALL_PHASE
if ALLOW_C_MIMIRCACHE and not INSTALL_PHASE:
    import PyMimircache.CMimircache.LRUProfiler as c_LRUProfiler

from PyMimircache.cacheReader.binaryReader import BinaryReader
from PyMimircache.cacheReader.abstractReader import AbstractReader
from PyMimircache.utils.printing import *

class CLRUProfiler:
    """
    LRUProfiler in C

    """
    all = ["get_hit_count",
           "get_hit_ratio",
           "get_reuse_distance",
           "plotHRC",
           "save_reuse_dist",
           "load_reuse_dist",
           "use_precomputedRD"]

    def __init__(self, reader, cache_size=-1, cache_params=None, **kwargs):
        """
        initialize a CLRUProfiler

        :param reader: reader for feeding data into profiler
        :param cache_size: size of cache, if -1, then use max possible size
        :param cache_params: parameters about cache, such as block_unit_size
        :param kwargs: no_load_rd
        """

        # make sure reader is valid
        assert isinstance(reader, AbstractReader), \
            "you provided an invalid cacheReader: {}".format(reader)

        self.cache_size = cache_size
        self.reader = reader
        self.cache_params = cache_params
        if self.cache_params is None:
            self.cache_params = {}

        # check whether user want to profling with size
        self.block_unit_size = self.cache_params.get("block_unit_size", 0)
        block_unit_size_names = {"unit_size", "block_size", "chunk_size"}
        for name in block_unit_size_names:
            if name in self.cache_params:
                self.block_unit_size = cache_params[name]
                break


        # this is for deprecated functions, as old version uses hit/miss rate instead of hit/miss ratio
        self.get_hit_rate = self.get_hit_ratio


        # INTERNAL USE to cache intermediate reuse distance
        self.already_load_rd = False
        if INTERNAL_USE and not kwargs.get("no_load_rd", False) and \
                socket.gethostname().lower() in ["master", "node2", "node3"] and \
                ".." not in reader.file_loc and os.path.dirname(reader.file_loc):
            if not reader.already_load_rd:
                self.use_precomputedRD()
                self.already_load_rd = True


    def save_reuse_dist(self, file_loc, rd_type):
        """
        save reuse distance to file_loc
        allowed reuse distance including normal reuse distance (rd),
        future/forward reuse distance (frd)
        :param file_loc:
        :param rd_type:
        :return:
        """
        assert rd_type == 'rd' or rd_type == 'frd', \
            "please provide a valid reuse distance type, currently support rd and frd"
        c_LRUProfiler.save_reuse_dist(self.reader.c_reader, file_loc, rd_type)

    def load_reuse_dist(self, file_loc, rd_type):
        """
        load reuse distance from file_loc
        allowed reuse distance including normal reuse distance (rd),
        future/forward reuse distance (frd)

        :param file_loc:
        :param rd_type:
        :return:
        """
        assert rd_type == 'rd' or rd_type == 'frd', \
            "please provide a valid reuse distance type, currently support rd and frd"
        if not os.path.exists(file_loc):
            WARNING("pre-computed reuse distance file does not exist")
        c_LRUProfiler.load_reuse_dist(self.reader.c_reader, file_loc, rd_type)
        self.reader.already_load_rd = True

    def _del_reuse_dist_file(self):
        """
        an internal function that deletes the pre-computed reuse distance
        """

        assert INTERNAL_USE == True, "this function is only used internally"
        rd_dat_path = self.reader.file_loc.replace("/home/jason/ALL_DATA/",
                                                   "/research/jason/preComputedData/RD/")
        rd_dat_path = rd_dat_path.replace("/home/cloudphysics/traces/",
                                          "/research/jason/preComputedData/RD/cphyVscsi")

        if not os.path.exists(rd_dat_path):
            WARNING("pre-computed reuse distance file does not exist")
        else:
            os.remove(rd_dat_path)


    def use_precomputedRD(self):
        """
        this is an internal used function, it tries to load precomputed reuse distance to avoid the expensive
        O(NlogN) reuse distance computation, if the data does not exists, then compute it and save it
        :return:
        """

        has_rd = True
        if not self.already_load_rd and not self.reader.already_load_rd:
            if "/home/cloudphysics/traces/" not in self.reader.file_loc and \
                    "/home/jason/ALL_DATA" not in self.reader.file_loc:
                has_rd = False

            rd_dat_path = self.reader.file_loc.replace("/home/jason/ALL_DATA/",
                                                       "/research/jason/preComputedData/RD/")
            rd_dat_path = rd_dat_path.replace("/home/cloudphysics/traces/",
                                              "/research/jason/preComputedData/RD/cphyVscsi")
            if rd_dat_path == self.reader.file_loc:
                rd_dat_path += ".rd"

            if has_rd and os.path.exists(rd_dat_path):
                DEBUG("loading reuse distance from {}".format(rd_dat_path))
                self.load_reuse_dist(rd_dat_path, "rd")
            else:
                if not os.path.exists(os.path.dirname(rd_dat_path)):
                    os.makedirs(os.path.dirname(rd_dat_path))
                self.save_reuse_dist(rd_dat_path, "rd")
                INFO("reuse distance calculated and saved at {}".format(rd_dat_path))
        return self.get_reuse_distance()


    def get_hit_count(self, **kargs):
        """
        0~size(included) are for counting rd=0~size, size+1 is
        out of range, size+2 is cold miss, so total is size+3 buckets
        :param kargs:
        :return:
        """
        if 'cache_size' not in kargs:
            kargs['cache_size'] = self.cache_size
        if self.block_unit_size != 0:
            WARNING("not supported yet")
            return None
        else:
            hit_count = c_LRUProfiler.get_hit_count_seq(self.reader.c_reader, **kargs)
        return hit_count

    def get_hit_ratio(self, **kwargs):
        """

        :param kwargs:
        :return: a numpy array of CACHE_SIZE+3, 0~CACHE_SIZE corresponds to hit rate of size 0~CACHE_SIZE,
         size 0 should always be 0, CACHE_SIZE+1 is out of range, CACHE_SIZE+2 is cold miss,
         so total is CACHE_SIZE+3 buckets
        """
        kargs = {"cache_size": kwargs.get("cache_size", self.cache_size)}

        # deprecated
        if 'begin' in kwargs:
            kargs['begin'] = kwargs['begin']
        if 'end' in kwargs:
            kargs['end'] = kwargs['end']

        if self.block_unit_size != 0 :
            hit_ratio = c_LRUProfiler.get_hit_ratio_with_size(self.reader.c_reader,
                                                            block_unit_size=self.block_unit_size, **kargs)
        else:
            hit_ratio = c_LRUProfiler.get_hit_ratio_seq(self.reader.c_reader, **kargs)
        return hit_ratio


    def get_hit_ratio_shards(self, sample_ratio=0.01, **kwargs):
        """
        experimental function
        :param sample_ratio:
        :param kwargs:
        :return:
        """
        # from PyMimircache.cacheReader.tracePreprocesser import TracePreprocessor
        # kargs = {}
        # if 'cache_size' not in kwargs:
        #     kargs['cache_size'] = self.cache_size
        # else:
        #     kargs['cache_size'] = kwargs['cache_size']

        # pp = TracePreprocessor(self.reader)
        # N1, N2, traceName, fmt = pp.prepare_for_shards(sample_ratio=sample_ratio, has_time=False)
        # correction = N2 - N1
        # print("correction: {}".format(correction))
        # # correction = 0
        # tempReader = BinaryReader(traceName, init_params={"label":1, "fmt": fmt})

        correction = 0
        kargs = {"cache_size": kwargs.get("cache_size", self.cache_size)}
        if self.block_unit_size != 0:
            print("not supported yet")
            return None
        else:
            hit_ratio = c_LRUProfiler.get_hit_ratio_seq_shards(self.reader.c_reader, sample_ratio=sample_ratio,
                                                       **kargs)
        return hit_ratio

    def get_hit_ratio_phase(self, **kwargs):
        """

        :param kwargs:
        :return: a numpy array of CACHE_SIZE+3, 0~CACHE_SIZE corresponds to hit rate of size 0~CACHE_SIZE,
         size 0 should always be 0, CACHE_SIZE+1 is out of range, CACHE_SIZE+2 is cold miss,
         so total is CACHE_SIZE+3 buckets for the given phase
        """

        kargs = {
            "current_phase": kwargs.get("current_phase"),
            "num_phases": kwargs.get("num_phases")
        }

        hit_ratio = c_LRUProfiler.get_hit_ratio_phase(self.reader.c_reader, **kargs)
        return hit_ratio

    def get_hit_ratio_phase_cont(self, **kwargs):
        """

        :param kwargs:
        :return: a numpy array of CACHE_SIZE+3, 0~CACHE_SIZE corresponds to hit rate of size 0~CACHE_SIZE,
         size 0 should always be 0, CACHE_SIZE+1 is out of range, CACHE_SIZE+2 is cold miss,
         so total is CACHE_SIZE+3 buckets for the given phase
        """

        kargs = {
            "current_phase": kwargs.get("current_phase"),
            "num_phases": kwargs.get("num_phases")
        }

        hit_ratio = c_LRUProfiler.get_hit_ratio_phase_cont(self.reader.c_reader, **kargs)
        return hit_ratio


    def get_reuse_distance(self, **kargs):
        """
        get reuse distance as a numpy array
        :param kargs:
        :return:
        """
        if self.block_unit_size != 0:
            WARNING("reuse distance calculation does not support variable obj size, "
                    "calculating without considering size")
        rd = c_LRUProfiler.get_reuse_dist_seq(self.reader.c_reader, **kargs)
        return rd

    def get_future_reuse_distance(self, **kargs):
        """
        get future reuse_distance as a numpy array
        :param kargs:
        :return:
        """
        if self.block_unit_size != 0:
            WARNING("future reuse distance calculation does not support variable obj size, "
                "calculating without considering size")
        frd = c_LRUProfiler.get_future_reuse_dist(self.reader.c_reader, **kargs)
        return frd


    def plotHRC(self, figname="HRC.png", auto_resize=False, threshold=0.98, **kwargs):
        """
        plot hit ratio curve
        :param figname:
        :param auto_resize:
        :param threshold:
        :param kwargs: cache_unit_size (in Byte)
        :return:
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        # EXTENTION_LENGTH is used only when auto_resize is enabled,
        # to extend the cache size from stop point for EXTENTION_LENGTH items
        # this is used to make the curve have some part of plateau
        EXTENTION_LENGTH = 1024
        # the last two are out-of-size ratio and cold miss ratio
        hit_ratio = self.get_hit_ratio(**kwargs)[:-3]

        # if auto_resize enabled, then we calculate the first cache size
        # at which hit ratio <= final hit ratio * threshold
        # this is used to remove the long plateau at the end of hit ratio curve
        if self.cache_size == -1 and 'cache_size' not in kwargs and auto_resize:
            stop_point = len(hit_ratio)
            for i in range(len(hit_ratio)-1, 0, -1):
                if hit_ratio[i] <= hit_ratio[-1] * threshold:
                    stop_point = i
                    break
            if stop_point + EXTENTION_LENGTH < len(hit_ratio):
                stop_point += EXTENTION_LENGTH
            hit_ratio = hit_ratio[:stop_point]

        plt.xlim(0, len(hit_ratio))
        plt.plot(hit_ratio)
        xlabel = "Cache Size (Items)"

        cache_unit_size = kwargs.get("cache_unit_size", self.block_unit_size)
        if cache_unit_size != 0:
            plt.gca().xaxis.set_major_formatter(FuncFormatter(
                    lambda x, p: int(x * cache_unit_size//1024//1024)))
            xlabel = "Cache Size (MB)"

        plt.xlabel(xlabel)
        plt.ylabel("Hit Ratio")
        plt.title('Hit Ratio Curve', fontsize=18, color='black')
        if not 'no_save' in kwargs or not kwargs['no_save']:
            plt.savefig(figname, dpi=600)
            INFO("plot is saved")
        try: plt.show()
        except: pass
        if not kwargs.get("no_clear", False):
            plt.clf()
        return hit_ratio


    def plotHRC_with_shards(self, figname="HRC.png", **kwargs):
        """
        plotHRC_with_shards
        :param figname: the name of the figure
        :param kwargs: cache_unit_size (in Byte)
        :return: the average error
        """

        # getting the original HRC and SHARDS HRC
        HRC_shards = self.get_hit_ratio_shards(**kwargs)
        HRC = self.get_hit_rate(**kwargs)
        assert len(HRC_shards) == len(HRC)

        # excluding the last three elements in the array which are not hit counts
        num_HRC = len(HRC) - 3
        final_HRC = HRC[:num_HRC]
        final_HRC_shards = HRC_shards[:num_HRC]

        # calculating the error by getting the different between each point
        error = 0
        for i in range(len(final_HRC)):
            error = error + abs(final_HRC[i] - final_HRC_shards[i])

        error = error/float(len(final_HRC))
        print("avg error %2.5f" % error)

        plt.xlim(0, len(final_HRC))
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        label_shards = "LRU_shards " + str(kwargs["sample_ratio"])
        plt.plot(final_HRC_shards, label=label_shards)
        plt.plot(final_HRC[1:], label="LRU", linestyle='dashed')
        plt.xlabel("Cache Size")
        plt.ylabel("Hit Ratio")
        plt.legend(loc="best")
        plt.title('Hit Ratio Curve ' + 'average error: ' + str(error), fontsize=14, color='black')
        plt.tight_layout()
        plt.savefig(figname, dpi=600)

        return error

