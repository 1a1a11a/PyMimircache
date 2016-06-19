import os
import logging
import mimircache.c_LRUProfiler as c_LRUProfiler

from mimircache.cacheReader.plainReader import plainCacheReader
from mimircache.cacheReader.vscsiReader import vscsiCacheReader
import matplotlib.pyplot as plt


class LRUProfiler:
    def __init__(self, cache_size, reader):
        self.cache_size = cache_size
        self.reader = reader

        # if the given file is not basic reader, needs conversion
        if not isinstance(reader, plainCacheReader) and not isinstance(reader, vscsiCacheReader):
            self.prepare_file()
        else:
            self.num_of_lines = self.reader.get_num_total_lines()

    def prepare_file_remove_one(self):
        """
        this function will prepare the file, meanwhile remove the request that appear only once
        :return:
        """
        self.num_of_lines = 0
        logging.debug("changing file format")
        seen_dict = {}
        for e in self.reader:
            seen_dict[e] = seen_dict.get(e, 0) + 1
        self.reader.reset()
        print(len(seen_dict))
        with open('temp.dat', 'w') as ofile:
            j = self.reader.read_one_element()
            while j is not None:
                self.num_of_lines += 1
                if seen_dict[j] > 1:
                    ofile.write(str(j) + '\n')
                j = self.reader.read_one_element()
        self.reader = plainCacheReader('temp.dat')
        print(self.num_of_lines)

    def prepare_file(self):
        self.num_of_lines = 0
        logging.debug("changing file format")
        with open('temp.dat', 'w') as ofile:
            i = self.reader.read_one_element()
            while i is not None:
                self.num_of_lines += 1
                ofile.write(str(i) + '\n')
                i = self.reader.read_one_element()
        self.reader = plainCacheReader('temp.dat')

    def addOneTraceElement(self, element):
        # do not need this function in this profiler
        pass

    def get_hit_count(self, **kargs):
        hit_count = c_LRUProfiler.get_hit_count_seq(self.reader.cReader, self.cache_size, **kargs)
        return hit_count

    def get_hit_rate(self, **kargs):
        hit_rate = c_LRUProfiler.get_hit_rate_seq(self.reader.cReader, self.cache_size, **kargs)
        return hit_rate

    def get_miss_rate(self, **kargs):
        miss_rate = c_LRUProfiler.get_miss_rate_seq(self.reader.cReader, self.cache_size, **kargs)
        return miss_rate

    def get_reuse_distance(self, **kargs):
        rd = c_LRUProfiler.get_reuse_dist_seq(self.reader.cReader, **kargs)
        return rd

    def get_rd_distribution(self, **kargs):
        rd_dist = c_LRUProfiler.get_rd_distribution_seq(self.reader.cReader, **kargs)
        return rd_dist

    def plotMRC(self, autosize=False, autosize_threshold=0.01, **kwargs):
        MRC = self.get_miss_rate(**kwargs)
        try:
            # change the x-axis range according to threshhold
            num_of_blocks = 0
            if autosize:
                for i in range(len(MRC) - 3, 2, -1):
                    if (MRC[i - 1] - MRC[i]) / MRC[i] > autosize_threshold:
                        break
                num_of_blocks = i

            else:
                num_of_blocks = len(MRC) - 2

            plt.plot(range(0, num_of_blocks, 1), MRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Miss Rate")
            plt.title('Miss Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig("figure_MRC.png")
            plt.clf()
        except Exception as e:
            plt.savefig("figure_MRC.png")
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def plotHRC(self, autosize=False, autosize_threshold=0.001):
        HRC = self.get_hit_rate()
        try:
            # change the x-axis range according to threshhold
            num_of_blocks = 0
            if autosize:
                for i in range(len(HRC) - 3, 2, -1):
                    if (HRC[i] - HRC[i - 1]) / HRC[i] > autosize_threshold:
                        break
                num_of_blocks = i

            else:
                num_of_blocks = len(HRC) - 2

            plt.plot(range(0, num_of_blocks, 1), HRC[:num_of_blocks])
            plt.xlabel("cache Size")
            plt.ylabel("Hit Rate")
            plt.title('Hit Rate Curve', fontsize=18, color='black')
            plt.show()
            plt.savefig("figure_HRC.png")
            plt.clf()
        except Exception as e:
            plt.savefig("figure_HRC.png")
            print("the plotting function is not wrong, is this a headless server?")
            print(e)

    def __del__(self):
        if os.path.exists('temp.dat'):
            os.remove('temp.dat')


if __name__ == "__main__":
    # p = pardaProfiler(30000, plainCacheReader("../data/parda.trace"))
    # p = pardaProfiler(30000, csvCacheReader("../data/trace_CloudPhysics_txt", 4))
    # p = parda(LRU, 3000000, basicCacheReader("temp.dat"))
    # p.run(parda_mode.seq, threads=4)
    # p.get_reuse_distance()
    import os
    import time

    # from pympler.tracker import SummaryTracker

    # tracker = SummaryTracker()

    t1 = time.time()
    for i in range(6):
        p = LRUProfiler(30000, vscsiCacheReader("../data//trace_CloudPhysics_bin"))
        rd_a = p.get_reuse_dist()
        print(rd_a)
    t2 = time.time()

    print(t2 - t1)

    p = LRUProfiler(30000, vscsiCacheReader("../data//trace_CloudPhysics_bin"))
    p.plotHRC()

    # print(len())
    # count = 0
    # for i in rd_a:
    #     if i==0:
    #         count+=1
    # print(count)

    # tracker.print_diff()


    # p = pardaProfiler(2000, csvCacheReader('../data/mining/mining.dat.original', 1))
    # p = pardaProfiler(2000, plainCacheReader('../data/mining/mining.dat.original'))
    # p._test()


    # for f in os.listdir('../data/mining/'):
    #     shutil.copy('../data/mining/' + f, '../data/mining/mining.dat')
    #     print(f)
    #     p._test()
    # p.run_with_specified_lines(10000, 20000)
    # p.plotHRC(autosize=True, autosize_threshhold=0.00001)
