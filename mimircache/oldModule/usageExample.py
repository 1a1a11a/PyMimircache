from mimircache.cache.LRU import LRU
from mimircache.cacheReader.csvReader import csvReader
from mimircache.cacheReader.plainReader import plainReader
from mimircache.cacheReader.vscsiReader import vscsiReader
from mimircache.oldModule.basicLRUProfiler import basicLRUProfiler
from mimircache.oldModule.pardaProfiler import pardaProfiler
from mimircache.oldModule.pardaProfiler import parda_mode

# first step: construct a reader for reading any kind of trace

# this one is the most basic one, each line is a label/tag
reader1 = plainReader("../data/parda.trace")

# this one reads csv file and choose one column as label/tag
reader2 = csvReader("../data/trace_CloudPhysics_txt", column=4)

# this one reads binary cloudphysics trace file
reader3 = vscsiReader("../data/trace_CloudPhysics_bin")

# reader is also a generator, for readers you can do the following thing:
# read one trace element at one time:
reader1.read_one_element()
# for loop:
# for element in reader1:
#     pass
# reset, after read some elements, you want to go back
reader1.reset()

# second step: construct a profiler for analyze

# basic mattson profiler (toooooo slow)
basic_profiler = basicLRUProfiler(LRU, cache_size=20000, bin_size=10, reader=reader1)
# or you can use a short way to construct, see below
# basic_profiler = getMRCBasicLRU(LRU, 20000, 10, reader1)

# now let's run it!
# basic_profiler.run()
# # after run, you can either plot it, print it, save the list, save the plot
# basic_profiler.plotHRC()  # the plot is also saved as figure_temp in the data folder in case you forget to save it
# basic_profiler.plotMRC()  # Wow, after run, you can also obtain MRC
# basic_profiler.printHRC()
# basic_profiler.printMRC()
# basic_profiler.outputHRC("I can be a folder")
# basic_profiler.outputMRC("I can also be a file name")



# the second profiler now it supports now is parda
p = pardaProfiler(30000, reader1)  # construction is same, but you don't need to specify bin_size
p.run(parda_mode.seq)  # let's run, it supports two mode, sequential mode and openmp mode
# # p.run(parda_mode.openmp, threads=4)
p.plotHRC()  # the rest is the same as all other profilers, plot, print, output
# p.plotMRC()                         # in case the specified cache-size is not large enough, I also saved
#                                     # parda histogram list in the same folder with data
