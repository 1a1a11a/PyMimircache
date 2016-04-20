# cachecow

This is the cachecow platform used for analyzing cache trace developed by Emory University, Ymir group 

Currently the platform is still under development. 

The current platform supports plotting hit rate curve(**HRC**) and miss rate curve(**MRC**) of a given trace. 

### Dependency
please have glib, numpy, scipy, matplotlib installed

##The following Usage is out-dated, waiting for update. 

### Usage:
```python
from cachecow.Cache.LRU import LRU
from cachecow.CacheReader.basicCacheReader import basicCacheReader
from cachecow.CacheReader.csvCacheReader import csvCacheReader
from cachecow.CacheReader.vscsiReader import vscsiReader
from cachecow.Profiler.getMRCBasicLRU import getMRCBasicLRU
from cachecow.Profiler.parda import parda

# first step: construct a reader for reading any kind of trace

# this one is the most basic one, each line is a label/tag/record
reader1 = basicCacheReader("../Data/parda.trace")

# this one reads csv file and choose one column as label/tag
reader2 = csvCacheReader("../Data/trace_CloudPhysics_txt", column=4)

# this one reads binary cloudphysics trace file
reader3 = vscsiReader("../Data/trace_CloudPhysics_bin")

# reader is also a generator, for readers you can do the following thing:
# read one trace element at one time:
reader1.read_one_element()
# for loop:
for element in reader1:
     do something
# reset, after read some elements, you want to go back
reader1.reset()

# second step: construct a profiler for analyze

# basic mattson profiler (toooooo slow)
basic_profiler = getMRCBasicLRU(LRU, cache_size=20000, bin_size=10, reader=reader1)
# or you can use a short way to construct, see below
basic_profiler = getMRCBasicLRU(LRU, 20000, 10, reader1)

# now let's run it!
basic_profiler.run()
# after run, you can either plot it, print it, save the list, save the plot
basic_profiler.plotHRC()  # the plot is also saved as figure_temp in the data folder in case you forget to save it
basic_profiler.plotMRC()  # Wow, after run, you can also obtain MRC
basic_profiler.printHRC()
basic_profiler.printMRC()
basic_profiler.outputHRC("I can be a folder")
basic_profiler.outputMRC("I can also be a file name")



# the second profiler now it supports now is parda
p = parda(LRU, 30000, reader1)      # construction is same, but you don't need to specify bin_size
p.run(parda_mode.seq)               # let's run, it supports two mode, sequential mode and openmp mode
# p.run(parda_mode.openmp, threads=4) # parallel analysis, not stable
p.plotHRC()                         # the rest is the same as all other profilers, plot, print, output
p.plotMRC()                         # in case the specified cache-size is not large enough, I also saved
                                     # parda histogram list in the same folder with data
