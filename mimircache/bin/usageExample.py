import mimircache as m

# initialize a cachecow object
# size is optional, if don’t give it here, you need to use set_size() before running your workload
c = m.cachecow(size=40000)
c.set_size(40000)  # you can set/change size at any time

# you can open three kinds of trace files (choose the one you need)
c.open("../data/trace.txt")  # this opens a normal text file, in which each line is a request
c.vscsi('../data/trace.vscsi')
# print cache requests:
# for req in c:
#       print(req)
# We can also do a reset to go back to beginning
c.reset()
# if you need need more operations about the data, please read advanced usage

# now, let’s run the workload to get miss rate curve(MRC),
# First get the profiler object
p = c.profiler('LRU')  # if you want to profile other cache replacement algorithms,
# also specify bin_size, which represents the sample interval
# currently all supported cache replacement algorithms include: LRU, MRU, Random(RR), SLRU, S4LRU, clock, ARC, FIFO, LFU with Random(LFU_RR), LFU with MRU(LFU_MRU), LFU with LRU(LFU_LRU)
# then run it
print(p.get_reuse_distance())
print(p.get_hit_count())
print(p.get_hit_rate())
print(p.get_miss_rate())
print(p.get_rd_distribution())

c.reset()
p.plotHRC()
p.plotMRC()
# after running, you can get MRC list(all operations to MRC also applies for HRC), or you can directly plot it
# you can also do print(p.HRC)
# print(p.get_miss_rate())
# p.plotMRC()  # you can also do p.plotHRC()
# if you only want reuse distance, then you can do:
# rdist is a list with all the reuse distances
