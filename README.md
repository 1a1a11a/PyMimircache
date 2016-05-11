# cachecow

This is the cachecow platform used for analyzing cache trace developed by Emory University, Ymir group 

Currently the platform is still under development. 

The current platform supports plotting hit rate curve(**HRC**) and miss rate curve(**MRC**) of a given trace, 
and also plotting heatmap of a workload under LRU between different time is supported. 

### Dependency
please have glib, numpy, scipy, matplotlib installed


### Mimircache Tutorial 
```python3
import mimircache as m

# initialize a cachecow object 
size is optional, if don’t give it here, you need to use set_size() before running your workload 
    c = m.cachecow(size=10000)	
    c.set_size(200)		        # you can set/change size at any time 
# you can open three kinds of trace files (choose the one you need)  
c.csv(“tracefile.csv”, 2)	# 2 is the column of the cache requests (column begins from 0) 
c.open(“trace.txt”)		    # this opens a normal text file, in which each line is a request
c.vscsi(“dat.vscsi”) 		# special for vscsi trace 
# print cache requests: 
for req in c: 
      print(req) 
# We can also do a reset to go back to beginning 
c.reset() 
# if you need need more operations about the data, please read advanced usage 

# now, let’s run the workload to get miss rate curve(MRC), 
# First get the profiler object 
p = c.profiler(‘LRU’)			# if you want to profile other cache replacement algorithms, 
also specify bin_size, which represents the sample interval
# currently all supported cache replacement algorithms include: LRU, MRU, Random(RR), SLRU, S4LRU, clock, ARC, FIFO, LFU with Random(LFU_RR), LFU with MRU(LFU_MRU), LFU with LRU(LFU_LRU) 
# then run it 
p.run()
# after running, you can get MRC list(all operations to MRC also applies for HRC), or you can directly plot it  
print(p.MRC)		# you can also do print(p.HRC)
p.plotMRC()		# you can also do p.plotHRC()
# if you only want reuse distance, then you can do: 
rdist = p.get_reuse_distance()		# rdist is a list with all the reuse distances 

# besides profiling for MRC/HRC, you can also plot heatmap (only support LRU and vscsi data now): 
c.heatmap(mode, interval)	# mode can be ‘r’ for real time or ‘v’ for virtual time, interval for how long the time interval should be, after calling this function. Besides, you can also specify num_of_process, recommends set this value to the same of cores in the running server, figname for the location and name for the created plot. After calling this function, a new plot will be created at current directory 
