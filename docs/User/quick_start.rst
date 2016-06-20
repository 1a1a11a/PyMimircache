Quick Start
=========== 

Get Prepared
------------ 
Testing/profiling cache replacement algorithms is very easy. 
Begin by getting a cachecow object from mimircache: 
    
    >>> import mimircache as m 
    >>> c = m.cachecow(size=200)

The size is an optional choice, but if you don't provide size here, you need to provide it later. 

Open Trace File
---------------
Now let's open a trace file, your have three choices for opening different types of trace files, choose the one suits your need. 

If you have a special data format, you can write your own reader in a few lines, see here about how to write your own cache reader !!!!!!!!!!!!!!!!! Link 

    >>> c.open("trace/file/location")
    >>> c.csv("trace/file/location", column=x)  # specify which column contains the request key 
    >>> c.vscsi("trace/file/location")          # for vscsi format data 

OK, data is ready, now let's play! 

Get Profiler/Doing Profiling 
----------------------------
Before we calculate/plot something, let's make one thing clear, the calculation and plotting is carried out by something called profiler, so for each cache replacement algorithm, we need to generate a profiler first. 

First, let's try LRU(least recently used), we need to get a profiler: 

    >>> profiler_LRU = c.profiler('LRU')

To get reuse distance, simply call the functions below, the returned result is a numpy array: 

    >>> reuse_dist = profiler_LRU.get_reuse_distance() 
    [-1 -1 -1 ..., -1 -1 -1]

To get hit count, hit rate, miss rate, we can do the following, the returned result is also a numpy array: 

    >>> profiler_LRU.get_hit_count()
    [ 2685   662   561 ...,     0    20 48974]

    >>> profiler_LRU.get_hit_rate()
    [  2.35791057e-02   2.93926504e-02   3.43192331e-02 ...,   5.69783568e-01   1.75635796e-04   4.30079401e-01]

    >>> profiler_LRU.get_miss_rate()
    [  9.76420879e-01   9.70607340e-01   9.65680778e-01 ...,   4.30216432e-01   1.75635796e-04   4.30079401e-01]

.. note:: 
for reuse distance, hit count, hit rate, miss rate, if you don't specify a size parameter or specify size=-1, it will use the largest possible size.

.. note::
the numpy array returned from hit count, hit rate, miss rate has a length of cache_size+3, the first cache_size+1 elements correspond to cache_size 0~cachesize, and the second to the last element corresponds to hit count/hit rate larger than the specified cache_size, while the last element corresponds to the cold miss count/cold miss rate (the requests that appear only once).

With the data calculated from profiler, you can do plotting yourself, or any other calculation. But for your convenience, we have also provided several plotting functions for you to use. 
    
    >>> profiler_LRU.plotHRC()

.. image::  ../images/example_HRC.png
:width: 45%
    :align: center

            MRC:

                >>> profiler_LRU.plotMRC()

.. image::  ../images/example_MRC.png
:width: 45%
    :align: center


            As a special characteristic of LRU, reuse distance has a lot of important usages, so here we also provided a function to help you to calculate the reuse distance distribution, basically it is a numpy array of length=cache_size+1 (0~cache_size), each element of the array corresponds to a bucket of the same size and it is used to calculate the number of reuse distance at this cache_size.

                >>> profiler_LRU.get_rd_distribution()
                [ 2685   662   561 ...,     0     0 48974]

            Apart from LRU, we have also provided a varieties of other cache replacement algorithms for you to play with, including MRU, LFU_RR, LFU_MRU, LFU_LRU, RR(Random), optimal, SLRU, S4LRU, FIFO, clock, ARC, to play with these cache replacement algorithms, you just substitue 'LRU' in the examples above with cache replacement algorithm you want. If you want to test your own cache replacement algorithms, check here_. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Link

.. warning:: 
reuse distance related operation is only allowed on LRU, so don't call get_reuse_distance and get_rd_distribution on non-LRU cache replacement algorithms.


Plotting Heatmaps 
----------------- 
The other great feature about mimircache is that it allows you to incorporate time component of a cache trace file into consideration, make the cache analysis from static to dynamic. 
Currently five types of heatmaps are supported: 

+-----------------------+-----------------------------+-----------------------------------------+
|       plot name       | type name(used in function) |                Details                  |
+=======================+=============================+=========================================+
| hit rate heatmap with | hit_rate_start_time         | a heatmap with hit rate as heat value,  |
| different start time  | _end_time                   | the x-axis is starting time, while      |
| and end time          |                             | y-axis is the ending time.              |
+-----------------------+-----------------------------+-----------------------------------------+
| body row 1            | column 2                    | column 3                                |
+-----------------------+-----------------------------+-----------------------------------------+
| body row 1            | column 2                    | column 3                                |
+-----------------------+-----------------------------+-----------------------------------------+
| body row 1            | column 2                    | column 3                                |
+-----------------------+-----------------------------+-----------------------------------------+
| body row 1            | column 2                    | column 3                                |
+-----------------------+-----------------------------+-----------------------------------------+
| body row 1            | column 2                    | column 3                                |
+-----------------------+-----------------------------+-----------------------------------------+




| body row 3 | Cells may  | - Cells   |
+------------+ span rows. | - contain |
| body row 4 |            | - blocks. |
+------------+------------+-----------+

        # 1. hit_rate_start_time_end_time
        # 2. hit_rate_start_time_cache_size
        # 3. avg_rd_start_time_end_time
        # 4. cold_miss_count_start_time_end_time
        # 5. rd_distribution



.. note:: 
Currently, heatmap real time plotting is only supported on vscsi format data, supporting for other kinds of data will be included later.


