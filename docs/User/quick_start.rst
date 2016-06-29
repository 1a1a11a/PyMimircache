.. _quick_start:

Quick Start
===========

Get Prepared
------------
With mimircache, testing/profiling cache replacement algorithms is very easy.
Let's begin by getting a cachecow object from mimircache:

    >>> import mimircache as m
    >>> c = m.cachecow()

Open Trace File
---------------
Now let's open a trace file, your have three choices for opening different types of trace files, choose the one suits your need.

    >>> c.open("trace/file/location")
    >>> c.csv("trace/file/location", column=x)  # specify which column contains the request key
    >>> c.vscsi("trace/file/location")          # for vscsi format data

OK, data is ready, now let's play!

.. note::
However, if you have a special data format, you can write your own reader in a few lines, see here about how to write your own cache reader :ref:`create_new_cacheReader`.



Get Profiler/Do Profiling
----------------------------
Before we calculate/plot something, let's make one thing clear, the calculation and plotting is carried out by something called profiler, so we need to obtain a profiler first.

First, let's try LRU(least recently used), we need to get a profiler:

    >>> profiler_LRU = c.profiler('LRU')


* To get reuse distance for each request, simply call the functions below, the returned result is a numpy array:
    >>> reuse_dist = profiler_LRU.get_reuse_distance()
    array([-1, -1, -1, ..., -1, -1, -1], dtype=int64)

* To get hit count, hit rate, miss rate, we can do the following:

* Hit count is a numpy array, the nth element of the array means among all requests, how many requests will be able to fit in the cache if we increase cache size from n-1 to n. The last two elements of the array are different from all others, the second to the last element means the number of requests that needs larger cache size(if you specified cache_size parameter, otherwise it is 0), the last element says the number of cold misses, meaning the number of unique requests.
    >>> profiler_LRU.get_hit_count()
    array([0,  2685,  662, ...,   0,   0,  48974], dtype=int64)

* Hit rate is a numpy array, the nth element of the array means the hit rate we can achieve given cache size of n. Similar to hit count, the last two elements says the ratio of requests that needs larger cache size and the ratio of requests that are unique.
    >>> profiler_LRU.get_hit_rate()
    array([ 0,  0.02357911,  0.02939265, ...,  0.56992061,   0,  0.43007939])

* Miss rate is a numpy array, the nth element of the array means the miss rate we will have given cache size of n. The last two elements of the array are the same as that of hit rate array.
    >>> profiler_LRU.get_miss_rate()
    array([ 1,  0.97642089,  0.97060735, ...,  0.43007939,   0,  0.43007939])

.. note::
for reuse distance, hit count, hit rate, miss rate, if you don't specify a cache_size parameter or specify cache_size=-1, it will use the largest possible size.

* There are also some other keyword arguments that can be used in the above functions, for example:
    >>> profiler_LRU.get_hit_rate(cache_size=2000, begin=1, end=10)

See below for all the available keyword arguments:




* With the data calculated from profiler, you can do plotting yourself, or any other calculation. But for your convenience, we have also provided several plotting functions for you to use.
    >>> profiler_LRU.plotHRC()


.. image::  ../images/example_HRC.png
:width: 45%
    :align: center

            MRC:

                >>> profiler_LRU.plotMRC()

.. image::  ../images/example_MRC.png
:width: 45%
    :align: center





    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |     arguments      |             cache_size               |                       begin                      |               end              |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |                    | the size of cache, default is -1,    | the place begin profiling,  begin with 0,        | the place stops profiling,     |
    |     functions      | which is the largest possible size   | default is 0, the beginning of file              | default is -1, the end of file |
    +====================+======================================+==================================================+================================+
    | get_reuse_distance |                 No                   |                        No                        |               No               |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |    get_hit_count   |                 Yes                  |                        Yes                       |               Yes              |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |    get_hit_rate    |                 Yes                  |                        Yes                       |               Yes              |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |    get_miss_rate   |                 Yes                  |                        Yes                       |               Yes              |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |       plotHRC      |                 Yes                  |                        No                        |               No               |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+
    |       plotMRC      |                 Yes                  |                        No                        |               No               |
    +--------------------+--------------------------------------+--------------------------------------------------+--------------------------------+

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

