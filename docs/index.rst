mimircache: a cache trace analysis platform
===========================================

Release v\ |version|. 

Welcome to the documentation of mimircache, a Python3 cache trace analysis platform.

**The power of mimircache**:
    >>> import mimircache as m
    >>> c = m.cachecow()
    >>> c.vscsi("trace.vscsi")      # find this data under data folder, other type of data supported too
    >>> print(c.stat())
    >>> print(c.get_reuse_distance())
    [-1 -1 -1 -1 -1 -1 11 7 11 8 8 8 -1 8]

    >>> print(c.get_hit_ratio_dict("LRU", cache_size=20))
    {0: 0.0, 1: 0.025256428270338627, 2: 0.031684698608964453, ... 20: 0.07794716875087819}

    >>> c.plotHRCs(["LRU", "LFU", "Optimal"])

    >>> c.heatmap('r', "hit_ratio_start_time_end_time", time_interval=10000000)

.. image::  images/github_HRC.png
    :width: 45%
.. image::  images/github_heatmap.png
    :width: 48%
    
An example of hit ratio curve plot and hit ratio heatmap.


The User Guide 
-------------- 
.. toctree::
        :maxdepth: 3

        User/intro
        User/installation
        User/quick_start
        User/Tutorial/open_trace
        User/Tutorial/profiling
        User/Tutorial/basic_plotting
        User/Tutorial/heatmap_plotting
        User/Appendix/algorithms
        User/AdvancedUsage/advanced_usage
        User/API





Supported Features
------------------ 
* Cache replacement algorithms simulation and trace visualization.
* A variety of cache replacement algorithms support, including LRU, LFU, MRU, FIFO, Clock, Random, ARC, SLRU, optimal and etc. 
* Hit/miss ratio curve(HRC/MRC) plotting.
* Efficient reuse distance calculation for LRU.
* Heatmap plotting for visulizing hit/miss ratio change over time, average reuse distance over time, etc.
* Reuse distance distribution plotting. 
* Cache replacement algorithm comparison.


Customization 
------------- 
Now you can customize mimircache to fit your own need, you can

* provide your own cache reader for reading your special cache trace files. 
* write your own profiling method for different cache replacement algorithms. 
* write a middleware for sampling your cache traces for analysis. 











Indices and tables
==================

* :ref:`search`
