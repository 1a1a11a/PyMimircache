.. mimircache documentation master file, created by sphinx-quickstart on Sun May 29 09:40:45 2016. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

mimircache: a cache trace analysis platform
===========================================

Release v\ |version|. 

Welcome to the documentation of mimircache, a python3 cache trace analysis platform.

**The power of mimircache**::
    >>> import mimircache as m 
    >>> c = m.cachecow()
    >>> c.open("trace.txt")
    >>> p = c.profiler('LRU')
    >>> p.get_reuse_dist()
    [-1 -1 -1 -1 -1 -1 11  7 11  8  8  8 -1  8]
    >>> p.plotMRC()
    >>> c.heatmap('r', 10000000, "hit_rate_start_time_end_time", data='../data/trace.vscsi', dataType='vscsi')

.. image::  images/example_MRC.png
:width: 45%
.. image::  images/example_heatmap.png
:width: 48%
    
An example of MRC plot and hit rate heatmap.


The User Guide 
-------------- 
.. toctree::
:maxdepth: 2

       User/intro
       User/installation
       User/quick_start



    User/advanced
    User/API




        See quick start for a more complete tutorial




Installation 
------------ 
Mimircache has the following dependencies: 
pkg-config, glib, scipy, numpy, matplotlib 

For mac users: 
1. install homebrew (Or macports if you prefer) 
2. brew install glib 





Supported Features
------------------ 
* Cache replacement algorithms simulation
* A variety of cache replacement algorithms support, including LRU, LFU, MRU, FIFO, Clock, Random, ARC, SLRU, optimal and etc. 
* Hit/miss rate curve(HRC/MRC) plotting 
* Reuse distance calculation for LRU 
* Heatmap plotting for visulizing hit/miss rate change over time, average reuse distance over time, etc.
* Reuse distance distribution plotting. 




Quickstart
----------
cachecow is the top level object that supports most common functionalities of mimircache. 
    >>> import mimircache as m 
    >>> c = m.cachecow(size=200)
    >>> c.open("trace.txt")
    >>> p = c.profiler('LRU')
    >>> p.get_reuse_dist()
    [-1 -1 -1 -1 -1 -1 11  7 11  8  8  8 -1  8]
    >>> p.plotMRC()


Advanced Usages
--------------- 
In mimircache, underneath cachecow, there are totally three types of objects, the first one is cache, which simulates corresponding cache replacement algorithm, 
the second one is cacheReader, which provides all the necessary functions for reading and examing trace data file, and most important of all, for extracting data for profiling. The third type of objects are the profilers. Currently, we have three kinds of profilers, the first one is LRU profiler, specially tailored for LRU; the second one is a general profiler for profiling all non-LRU cache replacement algorithms, of course, if you want, you can also use it for profiling LRU, but it runs more slowly than the LRU profiler; the third profiler is heatmap plot engine, currently supports a variety of heatmap. 

.. toctree:: 
:maxdepth: 1


    which has abstractCache as a base class, ;

Customization 
------------- 
Now you can customize mimircache for your own usage, you can 
* provide your own cache reader for reading your special cache trace files. 
* write your own profiling method for different cache replacement algorithms. 
* write a middleware for sampling your cache traces for analysis. 


API
--- 
If you want to know more mimircache and even advanced usage, please check this section. 


.. automodule:: mimircache

.. automodule::
mimircache.cache

.. automodule::
mimircache.cacheReader

.. automodule::
mimircache.profiler






Contents:





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |exampleMRC| image:: images/example_MRC.png
