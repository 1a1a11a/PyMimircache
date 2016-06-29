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
           User/algorithms
           User/advanced_usages




        See quick start for a more complete tutorial





Supported Features
------------------ 
* Cache replacement algorithms simulation
* A variety of cache replacement algorithms support, including LRU, LFU, MRU, FIFO, Clock, Random, ARC, SLRU, optimal and etc. 
* Hit/miss rate curve(HRC/MRC) plotting 
* Reuse distance calculation for LRU 
* Heatmap plotting for visulizing hit/miss rate change over time, average reuse distance over time, etc.
* Reuse distance distribution plotting. 


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
