Welcome to mimircache 
=====================
The study of cache has a long history, however, there is no single platform for complete analysis of cache traces. That's why we are building mimircache, a Python3 platform for analyzing cache traces. It allows you to visualize your cache traces from different perspectives, and it incorporates time factor when comparing cache replacement algorithms. [See documentation here](http://mimircacheemory.readthedocs.io) 

About mimircache
----------------
Mimircache is a cache trace analysis platform that supports **comparison of different cache replacement algorithms**, including Least Recent Used(LRU), Least Frequent Used(LFU), Most Recent Used(MRU), First In First Out(FIFO), Clock, Random, Segmented Least Recent Used(SLRU), Optimal, Adaptive Replacement Cache(ARC) and a lot of others. Best of all is that you can easily and quickly **implement your own cache replacement algorithm**.

For all cache replacement algorithms, including the ones built-in and the ones you implement yourself, mimircache supports all kinds of comparison, **there is nothing you can't do, there is only things that you can't imagine**. 

To help you better evaluate different cache replacement algorithms, we also include a variety of visulization tool inside mimircache, for example you can plot hit rate curve(HRC), miss rate curve(MRC), different variants of heatmaps and differential heatmaps. For LRU, it also supports reuse distance calculation, reuse distance distribution plotting and etc. 

A sample graph generated from mimircache: 
![Heatmap](https://raw.githubusercontent.com/1a1a11a/mimircache/develop/docs/Users/images/github_heatmap.png)


Supported Features
------------------ 
* Cache replacement algorithms simulation
* A variety of cache replacement algorithms support, including LRU, LFU, MRU, FIFO, Clock, Random, ARC, SLRU, optimal and etc. 
* Hit/miss rate curve(HRC/MRC) plotting 
* Reuse distance calculation for LRU 
* Heatmap plotting for visulizing hit/miss rate change over time, average reuse distance over time, etc.
* Reuse distance distribution plotting. 
* Cache replacement algorithm comparison.


Customization 
------------- 
Now you can customize mimircache to fit your own need, you can 
* provide your own cache reader for reading your special cache trace files. 
* write your own profiling method for different cache replacement algorithms. 
* write a middleware for sampling your cache traces for analysis. 


Authors and Contributors
----------------------------
Mimircache is created by [Juncheng Yang](http://junchengyang.com) of [Ymir](http://www.ymsir.com)'s group at Emory University with help from a lot of others. 
