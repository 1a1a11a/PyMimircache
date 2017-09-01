.. _introduction:

Introduction
============ 

The study of cache has a long history, however, there is no single platform for complete analysis of cache traces. That's why we are building mimircache, a python platform for analyzing cache traces. 

Mimircache is a cache trace analysis platform that supports **comparison of different cache replacement algorithms**, including Least Recent Used(LRU), Least Frequent Used(LFU), Most Recent Used(MRU), First In First Out(FIFO), Clock, Random, Segmented Least Recent Used(SLRU), optimal, Adaptive Replacement Cache(ARC) and we are currently adding more cache replacement algorithms. Best of all is that you can easily and quickly **implement your own cache replacement algorithm**.

For all cache replacement algorithms, including the ones built-in and the ones you implement yourself, mimircache supports all kinds of comparison, **there is nothing you can't do, there is only things that you can't imagine**. 

To help you better evaluate different cache replacement algorithms, we also include a variety of visulization tool inside mimircache, for example you can plot hit rate curve(HRC), miss rate curve(MRC), different variants of heatmaps and differential heatmaps. For LRU, it also supports reuse distance calculation, reuse distance distribution plotting and etc. 

Another great usage of mimircache is to use mimircache to **analyze and visulize your workloads**, so that you can be the tailor of your data, **design better cache replacement algorithms** to fit your data. Currently we have provided three types of readers, plainCacheReader for reading plain text file, csvCacheReader for reading csv cache file, vscsiCacheReader for reading vscsi trace files. If your data is in special format, don't worry, you can easily implement your own reader, within a few lines, you are good to go! 

Besides plugging in new reader for reading your specialized trace files, your own cache replacement algorithms for comparing with existing ones, you can even add a sampling layer for fast analysis. 

Our progress on mimircache
+++++++++++++++++++++++++++
Currently, most time-consuming components have being re-implemented in C, but python part is also kept for reader and cache replacement extensions. However, if you want your reader and cache replacement algorithm to be even faster, you have to implement them in C, check the source code and you will have clue.

