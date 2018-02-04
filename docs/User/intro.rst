.. _introduction:

Introduction
============

The study of cache has a long history, however, there is no single open-source platform for easy and efficient analysis of cache traces. That's why we are building PyMimircache, a Python3 platform for analyzing cache traces. Note that, PyPyMimircache only works on Python3, not Python2.

The target users of PyMimircache are **researchers** and **system administrators**. The goal behind PyMimircache is to provide a platform that

- allow **researchers** to study and design cache replacement algorithms easily and efficiently.
- allow **system administrators** to analyze and visualize their cache performance easily and efficiently.

The KEY philosophy is that we would like to design a cache analysis platform that is **efficient**, **flexible** and **easy to use**. With two in mind, we designed PyMimircache in Python3 for easy use and we implemented state-of-the-art algorithms in C as backend for efficiency. However, PyMimircache can also be used without C backend, in other words, PyMimircache depends on CMimircache (backend), but you can either of them independently.


Evaluate and Design Algorithm
*****************************

PyMimircache supports **comparison of different cache replacement algorithms**, including Least Recent Used (LRU), Least Frequent Used (LFU), Most Recent Used (MRU), First In First Out (FIFO), Clock, Random, Segmented Least Recent Used (SLRU), Optimal, Adaptive Replacement Cache (ARC) and we are actively adding more cache replacement algorithms. For an extensive list of supported cache replacement algorithms, see :ref:`here<_algorithm>`.

Best of all is that you can easily and quickly **implement your own cache replacement algorithm**.

For all cache replacement algorithms, including the ones built-in and the ones you implement yourself, PyMimircache supports all kinds of comparison, **there is nothing you can't do, there is only things that you can't imagine**.

To help you better evaluate different cache replacement algorithms, we also include a variety of visulization tool inside PyMimircache, for example you can plot hit ratio curve (HRC), miss ratio curve (MRC), different variants of heatmaps and differential heatmaps. For LRU, it also supports efficient reuse distance calculation, reuse distance distribution plotting and etc.


Visualize and Analyze Workload
******************************

Another great usage of PyMimircache is **understanding workloads**. so that you can be the tailor of your cache, **design better strategy** to cache your data or know why your cache has certain behavior and how your cache behaves with time.

In this part, we have figures that show you the hit ratio over time, request rate over time, object popularity distribution, reuse distance distribution, and different types of heat maps showing the opportunity of getting better cache performance.


Performance, Flexibility and Easy-to-Use
****************************************
Three feature provided by PyMimircache are **high performance**, **flexible on input**, and **easy to use**.

- **Performance**: PyMimircache uses CMimircache with state-of-the-art algorithm as backend for best performance.
- **Flexibility**: PyMimircache can also be used without CMimircache, thus using all Python based modules. However, both usages have the same interface, no need to learn different tools. Besides, PyMimircache supports three types of readers, PlainReader for reading plain text data, CsvReader for reading csv data, BinaryReader for reading arbitrary binary data. We also supports VscsiReader for reading vscsi data, which is a special type of binary data. If your data is in special format, don't worry, you can easily implement your own reader, within a few lines, you are good to go!
- **Easy-to-use**: We provide Cachecow as the top level interface, with provides most of the common usages. Besides, you can easily plug in new algorithms to see whether it can provide better performance than existing algorithms.


Work in Progress
****************

- More algorithms.
- Connection with Memcached and Redis.
- Windows support.
- GPU support.


