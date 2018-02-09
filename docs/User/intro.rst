.. _introduction:

Introduction
============

The study of cache has a long history; however, there is no single open-source platform for easy and efficient analysis of cache traces. That's why we are building PyMimircache, a Python3 platform for analyzing cache traces. Note that PyMimircache only works on Python3, not Python2.

The target users of PyMimircache are **researchers** and **system administrators**. The goal behind PyMimircache is to provide a platform that

- allows **researchers** to study and design cache easily and efficiently.
- allows **system administrators** to analyze and visualize their cache performance easily and efficiently.

The KEY philosophy is that we would like to design a cache analysis platform that is **efficient**, **flexible** and **easy to use**.
With these in mind, we designed PyMimircache in Python3 for easy usage, and we implemented state-of-the-art algorithms in C as the backend for efficiency.
However, PyMimircache can also be used without the C backend. In other words, PyMimircache depends on CMimircache (backend), but you can use either of them independently.
Besides, PyMimircache allows the user to plug in an external reader for reading special data and allows the user to write their own cache replacement algorithm easily.


Evaluate and Design Algorithm
*****************************

PyMimircache supports **comparison of different cache replacement algorithms**, including Least Recent Used (LRU), Least Frequent Used (LFU), Most Recent Used (MRU), First In First Out (FIFO), Clock, Random, Segmented Least Recent Used (SLRU), Optimal, Adaptive Replacement Cache (ARC).
We are actively adding more cache replacement algorithms. For an extensive list of supported cache replacement algorithms, see :ref:`here<_algorithms>`.

Best of all is that you can easily and quickly **implement your own cache replacement algorithm**.

For all cache replacement algorithms, including the ones built-in and the ones you implement yourself, PyMimircache supports all kinds of comparison, **there is nothing you can't do, there is only things that you can't imagine**.

To help you better evaluate different cache replacement algorithms, we also include a variety of visualization tools inside PyMimircache.
For example, you can plot the hit ratio curve (HRC), the miss ratio curve (MRC), different variants of heatmaps and differential heatmaps. For LRU, it also supports efficient reuse distance calculation, reuse distance distribution plotting, etc.


Visualize and Analyze Workload
******************************

Another great usage of PyMimircache is **understanding workloads** so that you can be the tailor of your cache, **design better strategy** to cache your data or know why your cache has certain behavior and how your cache behaves with time.

In this part, we have figures that show you the hit ratio over time, request rate over time, object popularity distribution, reuse distance distribution, and different types of heatmaps. These show the opportunity of getting better cache performance.


Performance, Flexibility and Easy-to-Use
****************************************
Three features provided by PyMimircache are **high performance**, **flexibility**, and **easy usage**.

- **Performance**: PyMimircache uses CMimircache with state-of-the-art algorithm as backend for best performance.
- **Flexibility**: PyMimircache can also be used without CMimircache, thus using all Python-based modules. However,
both usages have the same interface, so no need to learn different tools.
Besides, PyMimircache supports three types of readers: PlainReader for reading plain text data, CsvReader for reading csv data, and BinaryReader for reading arbitrary binary data.
We also supports VscsiReader for reading vscsi data, which is a special type of binary data.
If your data is in a special format, don't worry! You can easily implement your own reader within a few lines, and you are good to go!
- **Easy Usage**: We provide Cachecow as the top-level interface, which provides most of the common usages.
Alternatively, you can easily plug in a new algorithm to see whether it can provide better performance than existing algorithms.


Work in Progress
****************

- More algorithms.
- Connection with Memcached and Redis.
- Windows support.
- GPU support.


