PyMimircache
==========

[![GitHub version](https://badge.fury.io/gh/1a1a11a%2FPyMimircache.svg)](https://badge.fury.io/gh/1a1a11a%2FPyMimircache)
[![PyPI version](https://badge.fury.io/py/PyMimircache.svg)](https://badge.fury.io/py/PyMimircache)

NEWS
----

* [libCacheSim](https://www.github.com/1a1a11a/libcachesim) is a new library for cache simulations, which provides a 10x-100x performance boost compared to PyMimircache. 

* PyMimircache to appear at FAST tutorial. 



PyMimircache is a cache trace analysis platform that supports

-   **comparison of different cache replacement algorithms**

-   **visualization of cache traces**

-   **easy plugging in your own cache replacement algorithm**

Main users of PyMimircache include **researchers** and **system administrators**. PyMimircache provides researchers a tool to 
study existing algorithms, and devise and test new algorithms. While PyMimircache provides system administrators a simple tool helping 
them visualize and understand their cache. 
  
PyMimircache is an independent Python3 platform that supports all the described features. 
Besides, it also bundles with CMimircache for better performance. If you need a C/C++ platform, please check out CMimircache. 
 
PyMimircache current supports algorithms include Least Recent Used(LRU), Least Frequent Used(LFU), 
Most Recent Used(MRU), First In First Out(FIFO), Segmented LRU(SLRU), Clock, Random, Optimal, Adaptive Replacement Cache(ARC).
And we are actively adding more cache replacement algorithms.

Best of all is that you can easily and quickly **implement your own cache
replacement algorithm**. [See more information here](http://mimircache.info)


Dependency and Installation
---------------------------

#### System-wide library: glib, python3-pip, python3-matplotlib

On Ubuntu use the following command to install

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ sudo apt-get install libglib2.0-dev python3-pip python3-matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### Python Dependency: numpy, scipy, matplotlib, heapdict, mmh3

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ sudo pip3 install heapdict mmh3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### Installing PyMimircache

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ sudo pip3 install PyMimircache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#### Compatibility

*PyMimircache only support Python3 and 64bit platform*
 
#### git clone

If you use the Github repo, after the git clone, do `git submodules update --init` to clone the CMimircache module.


Alternative using docker
------------------------

As an alternative, you can use PyMimircache in a docker container. According to our simple benchmark, the performance difference between using a bare metal and a docker container is less than 10%.

### Use interactive shell

To enter an interactive shell and do plotting, you can use

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ sudo docker run -it --rm -v $(pwd):/PyMimircache/scripts -v PATH/TO/DATA:/PyMimircache/data 1a1a11a/PyMimircache /bin/bash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After you run this command, you will be in a shell with everything ready. Your
current directory is mapped to `/PyMimircache/scripts/`, and your data directory is
mapped to `/PyMimircache/data`. In addition, we have prepared a test dataset for
you at `/PyMimircache/testData`.
 

### Run scripts directly

If you don't want to use an interactive shell and you have your script ready,
then you can do

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ docker run --rm -v $(pwd):/PyMimircache/scripts -v PATH/TO/DATA:/PyMimircache/data 1a1a11a/PyMimircache python3 /PyMimircache/scripts/YOUR_PYTHON_SCRIPT.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, if you are new here or you have trouble using docker to run scripts
directly, we suggest using an interactive shell which can help you debug.


PyMimircache Tutorial
-------------------

We have prepared a wonderful tutorial here. [Check here for tutorial](http://pymimircache.readthedocs.io)

### PyMimircache Power

**The power of PyMimircache**

```python
>>> from PyMimircache import Cachecow
>>> c = Cachecow()
>>> c.vscsi("trace.vscsi")      # find this data under the data folder, other types of data are supported as well
>>> print(c.stat())
	# number of requests: 113872
	# number of uniq obj/blocks: 48974
	# cold miss ratio: 0.4301
	# top N popular (obj, num of requests):
	# [(3345071, 1630),
	#  (6160447, 1342),
	#  (6160455, 1341),
	#  (1313767, 652),
	#  (6160431, 360),
	#  (6160439, 360),
	#  (1313768, 326),
	#  (1329911, 326)]
	# number of obj/block accessed only once: 21049
	# frequency mean: 2.33
	# time span: 7200089885

>>> print(c.get_reuse_distance())
    # [-1 -1 -1 -1 -1 -1 11 7 11 8 8 8 -1 8]

>>> print(c.get_hit_ratio_dict("LRU", cache_size=20))
    # {0: 0.0, 1: 0.025256428270338627, 2: 0.031684698608964453, ... 20: 0.07794716875087819}

>>> c.plotHRCs(["LRU", "LFU", "Optimal"])

>>> c.heatmap('r', "hit_ratio_start_time_end_time", time_interval=10000000)

```

| [![HRC](https://github.com/1a1a11a/PyMimircache/blob/develop/docs/User/images/github_HRC.png)](https://github.com/1a1a11a/PyMimircache/blob/develop/docs/User/images/github_HRC.png)  | [![Heatmap](https://github.com/1a1a11a/PyMimircache/blob/develop/docs/User/images/github_heatmap.png)](https://github.com/1a1a11a/PyMimircache/blob/develop/docs/User/images/github_heatmap.png) |
|:---:|:---:|
| Hit Ratio Curve | Hit Ratio Heatmap |


Contributing
------------
PyMimircache and CMimircache are created by Juncheng Yang of the SimBioSys group at Emory University. CMimircache, previously Mimircache, was released as part of
`
MITHRIL: Mining Sporadic Associations for Cache Prefetching. Juncheng Yang , Reza Karimi, Trausti Saemundsson, Avani Wildani, Ymir Vigfusson. ACM Symposium on Cloud Computing (SoCC), 2017. `

Reference 
---------
```
@inproceedings{mithril,
	author = {Yang, Juncheng and Karimi, Reza and S\ae{}mundsson, Trausti and Wildani, Avani and Vigfusson, Ymir},
	title = {Mithril: Mining Sporadic Associations for Cache Prefetching},
	year = {2017},
	isbn = {9781450350280},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3127479.3131210},
	doi = {10.1145/3127479.3131210},
	booktitle = {Proceedings of the 2017 Symposium on Cloud Computing},
	pages = {66–79},
	numpages = {14},
	location = {Santa Clara, California},
	series = {SoCC '17}
}
```


This project has benefited from contributions from numerous people. You are more than welcome to make any contributions. Please create Pull Request for any changes.

LICENSE
-------
PyMimircache is provided under GPLv3 license.

Related
-------
[libCacheSim](https://github.com/1a1a11a/libCacheSim): a high-performance C++ library for cache simulations
