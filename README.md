PyMimircache
==========

[![Build Status](https://travis-ci.org/1a1a11a/PyMimircache.svg?branch=master)](https://travis-ci.org/1a1a11a/PyMimircache)
[![Documentation Status](https://readthedocs.org/projects/PyMimircache/badge/?version=develop)](http://PyMimircache.readthedocs.io/en/develop/?badge=develop)
[![GitHub version](https://badge.fury.io/gh/1a1a11a%2FPyMimircache.svg)](https://badge.fury.io/gh/1a1a11a%2FPyMimircache)
[![PyPI version](https://badge.fury.io/py/PyMimircache.svg)](https://badge.fury.io/py/PyMimircache)

PyMimircache is a cache trace analysis platform that supports

-   **comparison of different cache replacement algorithms**

-   **visualization of cache traces**

-   **easy plugging in your own cache replacement algorithm**


Current support algorithms include Least Recent Used(LRU), Least Frequent
Used(LFU), Most Recent Used(MRU), First In First Out(FIFO), Segmented LRU(SLRU),
Clock, Random, Optimal, Adaptive Replacement Cache(ARC).

And we are actively adding more cache replacement algorithms.

Best of all is that you can easily and quickly **implement your own cache
replacement algorithm**. [See more information here](http://mimircache.info)


Dependency and Installation
---------------------------

#### System-wide library: glib, python3-pip, python3-matplotlib

On Ubuntu using the following command to install

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ sudo apt-get install libglib2.0-dev python3-pip, python3-matplotlib
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

git clone needs to add --recurse-submodules option to automatically clone the CMimircache module.

Alternative using docker
------------------------

As an alternative, you can use PyMimircache in a docker container, according to our simple benchmark, the performance difference between using a bare metal and a docker container is less than 10%.

### Use interactive shell

To enter an interactive shell and do plotting, you can use

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ sudo docker run -it --rm -v $(pwd):/PyMimircache/scripts -v PATH/TO/DATA:/PyMimircache/data 1a1a11a/PyMimircache /bin/bash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After you run this command, you will be in a shell with everything ready, your
current directory is mapped to `/PyMimircache/scripts/` and your data directory is
mapped to `/PyMimircache/data`. In addition, we have prepared a test dataset for
you at `/PyMimircache/testData`.
 

### Run scripts directly

If you don't want to use an interactive shell and you have your script ready,
then you can do

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jason@myMachine: ~$ docker run --rm -v $(pwd):/PyMimircache/scripts -v PATH/TO/DATA:/PyMimircache/data 1a1a11a/PyMimircache python3 /PyMimircache/scripts/YOUR_PYTHON_SCRIPT.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

However, if you are new here or you have trouble using docker to run scripts
directly, we suggest using interactive shell which can help you debug.


PyMimircache Tutorial
-------------------

We have prepared a wonderful tutorial here. [Check here for tutorial](http://PyMimircacheemory.readthedocs.io)

### PyMimircache Power

**The power of PyMimircache**

```python
>>> import PyMimircache as m
>>> c = m.Cachecow()
>>> c.vscsi("trace.vscsi")      # find this data under data folder, other type of data supported too
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


Next Major Releases
-------------------
* **Architecture**: re-write py-PyMimircache generalProfiler (v0.3.0)
* **Architecture**: Separate py-PyMimircache and C-PyMimircache to allow them run independently (v0.4.0)
* **Feature**: Add block/obj size into cache replacement algorithm (v0.5.0)
* **Feature**: Add parallel support for LRUProfiling (v0.6.0)
* **Feature**: Add sampling layer for all profilers (v0.7.0)
* **Feature**: Add GPU profiling support (v0.8.0)
* **Feature**: Add Windows Support (v1.0.0)
* **Architecture**: C component optimization and allow to be used as an independent caching layer (v2.0.0)


Contributing
------------
You are more than welcome to make any contributions. Please create Pull Request for any changes.

LICENSE
-------
PyMimircache is provided under GPLv3 license.