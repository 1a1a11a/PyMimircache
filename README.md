mimircache
==========

<https://travis-ci.org/1a1a11a/mimircache>

Mimircache is a cache trace analysis platform that supports **comparison of
different cache replacement algorithms**, including Least Recent Used(LRU),
Least Frequent Used(LFU), Most Recent Used(MRU), First In First Out(FIFO),
Clock, Random, Segmented Least Recent Used(SLRU), optimal, Adaptive Replacement
Cache(ARC) and we are currently adding more cache replacement algorithms. Best
of all is that you can easily and quickly **implement your own cache replacement
algorithm**. [See more information here](http://mimircache.info)

 

Dependency and Installation 
----------------------------

### System-wide library: glib, python3-pip, python3-matplotlib 

On Ubuntu using the following command to install

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
apt-get install libglib2.0-dev python3-pip, python3-matplotlib 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Python Dependency: numpy, scipy, matplotlib, heapdict, mmh3 

Install using:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip3 install heapdict mmh3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### After installing all dependencies, running 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pip3 install mimircache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Compatibility

*mimircache only support Python3 and 64bit platform*

 

Alternative using docker
------------------------

As an alternative, you can using mimircache in docker container,

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
docker run --rm -v $(pwd):/opt/mimircache/user/ 1a1a11a/mimircache python3 /opt/mimircache/user/YOUR_PYTHON_FILE.py 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*As a reminder, please use absolute path(/opt/mimircache/user/YOUDATA) when
using container.*

 

mimircache Tutorial
-------------------

[Check here for tutorial](http://mimircacheemory.readthedocs.io/en/latest/)
