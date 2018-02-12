.. _advanced_usages:

Advanced Usages
===============

PyMimircache and its components
-------------------------------
Current version of PyMimircache is composed of three main components.

The first one is cache, which simulates corresponding cache replacement algorithm.

the second one is cacheReader, which provides all the necessary functions for reading and examing trace data file.

Most important of all, the third component is profilers, which extract data for profiling.

Currently, we have three kinds of profilers, the first one is LRU profiler, specially tailored for LRU;
the second one is a general profiler for profiling all non-LRU cache replacement algorithms;
the third profiler is a heatmap plot engine, which currently supports a variety of heatmaps.
LRUProfiler is in C, so it is pretty fast.
The other two profilers have corresponding C implementation (cGeneralProfiler and cHeatmap) used for caches available in C.

Each component has more functionality than described in tutorial, read the source code or raise a new issue in github if you want to know more or have questions.


.. _create_new_cache_reader:

Write your own cacheReader
--------------------------

Writing your own cacheReader is not difficult, just inherit abstractCacheReader.py.
Here is an example::

    from PyMimircache.cacheReader.abstractReader import AbstractReader

    class PlainReader(AbstractReader):
        """
        PlainReader class

        """
        all = ["read_one_req", "copy", "get_params"]

        def __init__(self, file_loc, data_type='c', open_c_reader=True, **kwargs):
            """
            :param file_loc:            location of the file
            :param data_type:           type of data, can be "l" for int/long, "c" for string
            :param open_c_reader:       bool for whether open reader in C backend
            :param kwargs:              not used now
            """

            super(PlainReader, self).__init__(file_loc, data_type, open_c_reader=open_c_reader, lock=kwargs.get("lock"))
            self.trace_file = open(file_loc, 'rb')
            if ALLOW_C_MIMIRCACHE and open_c_reader:
                self.c_reader = c_cacheReader.setup_reader(file_loc, 'p', data_type=data_type, block_unit_size=0)

        def read_one_req(self):
            """
            read one request
            :return: a request
            """
            super().read_one_req()

            line = self.trace_file.readline().decode()
            while line and len(line.strip()) == 0:
                line = self.trace_file.readline().decode()

            if line and len(line.strip()):
                return line.strip()
            else:
                return None

        def read_complete_req(self):
            """
            read all information about one record, which is the same as read_one_req for PlainReader
            """

            return self.read_one_req()

        def skip_n_req(self, n):
            """
            skip N requests from current position

            :param n: the number of requests to skip
            """

            for i in range(n):
                self.read_one_req()


        def copy(self, open_c_reader=False):
            """
            reader a deep copy of current reader with everything reset to initial state,
            the returned reader should not interfere with current reader

            :param open_c_reader: whether open_c_reader_or_not, default not open
            :return: a copied reader
            """

            return PlainReader(self.file_loc, data_type=self.data_type, open_c_reader=open_c_reader, lock=self.lock)

        def get_params(self):
            """
            return all the parameters for this reader instance in a dictionary
            :return: a dictionary containing all parameters
            """

            return {
                "file_loc": self.file_loc,
                "data_type": self.data_type,
                "open_c_reader": self.open_c_reader
            }

        def __next__(self):  # Python 3
            super().__next__()
            element = self.trace_file.readline().strip()
            if element:
                return element
            else:
                raise StopIteration

        def __repr__(self):
            return "PlainReader of trace {}".format(self.file_loc)


After writing your own cache reader, you can use it on generalProfiler and heatmap, for example:

    >>> reader = vscsiCacheReader(PATH/TO/DATA)
    >>> p = generalProfiler(reader, "FIFO", cache_size, bin_size=bin_size, num_of_process=8)

the first parameter is the cacheReader object of your own, the second is the cache replacement algorithm,
the third parameter is cache size, the fourth parameter is bin_size, and it can be omitted, in which case, the default bin_size if cache_size/100.


    >>> hm = heatmap()
    >>> hm.heatmap(reader, 'r', TIME_INTERVAL, "hit_rate_start_time_end_time", cache_size=CACHE_SIZE)



.. _create_new_cache_replacement_algorithms:

Write your own cache replacement algorithm
------------------------------------------

Writing your own cache in Python is not difficult, just inherit Cache.py::

    from PyMimircache.cache.abstractCache import Cache

    class LRU(Cache):
        """
        LRU class for simulating a LRU cache

        """

        def __init__(self, cache_size, **kwargs):

            super().__init__(cache_size, **kwargs)
            self.cacheline_dict = OrderedDict()

        def has(self, req_id, **kwargs):
            """
            check whether the given id in the cache or not

            :return: whether the given element is in the cache
            """
            if req_id in self.cacheline_dict:
                return True
            else:
                return False

        def _update(self, req_item, **kwargs):
            """ the given element is in the cache,
            now update cache metadata and its content

            :param **kwargs:
            :param req_item:
            :return: None
            """

            req_id = req_item
            if isinstance(req_item, Req):
                req_id = req_item.item_id

            self.cacheline_dict.move_to_end(req_id)

        def _insert(self, req_item, **kwargs):
            """
            the given element is not in the cache, now insert it into cache
            :param **kwargs:
            :param req_item:
            :return: evicted element or None
            """

            req_id = req_item
            if isinstance(req_item, Req):
                req_id = req_item.item_id

            self.cacheline_dict[req_id] = True

        def evict(self, **kwargs):
            """
            evict one cacheline from the cache

            :param **kwargs:
            :return: id of evicted cacheline
            """

            req_id = self.cacheline_dict.popitem(last=False)
            return req_id

        def access(self, req_item, **kwargs):
            """
            request access cache, it updates cache metadata,
            it is the underlying method for both get and put

            :param **kwargs:
            :param req_item: the request from the trace, it can be in the cache, or not
            :return: None
            """

            req_id = req_item
            if isinstance(req_item, Req):
                req_id = req_item.item_id

            if self.has(req_id):
                self._update(req_item)
                return True
            else:
                self._insert(req_item)
                if len(self.cacheline_dict) > self.cache_size:
                    self.evict()
                return False

        def __len__(self):
            return len(self.cacheline_dict)

        def __repr__(self):
            return "LRU cache of size: {}, current size: {}, {}".\
                format(self.cache_size, len(self.cacheline_dict), super().__repr__())

The usage of new cache replacement algorithm is the same as the one in last section, just replace the algorithm string
with your algorithm class.

Profiling in python is only applicable on small data set, so you can use it to verify your idea, when running on large
dataset, we suggested implemented the algorithms in C, check the source code to find out how to implement in C.



