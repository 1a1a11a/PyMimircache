.. _API:

API
===
The architecture of mimircache is shown in the diagram below, it contains three parts, profiler, cache and cacheReader, which provides the detailed functions. cachecow is an upper level API that provides most common basic operations. To perform detailed operations, you will need to deal with the three components, so check the API below.

.. image::  images/architecture.png
    :width: 100%

.. toctree::
        :maxdepth: 3

        API/cachecow
        API/profiler
        API/profiler/cLRUProfiler
        API/profiler/cGeneralProfiler
        API/profiler/cHeatmap
        API/profiler/pyHeatmap
        API/profiler/pyGeneralProfiler
        API/profiler/twoDPlots
        API/cache
        API/cacheReader
